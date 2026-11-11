/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';

const ALLOWED_GCS_BUCKET = 'mediapipe-assets';
const ALLOWED_GCS_PREFIX = `gs://${ALLOWED_GCS_BUCKET}/`;
const CORE_TASKS_PACKAGES = [
  '@mediapipe/tasks-vision',
  '@mediapipe/tasks-text',
  '@mediapipe/tasks-audio',
];
const DEPENDENCY_SECTIONS = ['dependencies', 'devDependencies', 'optionalDependencies'];
const WASM_COPY_TARGETS = [
  '@mediapipe/tasks-retrieval',
  '@mediapipe/tasks-vision',
  '@mediapipe/tasks-audio',
  '@mediapipe/tasks-text',
  '@mediapipe/tasks-decision',
];

/**
 * Validates that `gcsPath` points to a non-empty directory inside
 * `gs://mediapipe-assets/` without shell metacharacters or path traversal.
 */
export function validateGcsPath(rawPath) {
  if (typeof rawPath !== 'string' || rawPath.trim() === '') {
    throw new Error('GCS path must be a non-empty string.');
  }
  if (rawPath !== rawPath.trim()) {
    throw new Error(`GCS path must not contain leading or trailing whitespace: ${JSON.stringify(rawPath)}`);
  }
  if (!rawPath.startsWith(ALLOWED_GCS_PREFIX)) {
    throw new Error(`GCS path must start with "${ALLOWED_GCS_PREFIX}". Received: ${rawPath}`);
  }

  const trimmed = rawPath.slice(ALLOWED_GCS_PREFIX.length).replace(/\/+$/, '');
  if (!trimmed) {
    throw new Error(`GCS path must specify a subdirectory under "${ALLOWED_GCS_PREFIX}".`);
  }

  const segmentPattern = /^[A-Za-z0-9._+-]+$/;
  const segments = trimmed.split('/');
  for (const segment of segments) {
    if (!segment || segment === '.' || segment === '..' || !segmentPattern.test(segment)) {
      throw new Error(
        `GCS path contains an invalid path segment "${segment}". Only [A-Za-z0-9._+-] segments separated by single "/" are allowed.`
      );
    }
  }

  return {
    bucket: ALLOWED_GCS_BUCKET,
    prefix: trimmed,
    canonicalUri: `${ALLOWED_GCS_PREFIX}${trimmed}`,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(statusCode) {
  return statusCode === 429 || (statusCode >= 500 && statusCode <= 599);
}

function requestBuffer(urlString, headers = {}, redirectsRemaining = 5) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(urlString);
    const req = https.request(
      parsedUrl,
      { method: 'GET', headers },
      (res) => {
        const statusCode = res.statusCode ?? 0;
        if ([301, 302, 303, 307, 308].includes(statusCode) && res.headers.location) {
          res.resume();
          if (redirectsRemaining <= 0) {
            reject(new Error(`Too many redirects fetching ${urlString}`));
            return;
          }
          const nextUrl = new URL(res.headers.location, parsedUrl);
          const nextHeaders = { ...headers };
          if (nextUrl.origin !== parsedUrl.origin) {
            delete nextHeaders.Authorization;
          }
          requestBuffer(nextUrl.href, nextHeaders, redirectsRemaining - 1).then(resolve, reject);
          return;
        }

        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          resolve({ statusCode, body: Buffer.concat(chunks) });
        });
      }
    );
    req.on('error', reject);
    req.end();
  });
}

async function fetchWithRetry(urlString, headers = {}, attempts = 3) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const result = await requestBuffer(urlString, headers);
      if (result.statusCode >= 200 && result.statusCode < 300) {
        return result.body;
      }
      if (result.statusCode === 401 || result.statusCode === 403) {
        throw new Error(
          `GCS returned HTTP ${result.statusCode} for ${urlString}. ` +
            'If objects are not public-read, configure GCP_WORKLOAD_IDENTITY_PROVIDER/GCP_SERVICE_ACCOUNT ' +
            'or GCP_CREDENTIALS in repository settings.'
        );
      }
      const message = `HTTP ${result.statusCode} fetching ${urlString}: ${result.body.toString('utf8').slice(0, 400)}`;
      if (!isRetryableStatus(result.statusCode) || attempt === attempts) {
        throw new Error(message);
      }
      lastError = new Error(message);
    } catch (err) {
      if (err.message.includes('HTTP 401') || err.message.includes('HTTP 403') || attempt === attempts) {
        throw err;
      }
      lastError = err;
    }
    await sleep(500 * 2 ** (attempt - 1));
  }
  throw lastError;
}

async function listGcsTarballs({ bucket, prefix }, accessToken) {
  const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
  const dirPrefix = `${prefix}/`;
  const tarballs = [];
  let pageToken = '';

  do {
    const params = new URLSearchParams({
      prefix: dirPrefix,
      delimiter: '/',
    });
    if (pageToken) {
      params.set('pageToken', pageToken);
    }
    const listUrl = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(bucket)}/o?${params.toString()}`;
    const body = await fetchWithRetry(listUrl, headers);
    const data = JSON.parse(body.toString('utf8'));
    const items = Array.isArray(data.items) ? data.items : [];

    for (const item of items) {
      if (!item.name || !item.name.startsWith(dirPrefix)) {
        continue;
      }
      const fileName = item.name.slice(dirPrefix.length);
      if (!fileName || fileName.includes('/') || !fileName.endsWith('.tgz')) {
        continue;
      }
      if (!/^[A-Za-z0-9._@-]+\.tgz$/.test(fileName)) {
        throw new Error(`Refusing unsafe tarball filename from GCS: ${fileName}`);
      }
      tarballs.push({
        objectName: item.name,
        fileName,
        size: item.size ? Number(item.size) : null,
      });
    }

    pageToken = typeof data.nextPageToken === 'string' ? data.nextPageToken : '';
  } while (pageToken);

  tarballs.sort((a, b) => a.fileName.localeCompare(b.fileName));
  return tarballs;
}

/**
 * Validates tarball structure and extracts `{ name, version }` from
 * `package/package.json` inside the `.tgz` archive.
 */
export function readTarballMetadata(tarballPath) {
  const entriesRaw = execFileSync('tar', ['-tzf', tarballPath], { encoding: 'utf8' });
  const entries = entriesRaw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const pkgJsonEntry = entries.find(
    (entry) => entry === 'package/package.json' || entry === './package/package.json'
  );
  if (!pkgJsonEntry) {
    throw new Error(`Tarball ${path.basename(tarballPath)} does not contain package/package.json.`);
  }

  const pkgJsonRaw = execFileSync('tar', ['-xOzf', tarballPath, pkgJsonEntry], { encoding: 'utf8' });
  const pkgJson = JSON.parse(pkgJsonRaw);
  if (typeof pkgJson.name !== 'string' || !pkgJson.name || typeof pkgJson.version !== 'string' || !pkgJson.version) {
    throw new Error(`Tarball ${path.basename(tarballPath)} has invalid name or version in ${pkgJsonEntry}.`);
  }

  return {
    name: pkgJson.name,
    version: pkgJson.version,
    fileName: path.basename(tarballPath),
    tarballPath: path.resolve(tarballPath),
    sizeBytes: fs.statSync(tarballPath).size,
  };
}

/**
 * Reads all `.tgz` files in `stagedDir`, validates uniqueness, and verifies that
 * every required `@mediapipe/tasks-*` package is present.
 */
export function collectStagedTarballs(stagedDir, repoPkgJsonPath = path.resolve('package.json')) {
  if (!fs.existsSync(stagedDir)) {
    throw new Error(`Staged tarball directory does not exist: ${stagedDir}`);
  }
  const tgzFiles = fs
    .readdirSync(stagedDir)
    .filter((f) => f.endsWith('.tgz'))
    .sort()
    .map((f) => path.join(stagedDir, f));

  if (tgzFiles.length === 0) {
    throw new Error(`No .tgz tarballs found in ${stagedDir}.`);
  }

  const byPackage = new Map();
  for (const tgzPath of tgzFiles) {
    const meta = readTarballMetadata(tgzPath);
    if (!meta.name.startsWith('@mediapipe/tasks-')) {
      throw new Error(
        `Unexpected package "${meta.name}" in ${meta.fileName}; expected @mediapipe/tasks-* package.`
      );
    }
    if (byPackage.has(meta.name)) {
      const existing = byPackage.get(meta.name);
      throw new Error(
        `Duplicate tarballs found for ${meta.name}: ${existing.fileName} (${existing.version}) and ${meta.fileName} (${meta.version}).`
      );
    }
    byPackage.set(meta.name, meta);
  }

  const repoPkg = JSON.parse(fs.readFileSync(repoPkgJsonPath, 'utf8'));
  const required = new Set(CORE_TASKS_PACKAGES);
  for (const section of DEPENDENCY_SECTIONS) {
    const deps = repoPkg[section];
    if (!deps || typeof deps !== 'object') continue;
    for (const depName of Object.keys(deps)) {
      if (depName.startsWith('@mediapipe/tasks-')) {
        required.add(depName);
      }
    }
  }

  const missing = [...required].filter((pkgName) => !byPackage.has(pkgName)).sort();
  if (missing.length > 0) {
    throw new Error(
      `Missing required staged @mediapipe/tasks-* tarballs: ${missing.join(', ')}. ` +
        `Found: ${[...byPackage.keys()].sort().join(', ') || '(none)'}`
    );
  }

  return byPackage;
}

async function cmdDownloadTarballs(args) {
  const rawGcsPath = args['gcs-path'] || process.env.GCS_PATH;
  const outDir = path.resolve(args['out-dir'] || '');
  const repoPkgPath = path.resolve(args['package-json'] || 'package.json');
  if (!args['out-dir']) {
    throw new Error('--out-dir is required.');
  }

  const gcs = validateGcsPath(rawGcsPath);
  const accessToken = process.env.GCP_ACCESS_TOKEN || '';
  const authLabel = accessToken ? 'authenticated bearer token' : 'public-read HTTPS';
  console.log(`Listing staged .tgz tarballs in ${gcs.canonicalUri}/ (${authLabel})...`);

  const remoteTarballs = await listGcsTarballs(gcs, accessToken);
  if (remoteTarballs.length === 0) {
    throw new Error(`No .tgz files found at ${gcs.canonicalUri}/`);
  }

  fs.mkdirSync(outDir, { recursive: true });
  const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};

  for (const item of remoteTarballs) {
    const downloadUrl =
      `https://storage.googleapis.com/download/storage/v1/b/${encodeURIComponent(gcs.bucket)}` +
      `/o/${encodeURIComponent(item.objectName)}?alt=media`;
    const content = await fetchWithRetry(downloadUrl, headers);
    if (content.length === 0) {
      throw new Error(`Downloaded empty file for ${item.fileName}.`);
    }
    if (item.size !== null && content.length !== item.size) {
      throw new Error(
        `Size mismatch downloading ${item.fileName}: expected ${item.size} bytes, got ${content.length}.`
      );
    }
    fs.writeFileSync(path.join(outDir, item.fileName), content);
  }

  const stagedMap = collectStagedTarballs(outDir, repoPkgPath);
  console.log('\nDownloaded staged NPM tarballs:');
  for (const meta of [...stagedMap.values()].sort((a, b) => a.name.localeCompare(b.name))) {
    console.log(`  - ${meta.fileName}  ${meta.name}@${meta.version}  (${meta.sizeBytes} bytes)`);
  }
}

/**
 * Updates `package.json` so direct `@mediapipe/tasks-*` dependencies and
 * `pnpm.overrides` point to the staged `.tgz` tarballs.
 */
export function applyTarballOverrides(stagedMap, repoPkgJsonPath = path.resolve('package.json')) {
  const pkg = JSON.parse(fs.readFileSync(repoPkgJsonPath, 'utf8'));
  const directReplaced = [];

  for (const section of DEPENDENCY_SECTIONS) {
    const deps = pkg[section];
    if (!deps || typeof deps !== 'object') continue;
    for (const depName of Object.keys(deps)) {
      if (!depName.startsWith('@mediapipe/tasks-')) continue;
      const staged = stagedMap.get(depName);
      if (!staged) {
        throw new Error(`No staged tarball available for ${section}.${depName}`);
      }
      const prev = deps[depName];
      deps[depName] = `file:${staged.tarballPath}`;
      directReplaced.push({ section, name: depName, from: prev, to: deps[depName], version: staged.version });
    }
  }

  pkg.pnpm = pkg.pnpm && typeof pkg.pnpm === 'object' ? pkg.pnpm : {};
  pkg.pnpm.overrides = pkg.pnpm.overrides && typeof pkg.pnpm.overrides === 'object' ? pkg.pnpm.overrides : {};
  const overridesApplied = [];

  for (const [pkgName, staged] of [...stagedMap.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const fileSpec = `file:${staged.tarballPath}`;
    pkg.pnpm.overrides[pkgName] = fileSpec;
    overridesApplied.push({ name: pkgName, spec: fileSpec, version: staged.version });
  }

  fs.writeFileSync(repoPkgJsonPath, `${JSON.stringify(pkg, null, 2)}\n`);
  return { directReplaced, overridesApplied };
}

function cmdApplyOverrides(args) {
  if (!args['staged-dir']) {
    throw new Error('--staged-dir is required.');
  }
  const stagedDir = path.resolve(args['staged-dir']);
  const repoPkgPath = path.resolve(args['package-json'] || 'package.json');
  const stagedMap = collectStagedTarballs(stagedDir, repoPkgPath);
  const { directReplaced, overridesApplied } = applyTarballOverrides(stagedMap, repoPkgPath);

  console.log('Updated direct @mediapipe/tasks-* dependencies in package.json:');
  for (const entry of directReplaced) {
    console.log(`  - [${entry.section}] ${entry.name}: ${entry.from} -> ${entry.version} (${entry.to})`);
  }
  console.log('Configured pnpm.overrides for all staged tarballs:');
  for (const entry of overridesApplied) {
    console.log(`  - ${entry.name}@${entry.version} -> ${entry.spec}`);
  }
}

function collectTaskEntriesFromTree(roots, baseDir) {
  const found = [];
  const visitedNodes = new Set();

  function visitDeps(depsMap, trail) {
    if (!depsMap || typeof depsMap !== 'object') return;
    for (const [depKey, entry] of Object.entries(depsMap)) {
      if (!entry || typeof entry !== 'object') continue;
      const pkgName = entry.from || depKey;
      const nodeKey = `${entry.path || ''}::${pkgName}::${entry.resolved || ''}`;

      if (pkgName.startsWith('@mediapipe/tasks-')) {
        found.push({
          name: pkgName,
          version: entry.version,
          resolved: entry.resolved,
          pkgPath: entry.path,
          trail: [...trail, pkgName],
        });
      }

      if (visitedNodes.has(nodeKey)) continue;
      visitedNodes.add(nodeKey);

      for (const section of DEPENDENCY_SECTIONS) {
        if (entry[section]) {
          visitDeps(entry[section], [...trail, pkgName]);
        }
      }
    }
  }

  for (const root of Array.isArray(roots) ? roots : [roots]) {
    if (!root || typeof root !== 'object') continue;
    const rootPath = root.path || baseDir;
    for (const section of DEPENDENCY_SECTIONS) {
      if (root[section]) {
        visitDeps(root[section], [root.name || path.basename(rootPath)]);
      }
    }
  }

  return found;
}

/**
 * Verifies that every installed `@mediapipe/tasks-*` package (both direct and
 * transitive) resolves to the staged `.tgz` tarball and matches its version.
 */
export function verifyInstalledPackages({ stagedMap, repoDir = process.cwd(), pnpmListJson }) {
  const repoPkg = JSON.parse(fs.readFileSync(path.join(repoDir, 'package.json'), 'utf8'));
  const parsedRoots = JSON.parse(pnpmListJson);
  const treeEntries = collectTaskEntriesFromTree(parsedRoots, repoDir);

  if (treeEntries.length === 0) {
    throw new Error('No @mediapipe/tasks-* packages found in pnpm dependency tree.');
  }

  const verifiedDirect = new Set();

  for (const entry of treeEntries) {
    const expected = stagedMap.get(entry.name);
    if (!expected) {
      throw new Error(
        `Installed package ${entry.name}@${entry.version} (via ${entry.trail.join(' -> ')}) ` +
          'has no corresponding staged .tgz tarball.'
      );
    }
    if (entry.version !== expected.version) {
      throw new Error(
        `Version mismatch for ${entry.name} (${entry.trail.join(' -> ')}): ` +
          `expected ${expected.version}, found ${entry.version}.`
      );
    }
    if (!entry.resolved || !entry.resolved.startsWith('file:')) {
      throw new Error(
        `Package ${entry.name} (${entry.trail.join(' -> ')}) did not resolve from local tarball: ` +
          `resolved=${JSON.stringify(entry.resolved)}`
      );
    }
    const resolvedTarball = path.resolve(repoDir, entry.resolved.slice('file:'.length));
    if (!fs.existsSync(resolvedTarball) || fs.realpathSync(resolvedTarball) !== fs.realpathSync(expected.tarballPath)) {
      throw new Error(
        `Package ${entry.name} resolved to unexpected file ${resolvedTarball}; expected ${expected.tarballPath}.`
      );
    }
    if (!entry.pkgPath) {
      throw new Error(`Missing installed path for ${entry.name}.`);
    }
    const installedPkgJson = JSON.parse(fs.readFileSync(path.join(entry.pkgPath, 'package.json'), 'utf8'));
    if (installedPkgJson.name !== entry.name || installedPkgJson.version !== expected.version) {
      throw new Error(
        `On-disk package.json mismatch at ${entry.pkgPath}: ` +
          `got ${installedPkgJson.name}@${installedPkgJson.version}, expected ${entry.name}@${expected.version}.`
      );
    }
    verifiedDirect.add(entry.name);
  }

  // Verify direct node_modules/@mediapipe/tasks-* symlinks match repo package.json.
  for (const section of DEPENDENCY_SECTIONS) {
    const deps = repoPkg[section];
    if (!deps || typeof deps !== 'object') continue;
    for (const depName of Object.keys(deps)) {
      if (!depName.startsWith('@mediapipe/tasks-')) continue;
      const expected = stagedMap.get(depName);
      if (!expected) {
        throw new Error(`Missing staged tarball for declared dependency ${depName}.`);
      }
      const installedDir = path.join(repoDir, 'node_modules', ...depName.split('/'));
      const pkgJsonPath = path.join(installedDir, 'package.json');
      if (!fs.existsSync(pkgJsonPath)) {
        throw new Error(`Declared dependency ${depName} is not installed at ${pkgJsonPath}.`);
      }
      const installedPkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
      if (installedPkg.version !== expected.version) {
        throw new Error(
          `Direct dependency ${depName} has version ${installedPkg.version}, expected ${expected.version}.`
        );
      }
      if (!verifiedDirect.has(depName)) {
        throw new Error(`Declared dependency ${depName} was missing from pnpm list output.`);
      }
    }
  }

  // Ensure node_modules/@mediapipe/ has no unverified tasks-* packages.
  const mediapipeScopeDir = path.join(repoDir, 'node_modules', '@mediapipe');
  if (fs.existsSync(mediapipeScopeDir)) {
    for (const dirName of fs.readdirSync(mediapipeScopeDir)) {
      if (!dirName.startsWith('tasks-')) continue;
      const fullName = `@mediapipe/${dirName}`;
      const expected = stagedMap.get(fullName);
      if (!expected) {
        throw new Error(`Unexpected package ${fullName} present in node_modules/@mediapipe.`);
      }
      const pkgJson = JSON.parse(fs.readFileSync(path.join(mediapipeScopeDir, dirName, 'package.json'), 'utf8'));
      if (pkgJson.version !== expected.version) {
        throw new Error(`Version mismatch in ${fullName}/package.json: ${pkgJson.version} !== ${expected.version}.`);
      }
    }
  }

  // Ensure node_modules/.pnpm has no stale or registry-sourced @mediapipe/tasks-* versions.
  const pnpmStoreDir = path.join(repoDir, 'node_modules', '.pnpm');
  if (fs.existsSync(pnpmStoreDir)) {
    for (const storeEntry of fs.readdirSync(pnpmStoreDir)) {
      const scopeDir = path.join(pnpmStoreDir, storeEntry, 'node_modules', '@mediapipe');
      if (!fs.existsSync(scopeDir)) continue;
      for (const dirName of fs.readdirSync(scopeDir)) {
        if (!dirName.startsWith('tasks-')) continue;
        const fullName = `@mediapipe/${dirName}`;
        const expected = stagedMap.get(fullName);
        const pkgJson = JSON.parse(fs.readFileSync(path.join(scopeDir, dirName, 'package.json'), 'utf8'));
        if (!expected || pkgJson.version !== expected.version || !storeEntry.includes('@file+')) {
          throw new Error(
            `Found stale or non-staged package in node_modules/.pnpm/${storeEntry}: ${fullName}@${pkgJson.version}`
          );
        }
      }
    }
  }

  return treeEntries;
}

function cmdVerifyInstalled(args) {
  if (!args['staged-dir']) {
    throw new Error('--staged-dir is required.');
  }
  const stagedDir = path.resolve(args['staged-dir']);
  const repoDir = path.resolve(args['repo-dir'] || '.');
  const stagedMap = collectStagedTarballs(stagedDir, path.join(repoDir, 'package.json'));

  const pnpmListOutput = execFileSync('pnpm', ['list', '@mediapipe/*', '--depth=Infinity', '--json'], {
    cwd: repoDir,
    encoding: 'utf8',
  });
  const entries = verifyInstalledPackages({ stagedMap, repoDir, pnpmListJson: pnpmListOutput });

  const uniquePackages = new Map();
  for (const entry of entries) {
    uniquePackages.set(entry.name, entry);
  }
  console.log('Verified all installed @mediapipe/tasks-* packages match staged tarballs:');
  for (const [name, entry] of [...uniquePackages.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    console.log(`  - ${name}@${entry.version} (${entry.resolved})`);
  }
}

function fileSha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function cmdVerifyWasm(args) {
  const repoDir = path.resolve(args['repo-dir'] || '.');
  const checkDist = Boolean(args['check-dist']);
  const publicWasmDir = path.join(repoDir, 'public', 'wasm');
  const distWasmDir = path.join(repoDir, 'dist', 'wasm');

  if (!fs.existsSync(publicWasmDir)) {
    throw new Error(`Expected WASM output directory does not exist: ${publicWasmDir}`);
  }
  if (checkDist && !fs.existsSync(distWasmDir)) {
    throw new Error(`Expected built WASM directory does not exist: ${distWasmDir}`);
  }

  let verifiedFiles = 0;
  for (const pkgName of WASM_COPY_TARGETS) {
    const srcWasmDir = path.join(repoDir, 'node_modules', ...pkgName.split('/'), 'wasm');
    if (!fs.existsSync(srcWasmDir)) {
      throw new Error(`Missing wasm/ directory in installed package ${pkgName}: ${srcWasmDir}`);
    }
    const files = fs.readdirSync(srcWasmDir);
    const hasWasmBinary = files.some((f) => f.endsWith('.wasm'));
    const hasLoaderJs = files.some((f) => f.endsWith('.js') || f.endsWith('.mjs'));
    if (!hasWasmBinary || !hasLoaderJs) {
      throw new Error(`Package ${pkgName}/wasm is missing expected .wasm or .js files: ${files.join(', ')}`);
    }

    for (const file of files) {
      const srcFile = path.join(srcWasmDir, file);
      if (!fs.statSync(srcFile).isFile()) continue;
      const srcHash = fileSha256(srcFile);

      const publicFile = path.join(publicWasmDir, file);
      if (!fs.existsSync(publicFile) || fileSha256(publicFile) !== srcHash) {
        throw new Error(`public/wasm/${file} is missing or does not match ${pkgName}/wasm/${file}.`);
      }
      if (checkDist) {
        const distFile = path.join(distWasmDir, file);
        if (!fs.existsSync(distFile) || fileSha256(distFile) !== srcHash) {
          throw new Error(`dist/wasm/${file} is missing or does not match ${pkgName}/wasm/${file}.`);
        }
      }
      verifiedFiles++;
    }
  }

  console.log(
    `Verified ${verifiedFiles} WASM asset files in public/wasm${checkDist ? ' and dist/wasm' : ''} match installed @mediapipe/tasks-* packages.`
  );
}

function cmdSelfTest() {
  // 1. Test validateGcsPath security rules.
  const valid1 = validateGcsPath('gs://mediapipe-assets/sdk_validation_staging/web/candidate-1');
  assert.equal(valid1.bucket, 'mediapipe-assets');
  assert.equal(valid1.prefix, 'sdk_validation_staging/web/candidate-1');

  const valid2 = validateGcsPath('gs://mediapipe-assets/sdk_validation_staging/web/1.3.0_rc.1/');
  assert.equal(valid2.canonicalUri, 'gs://mediapipe-assets/sdk_validation_staging/web/1.3.0_rc.1');

  const invalidCases = [
    '',
    '   ',
    'https://storage.googleapis.com/mediapipe-assets/foo',
    'gs://other-bucket/sdk_validation_staging/web/rc1',
    'gs://mediapipe-assets/',
    'gs://mediapipe-assets/sdk_validation_staging/../secret',
    'gs://mediapipe-assets/sdk_validation_staging//web',
    'gs://mediapipe-assets/sdk_validation_staging/web/*.tgz',
    'gs://mediapipe-assets/sdk_validation_staging/web/rc1;rm -rf /',
    'gs://mediapipe-assets/sdk_validation_staging/web/$(whoami)',
    'gs://mediapipe-assets/sdk_validation_staging/web/rc1 | cat',
    'gs://mediapipe-assets/sdk_validation_staging/web/rc 1',
  ];
  for (const bad of invalidCases) {
    assert.throws(() => validateGcsPath(bad), `Expected validateGcsPath to reject: ${bad}`);
  }

  // 2. Test tarball inspection, override patching, and verification in a temp sandbox.
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'mp-validate-selftest-'));
  try {
    const tarDir = path.join(tmpRoot, 'tarballs');
    const stagePkgDir = path.join(tmpRoot, 'pkg-stage');
    const appDir = path.join(tmpRoot, 'app');
    fs.mkdirSync(tarDir, { recursive: true });
    fs.mkdirSync(path.join(stagePkgDir, 'package'), { recursive: true });
    fs.mkdirSync(appDir, { recursive: true });

    const allPackages = [
      { name: '@mediapipe/tasks-vision', short: 'vision', deps: {} },
      { name: '@mediapipe/tasks-text', short: 'text', deps: {} },
      { name: '@mediapipe/tasks-audio', short: 'audio', deps: {} },
      { name: '@mediapipe/tasks-retrieval', short: 'retrieval', deps: {} },
      { name: '@mediapipe/tasks-genai', short: 'genai', deps: {} },
      {
        name: '@mediapipe/tasks-decision',
        short: 'decision',
        deps: { '@mediapipe/tasks-vision': '^1.0.0', '@mediapipe/tasks-genai': '^1.0.0' },
      },
    ];

    for (const item of allPackages) {
      fs.writeFileSync(
        path.join(stagePkgDir, 'package', 'package.json'),
        JSON.stringify({ name: item.name, version: '9.9.9-rc.1', dependencies: item.deps }, null, 2)
      );
      fs.writeFileSync(path.join(stagePkgDir, 'package', 'index.js'), `export const pkg = "${item.name}";\n`);
      const tgzOut = path.join(tarDir, `mediapipe-tasks-${item.short}-9.9.9-rc.1.tgz`);
      execFileSync('tar', ['-czf', tgzOut, '-C', stagePkgDir, 'package']);
    }

    fs.writeFileSync(
      path.join(appDir, 'package.json'),
      JSON.stringify(
        {
          name: 'mediapipe-od-ts',
          private: true,
          packageManager: 'pnpm@9.0.0',
          dependencies: {
            '@mediapipe/tasks-audio': '^1.2.2',
            '@mediapipe/tasks-decision': '^1.2.2',
            '@mediapipe/tasks-retrieval': '^1.2.2',
            '@mediapipe/tasks-text': '^1.2.2',
            '@mediapipe/tasks-vision': '^1.2.2',
            'mediapipe-od-ts': 'link:',
          },
        },
        null,
        2
      )
    );

    const stagedMap = collectStagedTarballs(tarDir, path.join(appDir, 'package.json'));
    assert.equal(stagedMap.size, 6);

    // Ensure missing required package and duplicate tarball are rejected.
    const visionTgz = path.join(tarDir, 'mediapipe-tasks-vision-9.9.9-rc.1.tgz');
    const dupTgz = path.join(tarDir, 'mediapipe-tasks-vision-duplicate.tgz');
    fs.copyFileSync(visionTgz, dupTgz);
    assert.throws(() => collectStagedTarballs(tarDir, path.join(appDir, 'package.json')), /Duplicate tarballs/);
    fs.unlinkSync(dupTgz);

    const backupVision = path.join(tmpRoot, 'vision-bak.tgz');
    fs.renameSync(visionTgz, backupVision);
    assert.throws(
      () => collectStagedTarballs(tarDir, path.join(appDir, 'package.json')),
      /Missing required staged @mediapipe\/tasks-\* tarballs/
    );
    fs.renameSync(backupVision, visionTgz);

    applyTarballOverrides(stagedMap, path.join(appDir, 'package.json'));

    const patchedPkg = JSON.parse(fs.readFileSync(path.join(appDir, 'package.json'), 'utf8'));
    assert.equal(
      patchedPkg.dependencies['@mediapipe/tasks-vision'],
      `file:${path.join(tarDir, 'mediapipe-tasks-vision-9.9.9-rc.1.tgz')}`
    );
    assert.equal(patchedPkg.dependencies['@mediapipe/tasks-genai'], undefined);
    assert.equal(
      patchedPkg.pnpm.overrides['@mediapipe/tasks-genai'],
      `file:${path.join(tarDir, 'mediapipe-tasks-genai-9.9.9-rc.1.tgz')}`
    );

    execFileSync('pnpm', ['install', '--no-frozen-lockfile', '--ignore-scripts'], { cwd: appDir, stdio: 'pipe' });
    const pnpmListJson = execFileSync('pnpm', ['list', '@mediapipe/*', '--depth=Infinity', '--json'], {
      cwd: appDir,
      encoding: 'utf8',
    });
    const verified = verifyInstalledPackages({ stagedMap, repoDir: appDir, pnpmListJson });
    assert.ok(verified.some((entry) => entry.name === '@mediapipe/tasks-genai' && entry.version === '9.9.9-rc.1'));

    // Ensure version mismatch fails loudly.
    const visionPkgPath = path.join(appDir, 'node_modules', '@mediapipe', 'tasks-vision', 'package.json');
    const tampered = JSON.parse(fs.readFileSync(visionPkgPath, 'utf8'));
    tampered.version = '1.2.2';
    fs.writeFileSync(visionPkgPath, JSON.stringify(tampered));
    assert.throws(
      () => verifyInstalledPackages({ stagedMap, repoDir: appDir, pnpmListJson }),
      /Version mismatch|On-disk package\.json mismatch/
    );

    // Test WASM verification helper.
    for (const pkgName of WASM_COPY_TARGETS) {
      const short = pkgName.split('/')[1].replace('tasks-', '');
      const pkgWasmDir = path.join(appDir, 'node_modules', ...pkgName.split('/'), 'wasm');
      fs.mkdirSync(pkgWasmDir, { recursive: true });
      fs.writeFileSync(path.join(pkgWasmDir, `${short}_wasm_internal.js`), '// js\n');
      fs.writeFileSync(path.join(pkgWasmDir, `${short}_wasm_internal.wasm`), 'wasm');
      const pubWasmDir = path.join(appDir, 'public', 'wasm');
      const distWasmDir = path.join(appDir, 'dist', 'wasm');
      fs.mkdirSync(pubWasmDir, { recursive: true });
      fs.mkdirSync(distWasmDir, { recursive: true });
      fs.copyFileSync(path.join(pkgWasmDir, `${short}_wasm_internal.js`), path.join(pubWasmDir, `${short}_wasm_internal.js`));
      fs.copyFileSync(path.join(pkgWasmDir, `${short}_wasm_internal.wasm`), path.join(pubWasmDir, `${short}_wasm_internal.wasm`));
      fs.copyFileSync(path.join(pkgWasmDir, `${short}_wasm_internal.js`), path.join(distWasmDir, `${short}_wasm_internal.js`));
      fs.copyFileSync(path.join(pkgWasmDir, `${short}_wasm_internal.wasm`), path.join(distWasmDir, `${short}_wasm_internal.wasm`));
    }
    cmdVerifyWasm({ 'repo-dir': appDir, 'check-dist': true });
    fs.writeFileSync(path.join(appDir, 'dist', 'wasm', 'vision_wasm_internal.wasm'), 'corrupted');
    assert.throws(() => cmdVerifyWasm({ 'repo-dir': appDir, 'check-dist': true }), /does not match/);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }

  console.log('All validate_web_sdk.mjs self-tests passed.');
}

function parseCliArgs(argv) {
  const [command, ...rest] = argv;
  const args = {};
  for (let i = 0; i < rest.length; i++) {
    const token = rest[i];
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
    const key = token.slice(2);
    const next = rest[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      i++;
    }
  }
  return { command, args };
}

async function main() {
  const { command, args } = parseCliArgs(process.argv.slice(2));
  switch (command) {
    case 'validate-gcs-path': {
      const result = validateGcsPath(args['gcs-path'] || process.env.GCS_PATH);
      console.log(`Validated GCS path: ${result.canonicalUri}`);
      break;
    }
    case 'download-tarballs':
      await cmdDownloadTarballs(args);
      break;
    case 'apply-overrides':
      cmdApplyOverrides(args);
      break;
    case 'verify-installed':
      cmdVerifyInstalled(args);
      break;
    case 'verify-wasm':
      cmdVerifyWasm(args);
      break;
    case 'self-test':
      cmdSelfTest();
      break;
    default:
      throw new Error(
        `Unknown or missing command "${command || ''}". ` +
          'Expected one of: validate-gcs-path, download-tarballs, apply-overrides, verify-installed, verify-wasm, self-test'
      );
  }
}

const isEntrypoint = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (isEntrypoint) {
  main().catch((err) => {
    console.error(`ERROR: ${err?.message || err}`);
    process.exit(1);
  });
}
