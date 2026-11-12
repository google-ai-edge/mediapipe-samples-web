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
const CORE_TASKS_PACKAGES = ['@mediapipe/tasks-vision', '@mediapipe/tasks-text', '@mediapipe/tasks-audio'];
const DEPENDENCY_SECTIONS = ['dependencies', 'devDependencies', 'optionalDependencies'];
const WASM_COPY_TARGETS = [
  '@mediapipe/tasks-retrieval',
  '@mediapipe/tasks-vision',
  '@mediapipe/tasks-audio',
  '@mediapipe/tasks-text',
  '@mediapipe/tasks-decision',
];

/**
 * Validates that `rawPath` is a safe `gs://mediapipe-assets/<subpath>` directory URI.
 */
export function validateGcsPath(rawPath) {
  if (typeof rawPath !== 'string' || rawPath.trim() === '') {
    throw new Error('GCS path must be a non-empty string.');
  }
  if (rawPath !== rawPath.trim()) {
    throw new Error(`GCS path must not contain leading/trailing whitespace: ${JSON.stringify(rawPath)}`);
  }
  if (!rawPath.startsWith(ALLOWED_GCS_PREFIX)) {
    throw new Error(`GCS path must start with "${ALLOWED_GCS_PREFIX}". Received: ${rawPath}`);
  }
  const trimmed = rawPath.slice(ALLOWED_GCS_PREFIX.length).replace(/\/+$/, '');
  if (!trimmed) {
    throw new Error(`GCS path must specify a subdirectory under "${ALLOWED_GCS_PREFIX}".`);
  }
  const segmentPattern = /^[A-Za-z0-9._+-]+$/;
  for (const segment of trimmed.split('/')) {
    if (!segment || segment === '.' || segment === '..' || !segmentPattern.test(segment)) {
      throw new Error(`Invalid GCS path segment "${segment}".`);
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

function requestBuffer(urlString, headers = {}, redirectsRemaining = 5) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(urlString);
    const req = https.request(parsedUrl, { method: 'GET', headers }, (res) => {
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
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ statusCode, body: Buffer.concat(chunks) }));
    });
    req.on('error', reject);
    req.end();
  });
}

async function fetchWithRetry(urlString, headers = {}, attempts = 3) {
  let lastErr;
  for (let i = 1; i <= attempts; i++) {
    try {
      const res = await requestBuffer(urlString, headers);
      if (res.statusCode >= 200 && res.statusCode < 300) return res.body;
      if (res.statusCode === 401 || res.statusCode === 403) {
        throw new Error(
          `GCS returned HTTP ${res.statusCode} for ${urlString}. Configure GCP WIF or GCP_CREDENTIALS if private.`
        );
      }
      const msg = `HTTP ${res.statusCode} fetching ${urlString}`;
      if (!(res.statusCode === 429 || res.statusCode >= 500) || i === attempts) {
        throw new Error(msg);
      }
      lastErr = new Error(msg);
    } catch (err) {
      if (err.message.includes('HTTP 401') || err.message.includes('HTTP 403') || i === attempts) throw err;
      lastErr = err;
    }
    await sleep(500 * 2 ** (i - 1));
  }
  throw lastErr;
}

async function listGcsTarballs({ bucket, prefix }, accessToken) {
  const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
  const dirPrefix = `${prefix}/`;
  const tarballs = [];
  let pageToken = '';

  do {
    const params = new URLSearchParams({ prefix: dirPrefix, delimiter: '/' });
    if (pageToken) params.set('pageToken', pageToken);
    const url = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(bucket)}/o?${params.toString()}`;
    const body = await fetchWithRetry(url, headers);
    const data = JSON.parse(body.toString('utf8'));
    for (const item of Array.isArray(data.items) ? data.items : []) {
      if (!item.name || !item.name.startsWith(dirPrefix)) continue;
      const fileName = item.name.slice(dirPrefix.length);
      if (!fileName || fileName.includes('/') || !fileName.endsWith('.tgz')) continue;
      if (!/^[A-Za-z0-9._@-]+\.tgz$/.test(fileName)) {
        throw new Error(`Unsafe tarball filename from GCS: ${fileName}`);
      }
      tarballs.push({
        objectName: item.name,
        fileName,
        size: item.size ? Number(item.size) : null,
      });
    }
    pageToken = typeof data.nextPageToken === 'string' ? data.nextPageToken : '';
  } while (pageToken);

  return tarballs.sort((a, b) => a.fileName.localeCompare(b.fileName));
}

export function readTarballMetadata(tarballPath) {
  const entries = execFileSync('tar', ['-tzf', tarballPath], { encoding: 'utf8' })
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  const pkgEntry = entries.find((e) => e === 'package/package.json' || e === './package/package.json');
  if (!pkgEntry) {
    throw new Error(`Tarball ${path.basename(tarballPath)} is missing package/package.json.`);
  }
  const pkgJson = JSON.parse(execFileSync('tar', ['-xOzf', tarballPath, pkgEntry], { encoding: 'utf8' }));
  if (!pkgJson.name || !pkgJson.version) {
    throw new Error(`Invalid package.json in ${path.basename(tarballPath)}.`);
  }
  return {
    name: pkgJson.name,
    version: pkgJson.version,
    fileName: path.basename(tarballPath),
    tarballPath: path.resolve(tarballPath),
    sizeBytes: fs.statSync(tarballPath).size,
  };
}

export function collectStagedTarballs(stagedDir, repoPkgPath = path.resolve('package.json')) {
  const files = fs
    .readdirSync(stagedDir)
    .filter((f) => f.endsWith('.tgz'))
    .sort()
    .map((f) => path.join(stagedDir, f));
  if (files.length === 0) {
    throw new Error(`No .tgz files found in ${stagedDir}.`);
  }
  const byPackage = new Map();
  for (const file of files) {
    const meta = readTarballMetadata(file);
    if (!meta.name.startsWith('@mediapipe/tasks-')) {
      throw new Error(`Unexpected package ${meta.name} in ${meta.fileName}.`);
    }
    if (byPackage.has(meta.name)) {
      throw new Error(`Duplicate tarballs for ${meta.name}.`);
    }
    byPackage.set(meta.name, meta);
  }
  const repoPkg = JSON.parse(fs.readFileSync(repoPkgPath, 'utf8'));
  const required = new Set(CORE_TASKS_PACKAGES);
  for (const section of DEPENDENCY_SECTIONS) {
    for (const dep of Object.keys(repoPkg[section] || {})) {
      if (dep.startsWith('@mediapipe/tasks-')) required.add(dep);
    }
  }
  const missing = [...required].filter((p) => !byPackage.has(p));
  if (missing.length > 0) {
    throw new Error(`Missing required @mediapipe/tasks-* tarballs: ${missing.join(', ')}`);
  }
  return byPackage;
}

export function applyTarballOverrides(stagedMap, repoPkgPath = path.resolve('package.json')) {
  const pkg = JSON.parse(fs.readFileSync(repoPkgPath, 'utf8'));
  for (const section of DEPENDENCY_SECTIONS) {
    const deps = pkg[section];
    if (!deps || typeof deps !== 'object') continue;
    for (const depName of Object.keys(deps)) {
      if (!depName.startsWith('@mediapipe/tasks-')) continue;
      const staged = stagedMap.get(depName);
      if (!staged) throw new Error(`Missing staged tarball for ${section}.${depName}`);
      deps[depName] = `file:${staged.tarballPath}`;
    }
  }
  pkg.pnpm = pkg.pnpm && typeof pkg.pnpm === 'object' ? pkg.pnpm : {};
  pkg.pnpm.overrides = pkg.pnpm.overrides && typeof pkg.pnpm.overrides === 'object' ? pkg.pnpm.overrides : {};
  for (const [pkgName, staged] of stagedMap.entries()) {
    pkg.pnpm.overrides[pkgName] = `file:${staged.tarballPath}`;
  }
  fs.writeFileSync(repoPkgPath, `${JSON.stringify(pkg, null, 2)}\n`);
}

function collectTaskEntriesFromTree(roots) {
  const found = [];
  const visited = new Set();
  function visit(depsMap, trail) {
    if (!depsMap || typeof depsMap !== 'object') return;
    for (const [depKey, entry] of Object.entries(depsMap)) {
      if (!entry || typeof entry !== 'object') continue;
      const pkgName = entry.from || depKey;
      const key = `${entry.path || ''}::${pkgName}::${entry.resolved || ''}`;
      if (pkgName.startsWith('@mediapipe/tasks-')) {
        found.push({
          name: pkgName,
          version: entry.version,
          resolved: entry.resolved,
          pkgPath: entry.path,
          trail: [...trail, pkgName],
        });
      }
      if (visited.has(key)) continue;
      visited.add(key);
      for (const section of DEPENDENCY_SECTIONS) {
        if (entry[section]) visit(entry[section], [...trail, pkgName]);
      }
    }
  }
  for (const root of Array.isArray(roots) ? roots : [roots]) {
    if (!root || typeof root !== 'object') continue;
    for (const section of DEPENDENCY_SECTIONS) {
      if (root[section]) visit(root[section], [root.name || 'root']);
    }
  }
  return found;
}

export function verifyInstalledPackages({ stagedMap, repoDir = process.cwd(), pnpmListJson }) {
  const entries = collectTaskEntriesFromTree(JSON.parse(pnpmListJson));
  if (entries.length === 0) throw new Error('No @mediapipe/tasks-* packages found in pnpm dependency tree.');
  for (const entry of entries) {
    const expected = stagedMap.get(entry.name);
    if (!expected) throw new Error(`Unexpected @mediapipe/tasks-* dependency ${entry.name}@${entry.version}`);
    if (entry.version !== expected.version) {
      throw new Error(`Version mismatch for ${entry.name}: expected ${expected.version}, got ${entry.version}`);
    }
    if (!entry.resolved || !entry.resolved.startsWith('file:')) {
      throw new Error(`${entry.name} did not resolve from local tarball: ${entry.resolved}`);
    }
    const resolvedPath = path.resolve(repoDir, entry.resolved.slice('file:'.length));
    if (!fs.existsSync(resolvedPath) || fs.realpathSync(resolvedPath) !== fs.realpathSync(expected.tarballPath)) {
      throw new Error(`${entry.name} resolved to ${resolvedPath}, expected ${expected.tarballPath}`);
    }
  }
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
          throw new Error(`Stale/non-staged package in .pnpm/${storeEntry}: ${fullName}@${pkgJson.version}`);
        }
      }
    }
  }
}

export function readInstalledMediaPipeVersions(repoDir = process.cwd()) {
  const scopeDir = path.join(repoDir, 'node_modules', '@mediapipe');
  const versions = {};
  if (!fs.existsSync(scopeDir)) return versions;
  for (const dir of fs.readdirSync(scopeDir).sort()) {
    const pkgPath = path.join(scopeDir, dir, 'package.json');
    if (fs.existsSync(pkgPath)) {
      const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
      versions[pkg.name] = pkg.version;
    }
  }
  return versions;
}

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

export function verifyWasmAssets({ repoDir = process.cwd(), checkDist = false } = {}) {
  const publicWasm = path.join(repoDir, 'public', 'wasm');
  const distWasm = path.join(repoDir, 'dist', 'wasm');
  if (!fs.existsSync(publicWasm)) throw new Error(`Missing ${publicWasm}`);
  if (checkDist && !fs.existsSync(distWasm)) throw new Error(`Missing ${distWasm}`);

  let count = 0;
  for (const pkgName of WASM_COPY_TARGETS) {
    const srcWasm = path.join(repoDir, 'node_modules', ...pkgName.split('/'), 'wasm');
    if (!fs.existsSync(srcWasm)) throw new Error(`Missing wasm/ in ${pkgName}`);
    for (const file of fs.readdirSync(srcWasm)) {
      const srcFile = path.join(srcWasm, file);
      if (!fs.statSync(srcFile).isFile()) continue;
      const hash = sha256(srcFile);
      const pubFile = path.join(publicWasm, file);
      if (!fs.existsSync(pubFile) || sha256(pubFile) !== hash) {
        throw new Error(`public/wasm/${file} mismatch against ${pkgName}`);
      }
      if (checkDist) {
        const distFile = path.join(distWasm, file);
        if (!fs.existsSync(distFile) || sha256(distFile) !== hash) {
          throw new Error(`dist/wasm/${file} mismatch against ${pkgName}`);
        }
      }
      count++;
    }
  }
  return count;
}

function parseCliArgs(argv) {
  const [command, ...rest] = argv;
  const args = {};
  for (let i = 0; i < rest.length; i++) {
    const t = rest[i];
    if (!t.startsWith('--')) throw new Error(`Unexpected arg: ${t}`);
    const k = t.slice(2);
    const next = rest[i + 1];
    if (!next || next.startsWith('--')) args[k] = true;
    else {
      args[k] = next;
      i++;
    }
  }
  return { command, args };
}

async function cmdDownload(args) {
  const gcs = validateGcsPath(args['gcs-path'] || process.env.GCS_PATH);
  const outDir = path.resolve(args['out-dir'] || '');
  if (!args['out-dir']) throw new Error('--out-dir is required');
  const token = process.env.GCP_ACCESS_TOKEN || '';
  const items = await listGcsTarballs(gcs, token);
  if (items.length === 0) throw new Error(`No .tgz files found at ${gcs.canonicalUri}/`);
  fs.mkdirSync(outDir, { recursive: true });
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  for (const item of items) {
    const url = `https://storage.googleapis.com/download/storage/v1/b/${encodeURIComponent(gcs.bucket)}/o/${encodeURIComponent(item.objectName)}?alt=media`;
    const buf = await fetchWithRetry(url, headers);
    if (buf.length === 0 || (item.size !== null && buf.length !== item.size)) {
      throw new Error(`Corrupt download for ${item.fileName}`);
    }
    fs.writeFileSync(path.join(outDir, item.fileName), buf);
  }
  const stagedMap = collectStagedTarballs(outDir);
  console.log('Downloaded staged tarballs:');
  for (const meta of [...stagedMap.values()].sort((a, b) => a.name.localeCompare(b.name))) {
    console.log(`  - ${meta.fileName} (${meta.name}@${meta.version})`);
  }
}

function cmdSelfTest() {
  assert.equal(validateGcsPath('gs://mediapipe-assets/sdk_validation_staging/web/rc1/').prefix, 'sdk_validation_staging/web/rc1');
  for (const bad of ['', 'gs://other/foo', 'gs://mediapipe-assets/', 'gs://mediapipe-assets/../x', 'gs://mediapipe-assets/a;b']) {
    assert.throws(() => validateGcsPath(bad));
  }
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mp-release-selftest-'));
  try {
    const tarDir = path.join(tmp, 'tgz');
    const stage = path.join(tmp, 'stage');
    const app = path.join(tmp, 'app');
    fs.mkdirSync(tarDir);
    fs.mkdirSync(path.join(stage, 'package'), { recursive: true });
    fs.mkdirSync(app);
    for (const s of ['vision', 'text', 'audio', 'retrieval', 'decision']) {
      fs.writeFileSync(path.join(stage, 'package', 'package.json'), JSON.stringify({ name: `@mediapipe/tasks-${s}`, version: '9.9.9-rc.1' }));
      execFileSync('tar', ['-czf', path.join(tarDir, `mediapipe-tasks-${s}-9.9.9-rc.1.tgz`), '-C', stage, 'package']);
    }
    fs.writeFileSync(path.join(app, 'package.json'), JSON.stringify({
      name: 'mediapipe-od-ts',
      packageManager: 'pnpm@9.0.0',
      dependencies: { '@mediapipe/tasks-vision': '^1.2.2', '@mediapipe/tasks-text': '^1.2.2', '@mediapipe/tasks-audio': '^1.2.2', '@mediapipe/tasks-retrieval': '^1.2.2', '@mediapipe/tasks-decision': '^1.2.2' }
    }));
    const map = collectStagedTarballs(tarDir, path.join(app, 'package.json'));
    applyTarballOverrides(map, path.join(app, 'package.json'));
    execFileSync('pnpm', ['install', '--no-frozen-lockfile', '--ignore-scripts'], { cwd: app, stdio: 'pipe' });
    const pnpmListJson = execFileSync('pnpm', ['list', '@mediapipe/*', '--depth=Infinity', '--json'], { cwd: app, encoding: 'utf8' });
    verifyInstalledPackages({ stagedMap: map, repoDir: app, pnpmListJson });
    assert.equal(readInstalledMediaPipeVersions(app)['@mediapipe/tasks-vision'], '9.9.9-rc.1');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  console.log('release_validate/tarballs.mjs self-test passed.');
}

async function main() {
  const { command, args } = parseCliArgs(process.argv.slice(2));
  if (command === 'validate-gcs-path') {
    const res = validateGcsPath(args['gcs-path'] || process.env.GCS_PATH);
    console.log(`Validated GCS path: ${res.canonicalUri}`);
  } else if (command === 'download-tarballs') {
    await cmdDownload(args);
  } else if (command === 'apply-overrides') {
    const stagedDir = path.resolve(args['staged-dir']);
    const map = collectStagedTarballs(stagedDir);
    applyTarballOverrides(map);
  } else if (command === 'verify-installed') {
    const stagedDir = path.resolve(args['staged-dir']);
    const map = collectStagedTarballs(stagedDir);
    const pnpmListJson = execFileSync('pnpm', ['list', '@mediapipe/*', '--depth=Infinity', '--json'], { encoding: 'utf8' });
    verifyInstalledPackages({ stagedMap: map, pnpmListJson });
    console.log('Verified installed @mediapipe/tasks-* versions:', readInstalledMediaPipeVersions());
  } else if (command === 'verify-wasm') {
    const count = verifyWasmAssets({ checkDist: Boolean(args['check-dist']) });
    console.log(`Verified ${count} WASM asset files.`);
  } else if (command === 'self-test') {
    cmdSelfTest();
  } else {
    throw new Error(`Unknown command: ${command}`);
  }
}

const isEntry = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (isEntry) {
  main().catch((err) => {
    console.error(`ERROR: ${err?.message || err}`);
    process.exit(1);
  });
}
