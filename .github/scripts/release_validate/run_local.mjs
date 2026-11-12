#!/usr/bin/env node
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

/**
 * Runs the release validation on this machine with one command:
 *
 *   pnpm run validate:release
 *   pnpm run validate:release --tasks vision/object_detector,audio
 *   pnpm run validate:release --gpu software --skip-build
 *
 * It builds the app, checks that the build serves the Wasm files of the
 * installed @mediapipe/tasks-* packages, then runs runner.mjs step by step
 * (init, one process per domain, finalize) and writes the report to
 * --report-dir. The exit code is 0 only if every case passed.
 */

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { buildValidationMatrix, unknownTaskFilters } from './cases.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_DIR = path.resolve(SCRIPT_DIR, '..', '..', '..');
const RUNNER = path.join(SCRIPT_DIR, 'runner.mjs');
const TARBALLS = path.join(SCRIPT_DIR, 'tarballs.mjs');

// Cheap domains first, so their results are in the report before the
// embedding-model domains (which need a real GPU to finish in reasonable time).
const DOMAIN_ORDER = ['vision', 'text', 'audio', 'retrieval', 'decision'];

const HELP = `Usage: pnpm run validate:release [options]

Options:
  --tasks <list>         Comma-separated domains ("vision"), tasks
                         ("object_detector") or both ("audio/text_to_speech").
                         Default: all tasks (or $TASK_FILTER).
  --gpu <mode>           "hardware" uses this machine's GPU (default),
                         "software" forces the software renderer (SwiftShader).
  --require-hardware-gpu Stop before the first case unless Chrome has a real
                         GPU for both WebGL and WebGPU.
  --small-models-only    Skip the large models (ASR, TTS, EmbeddingGemma-2).
  --fail-on-fallback     Fail cases where a GPU request fell back to CPU.
  --skip-build           Validate the existing dist/ instead of rebuilding.
  --report-dir <dir>     Default: release-validation-report
  --user-data-dir <dir>  Chrome profile holding the downloaded models.
                         Default: <tmp>/mp-release-chrome-profile
  --port <port>          Port of the local preview server. Default: 4174.
                         Models are cached per port, so keep it the same.
  --help                 Show this text.
`;

const VALUE_OPTIONS = new Set(['tasks', 'gpu', 'report-dir', 'user-data-dir', 'port']);
const FLAG_OPTIONS = new Set(['require-hardware-gpu', 'small-models-only', 'fail-on-fallback', 'skip-build', 'help']);

export function parseOptions(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === '--') continue;
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const [name, inline] = token.slice(2).split(/=(.*)/s, 2);
    if (FLAG_OPTIONS.has(name)) {
      options[name] = true;
    } else if (VALUE_OPTIONS.has(name)) {
      const value = inline ?? argv[++i];
      if (value === undefined || value.startsWith('--')) throw new Error(`Option --${name} needs a value`);
      options[name] = value;
    } else {
      throw new Error(`Unknown option: --${name}`);
    }
  }
  if (options.gpu && !['hardware', 'software'].includes(options.gpu)) {
    throw new Error(`--gpu must be "hardware" or "software", got "${options.gpu}"`);
  }
  if (options.port && !/^\d+$/.test(options.port)) throw new Error(`--port must be a number, got "${options.port}"`);
  if (options['require-hardware-gpu'] && options.gpu === 'software') {
    throw new Error('--require-hardware-gpu cannot be combined with --gpu software');
  }
  return options;
}

/**
 * Why the probed Chrome (the `environment` of status.json) has no real GPU, or
 * '' if it has one. Chrome reports a software WebGPU adapter as a fallback
 * adapter, which the probe marks with "(software)".
 */
export function describeMissingHardwareGpu(environment) {
  const webgl = environment?.webglRenderer || 'unavailable';
  const webgpu = environment?.webgpuAdapter || 'unavailable';
  const software = /swiftshader|llvmpipe|\(software\)/i;
  const problems = [];
  if (webgl === 'unavailable' || software.test(webgl)) problems.push(`WebGL renderer: ${webgl}`);
  if (!environment?.hasWebGPU || software.test(webgpu)) problems.push(`WebGPU adapter: ${webgpu}`);
  return problems.join('; ');
}

/** Runs a command in the repository, streaming its output. Returns true if it exited with 0. */
function run(label, command, args, env, { shell = false } = {}) {
  console.log(`\n=== ${label} ===`);
  const result = spawnSync(command, args, { cwd: REPO_DIR, env, stdio: 'inherit', shell });
  if (result.error) console.error(`${label}: ${result.error.message}`);
  if (result.signal) {
    console.error(`${label}: interrupted (${result.signal})`);
    process.exit(130);
  }
  return result.status === 0;
}

function main() {
  let options;
  try {
    options = parseOptions(process.argv.slice(2));
  } catch (err) {
    console.error(`${err.message}\n\n${HELP}`);
    process.exit(2);
  }
  if (options.help) {
    console.log(HELP);
    return;
  }

  const taskFilter = options.tasks ?? process.env.TASK_FILTER ?? '';
  const includeLargeModels = options['small-models-only']
    ? false
    : (process.env.INCLUDE_LARGE_MODELS ?? 'true') !== 'false';
  const gpu = options.gpu ?? (options['require-hardware-gpu'] ? 'hardware' : (process.env.RELEASE_VALIDATION_GPU ?? 'hardware'));
  const reportDir = path.resolve(REPO_DIR, options['report-dir'] ?? 'release-validation-report');
  const userDataDir = path.resolve(options['user-data-dir'] ?? path.join(os.tmpdir(), 'mp-release-chrome-profile'));
  const port = options.port ?? '4174';

  const unknown = unknownTaskFilters(taskFilter);
  if (unknown.length > 0) {
    console.error(`Unknown task or domain in --tasks: ${unknown.join(', ')}`);
    process.exit(2);
  }

  const domains = DOMAIN_ORDER.map((domain) => ({
    domain,
    count: buildValidationMatrix({ domain, taskFilter, includeLargeModels }).length,
  })).filter((entry) => entry.count > 0);
  const total = domains.reduce((sum, entry) => sum + entry.count, 0);
  if (total === 0) {
    console.error(`No validation cases match --tasks "${taskFilter}".`);
    process.exit(2);
  }

  const env = {
    ...process.env,
    TASK_FILTER: taskFilter,
    INCLUDE_LARGE_MODELS: String(includeLargeModels),
    FAIL_ON_FALLBACK: options['fail-on-fallback'] ? 'true' : (process.env.FAIL_ON_FALLBACK ?? 'false'),
    // A case that ran without any output check must not count as a pass.
    FAIL_ON_UNVERIFIED: process.env.FAIL_ON_UNVERIFIED ?? 'true',
    RELEASE_VALIDATION_GPU: gpu,
  };
  // A local run reports to the terminal and the report directory only; with a
  // token in the environment the runner would post a check run to GitHub.
  delete env.GITHUB_TOKEN;

  console.log(
    `Release validation: ${total} cases in ${domains.map((d) => `${d.domain} (${d.count})`).join(', ')}; ` +
      `GPU: ${gpu}; large models: ${includeLargeModels ? 'yes' : 'no'}`
  );

  if (!options['skip-build']) {
    // Build with the package manager that started this script (pnpm by default).
    const packageManager = (process.env.npm_config_user_agent || '').split('/')[0] || 'pnpm';
    // On Windows the package managers are .cmd shims, which need a shell.
    if (!run('Build the app', packageManager, ['run', 'build'], env, { shell: process.platform === 'win32' })) {
      console.error('The build failed; nothing was validated.');
      process.exit(1);
    }
  }
  // The app must serve the Wasm files of the packages that are installed now;
  // a stale build would validate some other version.
  if (!run('Check the Wasm files of the build', process.execPath, [TARBALLS, 'verify-wasm', '--check-dist'], env)) {
    console.error(
      options['skip-build']
        ? 'dist/ does not match the installed packages; run again without --skip-build.'
        : 'The build does not match the installed packages; nothing was validated.'
    );
    process.exit(1);
  }

  const common = ['--report-dir', reportDir, '--user-data-dir', userDataDir];
  if (!run('Runner self-test', process.execPath, [RUNNER, 'self-test'], env)) process.exit(1);
  if (!run('Probe Chrome and start the report', process.execPath, [RUNNER, 'init', ...common], env)) process.exit(1);

  if (options['require-hardware-gpu']) {
    const status = JSON.parse(fs.readFileSync(path.join(reportDir, 'status.json'), 'utf8'));
    const missing = describeMissingHardwareGpu(status.environment);
    if (missing) {
      console.error(`\nChrome has no hardware GPU on this machine (${missing}). Nothing was validated.`);
      process.exit(1);
    }
    console.log(`Hardware GPU: ${status.environment.webglRenderer} | WebGPU: ${status.environment.webgpuAdapter}`);
  }

  // A domain that breaks does not stop the others.
  let crashed = false;
  for (const { domain, count } of domains) {
    const ok = run(
      `Validate ${domain} (${count} cases)`,
      process.execPath,
      [RUNNER, 'run-domain', '--domain', domain, ...common, '--port', port],
      env
    );
    if (!ok) crashed = true;
  }

  const passed = run('Write the report', process.execPath, [RUNNER, 'finalize', '--report-dir', reportDir], env);
  console.log(`\nReport: ${pathToFileURL(path.join(reportDir, 'index.html')).href}`);
  if (crashed) console.error('At least one domain did not run to the end; see the output above.');
  process.exit(passed && !crashed ? 0 : 1);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
