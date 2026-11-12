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

import fs from 'node:fs';
import https from 'node:https';
import path from 'node:path';

export function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function escapeMd(str) {
  return String(str ?? '')
    .replace(/\|/g, '\\|')
    .replace(/\r?\n/g, ' ')
    .trim();
}

export function statusIcon(status) {
  switch (status) {
    case 'PASS':
      return '✅ PASS';
    case 'FALLBACK_CPU':
      return '⚠️ FALLBACK_CPU';
    case 'FAIL':
      return '❌ FAIL';
    case 'SKIP':
      return '⏭️ SKIP';
    default:
      return status;
  }
}

export function loadState(statusPath) {
  if (fs.existsSync(statusPath)) {
    return JSON.parse(fs.readFileSync(statusPath, 'utf8'));
  }
  return {
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    runUrl: '',
    checkRunId: null,
    checkRunUrl: '',
    totalPlanned: 0,
    currentCase: null,
    environment: null,
    packageVersions: {},
    results: [],
  };
}

export function saveState(statusPath, state) {
  state.updatedAt = new Date().toISOString();
  fs.mkdirSync(path.dirname(statusPath), { recursive: true });
  fs.writeFileSync(statusPath, `${JSON.stringify(state, null, 2)}\n`);
}

export function summarizeCounts(results) {
  const counts = { total: results.length, pass: 0, fallback: 0, fail: 0, skip: 0 };
  for (const r of results) {
    if (r.status === 'PASS') counts.pass++;
    else if (r.status === 'FALLBACK_CPU') counts.fallback++;
    else if (r.status === 'FAIL') counts.fail++;
    else if (r.status === 'SKIP') counts.skip++;
  }
  return counts;
}

function githubApiRequest(method, apiPath, body) {
  const token = process.env.GITHUB_TOKEN;
  if (!token) return Promise.resolve(null);
  return new Promise((resolve) => {
    const payload = body ? JSON.stringify(body) : '';
    const req = https.request(
      {
        hostname: 'api.github.com',
        path: apiPath,
        method,
        headers: {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${token}`,
          'User-Agent': 'mediapipe-samples-web-release-validator',
          'X-GitHub-Api-Version': '2022-11-28',
          ...(payload
            ? {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload),
              }
            : {}),
        },
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8');
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(JSON.parse(text));
            } catch {
              resolve(null);
            }
          } else {
            console.warn(`GitHub Check Run API ${method} ${apiPath} returned HTTP ${res.statusCode}: ${text.slice(0, 200)}`);
            resolve(null);
          }
        });
      }
    );
    req.on('error', (err) => {
      console.warn(`GitHub Check Run API error: ${err.message}`);
      resolve(null);
    });
    if (payload) req.write(payload);
    req.end();
  });
}

function formatProgressBar(completed, total) {
  if (!total || total <= 0) return '';
  const width = 20;
  const filled = Math.min(width, Math.round((completed / total) * width));
  const pct = Math.round((completed / total) * 100);
  return `\`[${'#'.repeat(filled)}${'-'.repeat(width - filled)}]\` **${completed}/${total}** (${pct}%)`;
}

function buildCheckRunOutput(state, { completed = false } = {}) {
  const counts = summarizeCounts(state.results);
  const total = state.totalPlanned || counts.total;
  const progress = formatProgressBar(counts.total, total);

  const summaryLines = [
    progress ? `Progress: ${progress}` : `Completed: **${counts.total}** checks`,
    '',
    `- **Passed:** ${counts.pass}`,
    `- **Fallback to CPU:** ${counts.fallback}`,
    `- **Failed:** ${counts.fail}`,
    `- **Skipped:** ${counts.skip}`,
  ];
  if (!completed && state.currentCase) {
    const c = state.currentCase;
    summaryLines.push(
      `- **Currently running:** \`${c.domain}/${c.task}\` | model=\`${c.modelId}\` | mode=\`${c.mode}\` | delegate=\`${c.delegate}\``
    );
  }
  if (state.runUrl) {
    summaryLines.push(`- **Workflow Run & Artifacts:** ${state.runUrl}`);
  }
  if (state.environment) {
    const env = state.environment;
    summaryLines.push(
      `- **Browser:** \`${env.browserVersion || 'Chrome'}\` | **WebGL2:** \`${env.webglRenderer || 'unknown'}\` | **WebGPU:** \`${env.webgpuAdapter || 'unavailable'}\``
    );
  }
  if (Object.keys(state.packageVersions || {}).length > 0) {
    const pkgs = Object.entries(state.packageVersions)
      .map(([k, v]) => `\`${k}@${v}\``)
      .join(', ');
    summaryLines.push(`- **SDK Packages:** ${pkgs}`);
  }

  // Keep Check Run text well under GitHub's 65,535-char limit.
  const rowsHeader = [
    '| Task | Model | Mode | Req | Actual | Status | Time | Details |',
    '|---|---|---|---|---|---|---|---|',
  ];
  const ordered = [
    ...state.results.filter((r) => r.status === 'FAIL'),
    ...state.results.filter((r) => r.status === 'FALLBACK_CPU'),
    ...state.results.filter((r) => r.status !== 'FAIL' && r.status !== 'FALLBACK_CPU').reverse(),
  ];
  const lines = [...rowsHeader];
  let charBudget = 55000;
  let omitted = 0;
  for (const r of ordered) {
    const row =
      `| ${escapeMd(`${r.domain}/${r.task}`)} ` +
      `| ${escapeMd(r.modelId)} ` +
      `| ${escapeMd(r.mode)} ` +
      `| ${escapeMd(r.requestedDelegate)} ` +
      `| ${escapeMd(r.actualDelegate || '-')} ` +
      `| ${statusIcon(r.status)} ` +
      `| ${Math.round(r.durationMs || 0)}ms ` +
      `| ${escapeMd((r.error || r.details || '').slice(0, 140))} |`;
    if (charBudget - row.length < 0) {
      omitted++;
      continue;
    }
    lines.push(row);
    charBudget -= row.length + 1;
  }
  if (omitted > 0) {
    lines.push('', `_...and ${omitted} earlier passing checks (see full report artifact for all rows & screenshots)._`);
  }

  return {
    title: completed
      ? `Release validation finished: ${counts.pass} pass, ${counts.fallback} fallback, ${counts.fail} fail`
      : `Validating ${counts.total}/${total} (${counts.fail} failed, ${counts.fallback} fallback)`,
    summary: summaryLines.join('\n'),
    text: lines.join('\n'),
  };
}

export async function createLiveCheckRun(state) {
  const repo = process.env.GITHUB_REPOSITORY;
  const sha = process.env.RELEASE_HEAD_SHA || process.env.GITHUB_SHA;
  if (!repo || !sha || !process.env.GITHUB_TOKEN) return null;

  const created = await githubApiRequest('POST', `/repos/${repo}/check-runs`, {
    name: 'Release Matrix Validation',
    head_sha: sha,
    status: 'in_progress',
    started_at: state.startedAt,
    details_url: state.runUrl || undefined,
    output: buildCheckRunOutput(state),
  });
  if (created?.id) {
    state.checkRunId = created.id;
    state.checkRunUrl = created.html_url || '';
  }
  return created;
}

export async function updateLiveCheckRun(state, { completed = false, conclusion = 'success' } = {}) {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!repo || !state.checkRunId || !process.env.GITHUB_TOKEN) return null;

  const payload = {
    status: completed ? 'completed' : 'in_progress',
    ...(completed ? { conclusion, completed_at: new Date().toISOString() } : {}),
    output: buildCheckRunOutput(state, { completed }),
  };
  return githubApiRequest('PATCH', `/repos/${repo}/check-runs/${state.checkRunId}`, payload);
}

export function formatMarkdownTable(results) {
  const lines = [
    '| # | Task | Model | Mode | Req Delegate | Actual Delegate | Status | Duration | Inference | Details | Screenshot |',
    '|---|---|---|---|---|---|---|---|---|---|---|',
  ];
  results.forEach((r, idx) => {
    const shot = r.screenshot ? `\`${r.screenshot}\`` : '-';
    const detail = escapeMd((r.error || r.details || '').slice(0, 200));
    lines.push(
      `| ${idx + 1} | ${escapeMd(`${r.domain}/${r.task}`)} | ${escapeMd(r.modelId)} | ${escapeMd(r.mode)} | ` +
        `${escapeMd(r.requestedDelegate)} | ${escapeMd(r.actualDelegate || '-')} | ${statusIcon(r.status)} | ` +
        `${Math.round(r.durationMs || 0)}ms | ${escapeMd(r.inferenceTimeText || '-')} | ${detail} | ${shot} |`
    );
  });
  return lines.join('\n');
}

export function appendStepSummary(markdown) {
  const summaryFile = process.env.GITHUB_STEP_SUMMARY;
  if (!summaryFile) return;
  fs.appendFileSync(summaryFile, `${markdown}\n\n`);
}

export function writeReportFiles(reportDir, state) {
  fs.mkdirSync(reportDir, { recursive: true });
  const counts = summarizeCounts(state.results);

  const mdLines = [
    '# MediaPipe Web Release Validation Report',
    '',
    `- **Started:** ${state.startedAt}`,
    `- **Updated:** ${state.updatedAt}`,
    state.runUrl ? `- **Workflow Run:** ${state.runUrl}` : '',
    state.checkRunUrl ? `- **Live Check Run:** ${state.checkRunUrl}` : '',
    state.environment
      ? `- **Browser:** ${state.environment.browserVersion} | **WebGL2:** ${state.environment.webglRenderer} | **WebGPU:** ${state.environment.webgpuAdapter}`
      : '',
    '',
    '## Summary',
    `- Total checks: **${counts.total}**`,
    `- Passed: **${counts.pass}**`,
    `- Fallback to CPU: **${counts.fallback}**`,
    `- Failed: **${counts.fail}**`,
    `- Skipped: **${counts.skip}**`,
    '',
    '## Installed `@mediapipe/tasks-*` Packages',
    ...Object.entries(state.packageVersions || {}).map(([name, ver]) => `- \`${name}\`: \`${ver}\``),
    '',
    '## Results Matrix',
    formatMarkdownTable(state.results),
  ].filter((line) => line !== undefined);

  fs.writeFileSync(path.join(reportDir, 'report.md'), `${mdLines.join('\n')}\n`);

  const rowsHtml = state.results
    .map((r, idx) => {
      const statusClass = r.status.toLowerCase().replace(/[^a-z0-9]+/g, '-');
      const shotHtml = r.screenshot
        ? `<a href="${escapeHtml(r.screenshot)}" target="_blank"><img src="${escapeHtml(r.screenshot)}" loading="lazy" alt="${escapeHtml(r.id)}" /></a>`
        : '-';
      return `<tr data-status="${escapeHtml(r.status)}" data-domain="${escapeHtml(r.domain)}">
        <td>${idx + 1}</td>
        <td><code>${escapeHtml(r.domain)}/${escapeHtml(r.task)}</code></td>
        <td><code>${escapeHtml(r.modelId)}</code></td>
        <td><code>${escapeHtml(r.mode)}</code></td>
        <td>${escapeHtml(r.requestedDelegate)}</td>
        <td>${escapeHtml(r.actualDelegate || '-')}</td>
        <td><span class="badge badge-${statusClass}">${escapeHtml(r.status)}</span></td>
        <td>${Math.round(r.durationMs || 0)} ms</td>
        <td>${escapeHtml(r.inferenceTimeText || '-')}</td>
        <td class="details">${escapeHtml(r.error || r.details || '')}</td>
        <td class="thumb">${shotHtml}</td>
      </tr>`;
    })
    .join('\n');

  const pkgBadges = Object.entries(state.packageVersions || {})
    .map(([k, v]) => `<span class="pkg-pill">${escapeHtml(k)}@${escapeHtml(v)}</span>`)
    .join(' ');

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MediaPipe Web Release Validation Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }
    h1 { margin: 0 0 8px; }
    .meta { color: #475569; font-size: 14px; margin-bottom: 16px; }
    .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 18px; min-width: 120px; }
    .card .num { font-size: 24px; font-weight: 700; }
    .card .lbl { font-size: 12px; color: #64748b; text-transform: uppercase; }
    .pkg-pill { display: inline-block; background: #e2e8f0; border-radius: 4px; padding: 2px 8px; font-family: monospace; font-size: 12px; margin: 2px; }
    .filters { margin: 16px 0; display: flex; gap: 8px; flex-wrap: wrap; }
    .filters button { padding: 6px 12px; border: 1px solid #cbd5e1; background: #fff; border-radius: 6px; cursor: pointer; font-weight: 600; }
    .filters button.active { background: #0f172a; color: #fff; border-color: #0f172a; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0; font-size: 13px; }
    th, td { border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f1f5f9; position: sticky; top: 0; }
    .badge { padding: 2px 8px; border-radius: 999px; font-weight: 700; font-size: 11px; display: inline-block; }
    .badge-pass { background: #dcfce7; color: #166534; }
    .badge-fallback-cpu { background: #fef3c7; color: #92400e; }
    .badge-fail { background: #fee2e2; color: #991b1b; }
    .badge-skip { background: #e2e8f0; color: #475569; }
    .details { max-width: 340px; word-break: break-word; }
    .thumb img { width: 180px; height: auto; border: 1px solid #cbd5e1; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>MediaPipe Web Release Validation Report</h1>
  <div class="meta">
    <div>Started: ${escapeHtml(state.startedAt)} | Updated: ${escapeHtml(state.updatedAt)}</div>
    ${state.environment ? `<div>Browser: <code>${escapeHtml(state.environment.browserVersion)}</code> | WebGL2: <code>${escapeHtml(state.environment.webglRenderer)}</code> | WebGPU: <code>${escapeHtml(state.environment.webgpuAdapter)}</code></div>` : ''}
    <div style="margin-top:6px;">${pkgBadges}</div>
  </div>
  <div class="cards">
    <div class="card"><div class="num">${counts.total}</div><div class="lbl">Total</div></div>
    <div class="card"><div class="num" style="color:#166534">${counts.pass}</div><div class="lbl">Pass</div></div>
    <div class="card"><div class="num" style="color:#92400e">${counts.fallback}</div><div class="lbl">Fallback CPU</div></div>
    <div class="card"><div class="num" style="color:#991b1b">${counts.fail}</div><div class="lbl">Fail</div></div>
    <div class="card"><div class="num" style="color:#475569">${counts.skip}</div><div class="lbl">Skip</div></div>
  </div>
  <div class="filters">
    <button class="active" onclick="filterRows('ALL', this)">All (${counts.total})</button>
    <button onclick="filterRows('PASS', this)">Pass (${counts.pass})</button>
    <button onclick="filterRows('FALLBACK_CPU', this)">Fallback CPU (${counts.fallback})</button>
    <button onclick="filterRows('FAIL', this)">Fail (${counts.fail})</button>
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Task</th><th>Model</th><th>Mode</th><th>Req</th><th>Actual</th><th>Status</th><th>Duration</th><th>Inference</th><th>Details</th><th>Screenshot</th>
      </tr>
    </thead>
    <tbody>
      ${rowsHtml}
    </tbody>
  </table>
  <script>
    function filterRows(status, btn) {
      document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('tbody tr').forEach(tr => {
        tr.style.display = (status === 'ALL' || tr.dataset.status === status) ? '' : 'none';
      });
    }
  </script>
</body>
</html>`;

  fs.writeFileSync(path.join(reportDir, 'index.html'), html);
}
