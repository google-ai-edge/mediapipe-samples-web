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
    case 'UNVERIFIED':
      return '❔ UNVERIFIED';
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
  // PASS and FALLBACK_CPU mean the output was verified; UNVERIFIED ran without any output assertion.
  const counts = { total: results.length, pass: 0, fallback: 0, fail: 0, unverified: 0, skip: 0 };
  for (const r of results) {
    if (r.status === 'PASS') counts.pass++;
    else if (r.status === 'FALLBACK_CPU') counts.fallback++;
    else if (r.status === 'FAIL') counts.fail++;
    else if (r.status === 'UNVERIFIED') counts.unverified++;
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
    `- **Passed (output verified):** ${counts.pass}`,
    `- **Fallback to CPU (output verified):** ${counts.fallback}`,
    `- **Failed:** ${counts.fail}`,
    `- **Unverified (ran, output not checked):** ${counts.unverified}`,
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
      `| ${escapeMd((r.error || r.verification?.summary || r.details || '').slice(0, 300))} |`;
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
      ? `Release validation finished: ${counts.pass} pass, ${counts.fallback} fallback, ${counts.fail} fail, ${counts.unverified} unverified`
      : `Validating ${counts.total}/${total} (${counts.fail} failed, ${counts.fallback} fallback, ${counts.unverified} unverified)`,
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
    '| # | Task | Model | Mode | Req Delegate | Actual Delegate | Status | Output checks | Duration | Inference | Details | Screenshot |',
    '|---|---|---|---|---|---|---|---|---|---|---|---|',
  ];
  results.forEach((r, idx) => {
    const shot = r.screenshot ? `\`${r.screenshot}\`` : '-';
    const detail = escapeMd((r.error || r.details || '').slice(0, 400));
    const outputChecks = escapeMd(r.verification?.summary || '-');
    lines.push(
      `| ${idx + 1} | ${escapeMd(`${r.domain}/${r.task}`)} | ${escapeMd(r.modelId)} | ${escapeMd(r.mode)} | ` +
        `${escapeMd(r.requestedDelegate)} | ${escapeMd(r.actualDelegate || '-')} | ${statusIcon(r.status)} | ${outputChecks} | ` +
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
    `- Passed (output verified): **${counts.pass}**`,
    `- Fallback to CPU (output verified): **${counts.fallback}**`,
    `- Failed: **${counts.fail}**`,
    `- Unverified (ran, output not checked): **${counts.unverified}**`,
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
      // Every output check with what was expected and what the task produced.
      const checks = r.verification?.checks || [];
      const checksHtml =
        checks.length > 0
          ? `<details${checks.some((c) => !c.pass) ? ' open' : ''}><summary>${escapeHtml(r.verification.summary)}</summary><ul>${checks
              .map(
                (c) =>
                  `<li class="${c.pass ? 'ok' : 'bad'}">${c.pass ? '&#10003;' : '&#10007;'} ${escapeHtml(c.name)}: expected ${escapeHtml(c.expected)}, got ${escapeHtml(c.actual)}</li>`
              )
              .join('')}</ul></details>`
          : escapeHtml(r.verification?.summary || '-');
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
        <td class="checks">${checksHtml}</td>
        <td class="details">${escapeHtml(r.error || r.details || '')}</td>
        <td class="thumb">${shotHtml}</td>
      </tr>`;
    })
    .join('\n');

  // Data for the review mode. "<" is escaped so the JSON cannot end the script element.
  const casesJson = JSON.stringify(
    state.results.map((r) => ({
      id: r.id,
      title: `${r.domain}/${r.task}`,
      modelId: r.modelId,
      mode: r.mode,
      requestedDelegate: r.requestedDelegate,
      actualDelegate: r.actualDelegate,
      status: r.status,
      durationMs: Math.round(r.durationMs || 0),
      inferenceTimeText: r.inferenceTimeText || '',
      details: r.details || '',
      error: r.error || '',
      screenshot: r.screenshot || '',
      verification: r.verification || null,
      review: r.review || null,
      observation: r.observation || null,
    }))
  ).replace(/</g, '\\u003c');

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
    .badge-unverified { background: #e0e7ff; color: #3730a3; }
    .checks { max-width: 380px; word-break: break-word; }
    .checks ul { margin: 4px 0 0; padding-left: 16px; }
    .checks li.ok { color: #166534; }
    .checks li.bad { color: #991b1b; font-weight: 600; }
    .details { max-width: 340px; word-break: break-word; }
    .thumb img { width: 180px; height: auto; border: 1px solid #cbd5e1; border-radius: 4px; }
    .view-toggle { margin: 16px 0 0; display: flex; gap: 8px; align-items: center; }
    .view-toggle button { padding: 8px 16px; border: 1px solid #0f172a; background: #fff; border-radius: 6px; cursor: pointer; font-weight: 700; }
    .view-toggle button.active { background: #0f172a; color: #fff; }
    .view-toggle .hint { color: #64748b; font-size: 12px; }
    #review-view { display: none; }
    body.review #review-view { display: block; }
    body.review #table-view { display: none; }
    .review-nav { display: flex; gap: 12px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
    .review-nav button { padding: 10px 22px; border: 1px solid #0f172a; background: #0f172a; color: #fff; border-radius: 6px; cursor: pointer; font-weight: 700; font-size: 15px; }
    .review-nav button.secondary { background: #fff; color: #0f172a; }
    .review-nav button:disabled { opacity: 0.35; cursor: default; }
    .review-nav .position { font-weight: 700; }
    .review-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; display: grid; grid-template-columns: minmax(320px, 2fr) minmax(360px, 3fr); gap: 24px; }
    .review-card h2 { margin: 0 0 4px; font-size: 18px; }
    .review-card h3 { margin: 16px 0 6px; font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
    .review-card .sub { color: #475569; font-size: 13px; margin-bottom: 8px; }
    .review-card .description { font-size: 14px; line-height: 1.45; }
    .review-inputs { display: flex; gap: 12px; flex-wrap: wrap; }
    .review-inputs figure { margin: 0; }
    .review-inputs img { width: 170px; height: auto; border: 1px solid #cbd5e1; border-radius: 4px; display: block; }
    .review-inputs figcaption, .review-inputs .label { font-size: 12px; color: #475569; margin-top: 4px; }
    .review-inputs blockquote { margin: 0; padding: 8px 12px; background: #f1f5f9; border-left: 3px solid #94a3b8; font-size: 13px; max-width: 420px; }
    .review-inputs audio { display: block; width: 300px; }
    .review-checks { list-style: none; margin: 0; padding: 0; font-size: 13px; }
    .review-checks li { padding: 3px 0; border-bottom: 1px solid #f1f5f9; }
    .review-checks li.ok::before { content: '\\2713  '; color: #166534; font-weight: 700; }
    .review-checks li.bad::before { content: '\\2717  '; color: #991b1b; font-weight: 700; }
    .review-checks li.bad { color: #991b1b; font-weight: 600; }
    .review-checks .values { color: #475569; font-weight: 400; }
    .review-error { color: #991b1b; font-weight: 600; font-size: 13px; }
    .review-shot img { width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 4px; }
    .review-steps { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
    .review-steps figure { margin: 0; }
    .review-steps figcaption { font-size: 12px; color: #475569; margin-top: 4px; }
    .review-card details { margin-top: 12px; font-size: 12px; }
    .review-card pre { background: #f8fafc; border: 1px solid #e2e8f0; padding: 8px; overflow: auto; max-height: 260px; }
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
    <div class="card"><div class="num" style="color:#3730a3">${counts.unverified}</div><div class="lbl">Unverified</div></div>
    <div class="card"><div class="num" style="color:#475569">${counts.skip}</div><div class="lbl">Skip</div></div>
  </div>
  <div class="view-toggle">
    <button id="show-table" class="active" onclick="setView('table')">Table</button>
    <button id="show-review" onclick="setView('review')">Review cases one by one</button>
    <span class="hint">In review mode, use Next / Previous or the arrow keys. The status filter applies to both views.</span>
  </div>
  <div class="filters">
    <button class="active" onclick="filterRows('ALL', this)">All (${counts.total})</button>
    <button onclick="filterRows('PASS', this)">Pass (${counts.pass})</button>
    <button onclick="filterRows('FALLBACK_CPU', this)">Fallback CPU (${counts.fallback})</button>
    <button onclick="filterRows('FAIL', this)">Fail (${counts.fail})</button>
    <button onclick="filterRows('UNVERIFIED', this)">Unverified (${counts.unverified})</button>
  </div>
  <div id="review-view">
    <div class="review-nav">
      <button class="secondary" id="review-prev" onclick="stepReview(-1)">&larr; Previous</button>
      <button id="review-next" onclick="stepReview(1)">Next &rarr;</button>
      <span class="position" id="review-position"></span>
    </div>
    <div class="review-card" id="review-card"></div>
  </div>
  <div id="table-view">
  <table>
    <thead>
      <tr>
        <th>#</th><th>Task</th><th>Model</th><th>Mode</th><th>Req</th><th>Actual</th><th>Status</th><th>Duration</th><th>Inference</th><th>Output checks</th><th>Details</th><th>Screenshot</th>
      </tr>
    </thead>
    <tbody>
      ${rowsHtml}
    </tbody>
  </table>
  </div>
  <script type="application/json" id="cases-data">${casesJson}</script>
  <script>
    var cases = JSON.parse(document.getElementById('cases-data').textContent);
    var statusFilter = 'ALL';
    var reviewIndex = 0;

    function filterRows(status, btn) {
      document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      statusFilter = status;
      document.querySelectorAll('tbody tr').forEach(tr => {
        tr.style.display = (status === 'ALL' || tr.dataset.status === status) ? '' : 'none';
      });
      reviewIndex = 0;
      renderReview();
    }

    function setView(view) {
      document.body.classList.toggle('review', view === 'review');
      document.getElementById('show-table').classList.toggle('active', view !== 'review');
      document.getElementById('show-review').classList.toggle('active', view === 'review');
      if (view === 'review') renderReview();
    }

    function visibleCases() {
      return cases.filter(function (c) { return statusFilter === 'ALL' || c.status === statusFilter; });
    }

    function stepReview(delta) {
      var count = visibleCases().length;
      if (count === 0) return;
      reviewIndex = Math.min(count - 1, Math.max(0, reviewIndex + delta));
      renderReview();
    }

    // Builds an element; text is always set as textContent, never as HTML.
    function el(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    }

    function renderInput(input) {
      var wrap;
      if (input.kind === 'image') {
        wrap = el('figure');
        var img = el('img');
        img.src = input.src;
        img.alt = input.label || '';
        wrap.appendChild(img);
        wrap.appendChild(el('figcaption', '', input.label));
      } else if (input.kind === 'audio') {
        wrap = el('div');
        var audio = el('audio');
        audio.controls = true;
        audio.src = input.src;
        wrap.appendChild(audio);
        wrap.appendChild(el('div', 'label', input.label));
      } else {
        wrap = el('div');
        wrap.appendChild(el('blockquote', '', input.value));
        wrap.appendChild(el('div', 'label', input.label));
      }
      return wrap;
    }

    function renderReview() {
      var list = visibleCases();
      var card = document.getElementById('review-card');
      card.textContent = '';
      document.getElementById('review-prev').disabled = reviewIndex <= 0;
      document.getElementById('review-next').disabled = reviewIndex >= list.length - 1;
      if (list.length === 0) {
        document.getElementById('review-position').textContent = 'No cases with this status';
        return;
      }
      var c = list[reviewIndex];
      document.getElementById('review-position').textContent = 'Case ' + (reviewIndex + 1) + ' of ' + list.length;

      var left = el('div');
      left.appendChild(el('h2', '', c.title + ' \\u00b7 ' + c.modelId + ' \\u00b7 ' + c.mode));
      var sub = el('div', 'sub');
      var badge = el('span', 'badge badge-' + c.status.toLowerCase().replace(/[^a-z0-9]+/g, '-'), c.status);
      sub.appendChild(badge);
      sub.appendChild(document.createTextNode(
        '  Delegate requested ' + c.requestedDelegate + ', used ' + (c.actualDelegate || '-') +
        ' \\u00b7 ' + c.durationMs + ' ms' + (c.inferenceTimeText ? ' \\u00b7 ' + c.inferenceTimeText : '')));
      left.appendChild(sub);

      left.appendChild(el('h3', '', 'What this case does'));
      left.appendChild(el('div', 'description', (c.review && c.review.description) || 'No description recorded for this case.'));

      left.appendChild(el('h3', '', 'Input'));
      var inputs = el('div', 'review-inputs');
      ((c.review && c.review.inputs) || []).forEach(function (input) { inputs.appendChild(renderInput(input)); });
      if (!inputs.childNodes.length) inputs.appendChild(el('div', 'label', 'No input recorded for this case.'));
      left.appendChild(inputs);

      left.appendChild(el('h3', '', 'Output checks'));
      if (c.error) left.appendChild(el('div', 'review-error', c.error));
      var verification = c.verification || {};
      left.appendChild(el('div', 'sub', verification.summary || 'No output checks recorded.'));
      var checks = el('ul', 'review-checks');
      (verification.checks || []).forEach(function (check) {
        var li = el('li', check.pass ? 'ok' : 'bad', check.name);
        li.appendChild(el('span', 'values', ' \\u2014 expected ' + check.expected + ', got ' + check.actual));
        checks.appendChild(li);
      });
      left.appendChild(checks);
      if (c.details) {
        left.appendChild(el('h3', '', 'Shown in the UI'));
        left.appendChild(el('div', 'sub', c.details));
      }
      if (c.observation) {
        var raw = el('details');
        raw.appendChild(el('summary', '', 'Raw output read from the app'));
        raw.appendChild(el('pre', '', JSON.stringify(c.observation, null, 2)));
        left.appendChild(raw);
      }

      var right = el('div', 'review-shot');
      right.appendChild(el('h3', '', 'Output as shown by the app'));
      if (c.screenshot) {
        var link = el('a');
        link.href = c.screenshot;
        link.target = '_blank';
        var shot = el('img');
        shot.src = c.screenshot;
        shot.alt = c.id;
        link.appendChild(shot);
        right.appendChild(link);
      } else {
        right.appendChild(el('div', 'sub', 'No screenshot.'));
      }
      var steps = (c.review && c.review.steps) || [];
      if (steps.length > 1) {
        right.appendChild(el('h3', '', 'After each step'));
        var gallery = el('div', 'review-steps');
        steps.forEach(function (step, i) {
          var figure = el('figure');
          var stepLink = el('a');
          stepLink.href = step.src;
          stepLink.target = '_blank';
          var stepImg = el('img');
          stepImg.src = step.src;
          stepImg.alt = step.label;
          stepLink.appendChild(stepImg);
          figure.appendChild(stepLink);
          figure.appendChild(el('figcaption', '', (i + 1) + '. ' + step.label));
          gallery.appendChild(figure);
        });
        right.appendChild(gallery);
      }
      card.appendChild(left);
      card.appendChild(right);
    }

    document.addEventListener('keydown', function (event) {
      if (!document.body.classList.contains('review')) return;
      if (event.key === 'ArrowRight' || event.key === ' ') { event.preventDefault(); stepReview(1); }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); stepReview(-1); }
      else if (event.key === 'Escape') { setView('table'); }
    });
    if (location.hash === '#review') setView('review');
  </script>
</body>
</html>`;

  fs.writeFileSync(path.join(reportDir, 'index.html'), html);
}
