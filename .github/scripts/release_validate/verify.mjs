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
 * Output verification: pure functions that compare what a task produced (read
 * from the app's result hook and from the visible UI) with an expectation from
 * `expectations.mjs`. No browser access here, so every verifier is unit-tested
 * by `runner.mjs self-test` with both a correct and an incorrect output.
 */

/** A case is VERIFIED only if every check passed; UNVERIFIED means nothing was asserted. */
export const VERIFICATION = { VERIFIED: 'VERIFIED', FAILED: 'FAILED', UNVERIFIED: 'UNVERIFIED' };

function fmt(value) {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function check(name, pass, expected, actual) {
  return { name, pass: Boolean(pass), expected: fmt(expected), actual: fmt(actual) };
}

function inRange(value, min, max) {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
}

/** Intersection over union of two `{x, y, width, height}` boxes. */
export function boxIoU(a, b) {
  if (!a || !b) return 0;
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/** Category label as the UI shows it (the sentiment models report `1` / `0`). */
export function displayLabel(label) {
  if (label === '1') return 'positive';
  if (label === '0') return 'negative';
  return label;
}

function topCategory(categories) {
  return [...(categories || [])].sort((a, b) => b.score - a.score)[0] || null;
}

/** Checks shared by every hook payload: it exists, and came from the requested model and delegate. */
function verifyProvenance(hook, context) {
  const checks = [];
  checks.push(check('result published by the app', Boolean(hook) && !hook.error, 'a result', hook ? hook.error || 'ok' : 'none'));
  const payload = hook?.payload;
  if (!payload) return checks;
  if (context.model !== undefined) {
    checks.push(check('result came from the selected model', payload.model === context.model, context.model, payload.model));
  }
  if (context.delegate !== undefined) {
    checks.push(
      check('result came from the selected delegate', payload.delegate === context.delegate, context.delegate, payload.delegate)
    );
  }
  return checks;
}

/** Lower-cased words of a sentence, without punctuation. */
export function normalizeWords(sentence) {
  return String(sentence || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}'\s]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

/** Word error rate: the word-level edit distance from the reference, relative to its length. */
export function wordErrorRate(reference, hypothesis) {
  const ref = normalizeWords(reference);
  const hyp = normalizeWords(hypothesis);
  if (ref.length === 0) return hyp.length === 0 ? 0 : 1;
  let previous = Array.from({ length: hyp.length + 1 }, (_, j) => j);
  for (let i = 1; i <= ref.length; i++) {
    const current = [i];
    for (let j = 1; j <= hyp.length; j++) {
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
    }
    previous = current;
  }
  return previous[hyp.length] / ref.length;
}

/** Share of the reference words that appear in the hypothesis, in order (longest common subsequence). */
export function wordRecall(reference, hypothesis) {
  const ref = normalizeWords(reference);
  const hyp = normalizeWords(hypothesis);
  if (ref.length === 0) return 1;
  let previous = new Array(hyp.length + 1).fill(0);
  for (let i = 1; i <= ref.length; i++) {
    const current = [0];
    for (let j = 1; j <= hyp.length; j++) {
      current[j] = ref[i - 1] === hyp[j - 1] ? previous[j - 1] + 1 : Math.max(previous[j], current[j - 1]);
    }
    previous = current;
  }
  return previous[hyp.length] / ref.length;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Checks that a point lies within `target.tolerance` of where it really is in the image. */
function checkNear(name, point, target) {
  const ok = Boolean(point) && Number.isFinite(point.x) && Number.isFinite(point.y) && distance(point, target) <= target.tolerance;
  return check(
    name,
    ok,
    `within ${target.tolerance} of (${fmt(target.x)}, ${fmt(target.y)})`,
    point ? `(${fmt(point.x)}, ${fmt(point.y)})` : 'none'
  );
}

/** Reads `key` from a payload; `a.b` style paths are not needed, keys are top level. */
function asList(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

const VERIFIERS = {
  /** Object detection: each required object is found, confidently, roughly where it really is. */
  detections(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    const detections = payload.detections || [];

    if (expectation.mode) checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    const badBoxes = detections.filter(
      (d) =>
        !d.box ||
        !inRange(d.box.width, 1, payload.imageWidth) ||
        !inRange(d.box.height, 1, payload.imageHeight) ||
        !inRange(d.score, 0, 1)
    );
    checks.push(check('every detection has a valid box and score', badBoxes.length === 0, 0, badBoxes.length));

    for (const required of expectation.objects) {
      const candidates = detections.filter((d) => d.label === required.label);
      const best = [...candidates].sort((a, b) => b.score - a.score)[0];
      checks.push(
        check(
          `detects "${required.label}"`,
          Boolean(best),
          required.label,
          detections.map((d) => d.label).join(', ') || 'nothing'
        )
      );
      if (!best) continue;
      checks.push(
        check(`"${required.label}" score`, best.score >= required.minScore, `>= ${required.minScore}`, best.score)
      );
      // The true box is given relative to the image, so it applies to any frame size.
      const truth = {
        x: required.box.x * payload.imageWidth,
        y: required.box.y * payload.imageHeight,
        width: required.box.width * payload.imageWidth,
        height: required.box.height * payload.imageHeight,
      };
      const iou = boxIoU(best.box, truth);
      checks.push(check(`"${required.label}" box overlaps the object`, iou >= required.minIoU, `IoU >= ${required.minIoU}`, iou));
    }
    return checks;
  },

  /**
   * Text classification and language detection: each input gets its true
   * label, and the UI shows that same label.
   */
  textLabels(observation, expectation, context) {
    const checks = [];
    const runs = observation.runs || [];
    checks.push(check('every input was classified', runs.length === expectation.inputs.length, expectation.inputs.length, runs.length));
    expectation.inputs.forEach((input, idx) => {
      const run = runs[idx];
      const tag = `input ${idx + 1} (${input.label})`;
      if (!run) return;
      checks.push(...verifyProvenance(run.hook, context).map((c) => ({ ...c, name: `${tag}: ${c.name}` })));
      const payload = run.hook?.payload;
      if (!payload) return;
      checks.push(check(`${tag}: result is for the submitted text`, payload.text === input.text, input.text, payload.text));
      // Language detection reports `languages` with a code and a probability.
      const categories = payload.categories || (payload.languages || []).map((l) => ({ label: l.code, score: l.probability }));
      const top = topCategory(categories);
      const topLabel = top ? displayLabel(top.label) : 'none';
      checks.push(check(`${tag}: top label`, topLabel === input.label, input.label, topLabel));
      checks.push(check(`${tag}: top score is a probability`, Boolean(top) && inRange(top.score, 0, 1), '0 to 1', top?.score ?? 'none'));
      const uiTop = run.ui?.[0];
      checks.push(check(`${tag}: UI shows the top label`, uiTop?.label === topLabel, topLabel, uiTop?.label ?? 'none'));
      const expectedScoreText = top ? `${Math.round(top.score * 100)}%` : 'none';
      checks.push(check(`${tag}: UI shows the top score`, uiTop?.score === expectedScoreText, expectedScoreText, uiTop?.score ?? 'none'));
    });
    return checks;
  },

  /** Face detection: the right number of faces, each with its eyes and nose where they really are. */
  faces(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    if (expectation.mode) checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    const detections = payload.detections || [];
    checks.push(check('faces found', detections.length === expectation.count, expectation.count, detections.length));
    const face = [...detections].sort((a, b) => b.score - a.score)[0];
    if (!face) return checks;
    checks.push(check('face score', inRange(face.score, expectation.minScore, 1), `>= ${expectation.minScore}`, face.score));
    for (const keypoint of expectation.keypoints) {
      checks.push(checkNear(`${keypoint.name} position`, face.keypoints?.[keypoint.index], keypoint));
    }
    const box = face.box;
    const widthShare = box && payload.imageWidth > 0 ? box.width / payload.imageWidth : NaN;
    const [minWidth, maxWidth] = expectation.boxWidth;
    checks.push(check('face box width as a share of the image', inRange(widthShare, minWidth, maxWidth), `${minWidth} to ${maxWidth}`, widthShare));
    const inside = expectation.boxContains;
    const px = inside.x * payload.imageWidth;
    const py = inside.y * payload.imageHeight;
    const contains = Boolean(box) && px >= box.x && px <= box.x + box.width && py >= box.y && py <= box.y + box.height;
    checks.push(check('face box contains the nose', contains, 'yes', contains ? 'yes' : 'no'));
    return checks;
  },

  /**
   * Landmark tasks (face, hand, pose, holistic, gesture): the right number of
   * instances, the documented number of landmarks each, and named landmarks
   * where that body part really is in the photo.
   */
  landmarks(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    if (expectation.mode) checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));

    for (const group of expectation.groups) {
      const instances = asList(payload[group.key]).filter((instance) => instance && instance.count > 0);
      checks.push(check(`${group.name}s found`, instances.length === group.instances, group.instances, instances.length));
      const instance = instances[0];
      if (!instance) continue;
      checks.push(check(`${group.name} landmark count`, instance.count === group.landmarkCount, group.landmarkCount, instance.count));
      checks.push(check(`${group.name} landmarks are finite`, instance.nonFiniteCount === 0, 0, instance.nonFiniteCount));
      const box = instance.box;
      // Landmarks may fall slightly outside the frame, but not far.
      const inFrame = Boolean(box) && box.minX >= -0.25 && box.minY >= -0.25 && box.maxX <= 1.25 && box.maxY <= 1.25;
      checks.push(
        check(
          `${group.name} landmarks lie in the image`,
          inFrame,
          'within the frame',
          box ? `x ${fmt(box.minX)}-${fmt(box.maxX)}, y ${fmt(box.minY)}-${fmt(box.maxY)}` : 'none'
        )
      );
      for (const [name, target] of Object.entries(group.points || {})) {
        checks.push(checkNear(`${group.name} ${name} position`, instance.points?.[name], target));
      }
      for (const [field, value] of Object.entries(group.fields || {})) {
        checks.push(check(`${group.name} ${field}`, instance[field] === value, value, instance[field] ?? 'none'));
      }
      if (group.minScore !== undefined) {
        checks.push(check(`${group.name} score`, inRange(instance.score, group.minScore, 1), `>= ${group.minScore}`, instance.score ?? 'none'));
      }
    }
    for (const { key, value } of expectation.equals || []) {
      checks.push(check(key, JSON.stringify(payload[key]) === JSON.stringify(value), value, payload[key] ?? 'none'));
    }
    if (expectation.gestureRowInUi) {
      // The app shows "<handedness>: <gesture>" and its score for each hand.
      const hand = asList(payload.hands)[0];
      const uiTop = observation.ui?.[0];
      const expectedLabel = hand ? `${hand.handednessDisplayName}: ${hand.gesture}` : 'none';
      checks.push(check('UI shows the gesture', Boolean(hand) && uiTop?.label === expectedLabel, expectedLabel, uiTop?.label ?? 'none'));
      const expectedScore = hand ? `${Math.round(hand.score * 100)}%` : 'none';
      checks.push(check('UI shows the gesture score', uiTop?.score === expectedScore, expectedScore, uiTop?.score ?? 'none'));
    }
    return checks;
  },

  /** Image classification: the top label is what the photo shows, and the UI shows it. */
  imageClassification(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    if (expectation.mode) checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    const top = topCategory(payload.categories);
    checks.push(check('top label', top?.label === expectation.label, expectation.label, top?.label ?? 'none'));
    checks.push(check('top score', Boolean(top) && inRange(top.score, expectation.minScore, 1), `>= ${expectation.minScore}`, top?.score ?? 'none'));
    const uiTop = observation.ui?.[0];
    checks.push(check('UI shows the top label', Boolean(top) && uiTop?.label === top.label, top?.label ?? 'none', uiTop?.label ?? 'none'));
    const expectedScore = top ? `${Math.round(top.score * 100)}%` : 'none';
    checks.push(check('UI shows the top score', uiTop?.score === expectedScore, expectedScore, uiTop?.score ?? 'none'));
    return checks;
  },

  /**
   * Embedders: the similarity of each pair is in the range its content implies,
   * pairs are ordered as their meaning implies, and the UI shows the same number.
   */
  similarity(observation, expectation, context) {
    const checks = [];
    const runs = observation.runs || [];
    checks.push(check('every pair was compared', runs.length === expectation.runs.length, expectation.runs.length, runs.length));
    const values = [];
    expectation.runs.forEach((expected, idx) => {
      const run = runs[idx];
      if (!run) return;
      const tag = expected.name;
      checks.push(...verifyProvenance(run.hook, context).map((c) => ({ ...c, name: `${tag}: ${c.name}` })));
      const payload = run.hook?.payload;
      if (!payload) return;
      const key = payload[expectation.keyField];
      checks.push(check(`${tag}: result is for the chosen pair`, JSON.stringify(key) === JSON.stringify(expected.key), expected.key, key ?? 'none'));
      const [lo, hi] = expected.range;
      checks.push(check(`${tag}: similarity`, inRange(payload.similarity, lo, hi), `${lo} to ${hi}`, payload.similarity ?? 'none'));
      if (expectation.sameDimensions) {
        const [a, b] = payload.dimensions || [];
        checks.push(check(`${tag}: embedding sizes`, a > 0 && a === b, 'equal and > 0', `${a} and ${b}`));
      }
      if (expectation.l2Normalize !== undefined) {
        checks.push(check(`${tag}: L2 normalization setting`, payload.l2Normalize === expectation.l2Normalize, expectation.l2Normalize, payload.l2Normalize));
        const norms = payload.norms || [];
        const finite = norms.length === 2 && norms.every((n) => Number.isFinite(n) && n > 0);
        // Normalized embeddings have length 1.
        const ok = expectation.l2Normalize ? finite && norms.every((n) => Math.abs(n - 1) <= 0.01) : finite;
        checks.push(check(`${tag}: embedding lengths`, ok, expectation.l2Normalize ? '1 (within 0.01)' : 'finite and > 0', norms.map(fmt).join(' and ') || 'none'));
      }
      const shown = typeof payload.similarity === 'number' ? payload.similarity.toFixed(4) : 'none';
      checks.push(check(`${tag}: UI shows the similarity`, run.ui === shown, shown, run.ui ?? 'none'));
      values[idx] = payload.similarity;
    });
    for (const order of expectation.ordered || []) {
      const higher = values[order.higher];
      const lower = values[order.lower];
      const gap = typeof higher === 'number' && typeof lower === 'number' ? higher - lower : NaN;
      checks.push(
        check(
          `${expectation.runs[order.higher].name} score above ${expectation.runs[order.lower].name}`,
          gap >= order.by,
          `by >= ${order.by}`,
          gap
        )
      );
    }
    return checks;
  },

  /**
   * Image segmentation: each class the photo contains covers the share of the
   * image it really occupies, in the right place, and nothing else is found.
   * In confidence mode, the confidence mask of the chosen class is checked too.
   */
  segmentation(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    checks.push(check('output type', payload.outputType === expectation.outputType, expectation.outputType, payload.outputType));
    const stats = payload.maskStats;
    checks.push(
      check('mask produced', payload.hasMask === true && Boolean(stats) && !stats?.error, 'a mask with statistics', stats?.error || (stats ? 'ok' : 'none'))
    );
    if (!stats || stats.error) return checks;

    const labels = payload.labels || [];
    // Pixels of an index without a label (255 for the selfie models) are background.
    const labelOf = (index) => labels[index] ?? 'background';
    const found = (stats.categories || []).map((c) => ({ ...c, label: labelOf(c.index) }));
    const listed = new Set(expectation.classes.map((c) => c.label));
    for (const expected of expectation.classes) {
      const actual = found.find((c) => c.label === expected.label);
      const [lo, hi] = expected.fraction;
      checks.push(check(`"${expected.label}" share of the image`, inRange(actual?.fraction ?? 0, lo, hi), `${lo} to ${hi}`, actual?.fraction ?? 0));
      if (expected.centroidWithin) {
        const c = actual?.centroid;
        const b = expected.centroidWithin;
        const inside = Boolean(c) && inRange(c.x, b.minX, b.maxX) && inRange(c.y, b.minY, b.maxY);
        checks.push(
          check(`"${expected.label}" centre`, inside, `x ${b.minX}-${b.maxX}, y ${b.minY}-${b.maxY}`, c ? `x ${fmt(c.x)}, y ${fmt(c.y)}` : 'none')
        );
      }
    }
    const unexpected = found.filter((c) => !listed.has(c.label) && c.label !== 'background' && c.fraction > expectation.maxOther);
    checks.push(
      check(
        'no other class in the image',
        unexpected.length === 0,
        `every other class <= ${expectation.maxOther}`,
        unexpected.map((c) => `${c.label} ${fmt(c.fraction)}`).join(', ') || 'none'
      )
    );

    if (expectation.confidence) {
      const expected = expectation.confidence;
      const index = labels.indexOf(expected.label);
      const confidence = stats.confidence;
      checks.push(check('confidence mask class chosen in the UI', payload.confidenceClass === index && confidence?.index === index, `${expected.label} (${index})`, payload.confidenceClass));
      if (confidence) {
        checks.push(check('confidence values are finite', confidence.nonFiniteCount === 0, 0, confidence.nonFiniteCount));
        checks.push(
          check('confidence values are probabilities', inRange(confidence.min, 0, 1) && inRange(confidence.max, 0, 1), '0 to 1', `${fmt(confidence.min)} to ${fmt(confidence.max)}`)
        );
        checks.push(check('highest confidence', confidence.max >= expected.minPeak, `>= ${expected.minPeak}`, confidence.max));
        const [lo, hi] = expected.fraction;
        checks.push(check(`share of the image confidently "${expected.label}"`, inRange(confidence.aboveHalfFraction, lo, hi), `${lo} to ${hi}`, confidence.aboveHalfFraction));
        checks.push(check('mean confidence', inRange(confidence.mean, lo, hi), `${lo} to ${hi}`, confidence.mean));
      }
    }
    return checks;
  },

  /** Audio classification: the clip is labelled correctly in (almost) every window, and the UI agrees. */
  audioClassification(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    if (expectation.mode) checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    const windows = payload.results || [];
    const tops = windows.map((w) => topCategory(w.categories));
    checks.push(check('classified windows', windows.length >= expectation.fullWindows, `>= ${expectation.fullWindows}`, windows.length));
    // Only the full windows are judged; a trailing partial window may be inconclusive.
    const full = tops.slice(0, expectation.fullWindows);
    const matching = full.filter((t) => t && t.label === expectation.label && t.score >= expectation.minScore).length;
    checks.push(
      check(
        `full windows labelled "${expectation.label}" (score >= ${expectation.minScore})`,
        full.length === expectation.fullWindows && matching === expectation.fullWindows,
        `${expectation.fullWindows} of ${expectation.fullWindows}`,
        `${matching} of ${full.length} (${full.map((t) => (t ? `${t.label} ${fmt(t.score)}` : 'none')).join(', ')})`
      )
    );
    // The app shows the first window only.
    const first = tops[0];
    const uiTop = observation.ui?.[0];
    checks.push(check('UI shows the first window label', Boolean(first) && uiTop?.label === first.label, first?.label ?? 'none', uiTop?.label ?? 'none'));
    return checks;
  },

  /** Semantic retrieval: for each query, the photo it describes is the top result, and the UI lists it first. */
  retrieval(observation, expectation, context) {
    const checks = [];
    const runs = observation.runs || [];
    checks.push(check('every query was searched', runs.length === expectation.queries.length, expectation.queries.length, runs.length));
    expectation.queries.forEach((expected, idx) => {
      const run = runs[idx];
      if (!run) return;
      const tag = `"${expected.query}"`;
      checks.push(...verifyProvenance(run.hook, context).map((c) => ({ ...c, name: `${tag}: ${c.name}` })));
      const payload = run.hook?.payload;
      if (!payload) return;
      checks.push(check(`${tag}: result is for the typed query`, payload.query === expected.query, expected.query, payload.query));
      checks.push(check(`${tag}: photos in the index`, payload.indexedCount === expectation.corpusSize, expectation.corpusSize, payload.indexedCount));
      const results = payload.results || [];
      checks.push(check(`${tag}: top result`, results[0]?.id === expected.topId, expected.topId, results.map((r) => r.id).slice(0, 3).join(', ') || 'none'));
      const scores = results.map((r) => r.score);
      const sorted = scores.every((score, i) => i === 0 || scores[i - 1] >= score);
      const valid = scores.length > 0 && scores.every((score) => inRange(score, -1.0001, 1.0001));
      checks.push(check(`${tag}: scores are valid and sorted`, sorted && valid, 'descending, -1 to 1', scores.map(fmt).join(', ') || 'none'));
      checks.push(check(`${tag}: UI lists it first`, run.ui?.title === expected.topTitle, expected.topTitle, run.ui?.title ?? 'none'));
      const shownScore = results[0] ? `Similarity ${results[0].score.toFixed(4)}` : 'none';
      checks.push(check(`${tag}: UI shows its score`, run.ui?.score === shownScore, shownScore, run.ui?.score ?? 'none'));
    });
    return checks;
  },

  /**
   * Decision Maker: each support ticket is triaged the way a person would, in
   * whatever form the tab reports it (a whole schema, a yes/no, a choice or a
   * score), and the UI shows that decision.
   */
  decisions(observation, expectation, context) {
    const checks = [];
    const runs = observation.runs || [];
    checks.push(check('every ticket was evaluated', runs.length === expectation.tickets.length, expectation.tickets.length, runs.length));
    const severities = [];
    expectation.tickets.forEach((ticket, idx) => {
      const run = runs[idx];
      if (!run) return;
      const tag = ticket.name;
      checks.push(...verifyProvenance(run.hook, context).map((c) => ({ ...c, name: `${tag}: ${c.name}` })));
      const payload = run.hook?.payload;
      if (!payload) return;
      checks.push(check(`${tag}: evaluation succeeded`, payload.ok === true, 'ok', payload.ok ? 'ok' : payload.error || 'failed'));
      checks.push(check(`${tag}: request kind`, payload.kind === expectation.kind, expectation.kind, payload.kind));
      checks.push(check(`${tag}: result is for the typed ticket`, payload.text === ticket.text, ticket.text, payload.text));
      const result = payload.result || {};
      const ui = run.ui || {};
      if (expectation.kind === 'schema') {
        const urgent = result.is_urgent?.label;
        const category = result.category?.label;
        checks.push(check(`${tag}: urgent outage`, urgent === String(ticket.urgent), String(ticket.urgent), urgent ?? 'none'));
        checks.push(check(`${tag}: department`, category === ticket.category, ticket.category, category ?? 'none'));
        severities[idx] = result.severity?.expectedScore;
        const shown = `is_urgent: ${ticket.urgent ? 'TRUE' : 'FALSE'}`;
        checks.push(check(`${tag}: UI shows the urgency`, (ui.subtitle || '').includes(shown), shown, ui.subtitle || 'none'));
        checks.push(check(`${tag}: UI shows the department`, (ui.subtitle || '').includes(`category: ${category}`), `category: ${category}`, ui.subtitle || 'none'));
      } else if (expectation.kind === 'boolean') {
        checks.push(check(`${tag}: urgent outage`, result.value === ticket.urgent, ticket.urgent, result.value ?? 'none'));
        checks.push(check(`${tag}: probability is valid`, inRange(result.probabilityTrue, 0, 1), '0 to 1', result.probabilityTrue ?? 'none'));
        const shown = result.value ? 'Yes' : 'No';
        checks.push(check(`${tag}: UI shows the answer`, ui.headline === shown, shown, ui.headline || 'none'));
      } else if (expectation.kind === 'choice') {
        checks.push(check(`${tag}: department`, result.selectedKey === ticket.category, ticket.category, result.selectedKey ?? 'none'));
        checks.push(check(`${tag}: UI shows the choice`, ui.headline === result.selectedKey, result.selectedKey ?? 'none', ui.headline || 'none'));
      } else if (expectation.kind === 'score') {
        severities[idx] = result.expectedScore;
        checks.push(check(`${tag}: severity score is valid`, inRange(result.expectedScore, 0, 4), '0 to 4 (shown as 1 to 5)', result.expectedScore ?? 'none'));
        const shown = typeof result.expectedScore === 'number' ? `${(result.expectedScore + 1).toFixed(2)} / 5` : 'none';
        checks.push(check(`${tag}: UI shows the score`, ui.headline === shown, shown, ui.headline || 'none'));
      }
    });
    if (expectation.kind === 'schema' || expectation.kind === 'score') {
      const gap = typeof severities[0] === 'number' && typeof severities[1] === 'number' ? severities[0] - severities[1] : NaN;
      checks.push(
        check(
          `${expectation.tickets[0].name} is more severe than the ${expectation.tickets[1].name}`,
          gap >= expectation.minSeverityGap,
          `by >= ${expectation.minSeverityGap} point`,
          gap
        )
      );
    }
    return checks;
  },

  /**
   * ID Photo: the compliant photo passes every rule of the pack and the photo
   * that breaks a rule fails it, both in the model's result and in the UI.
   */
  photoCompliance(observation, expectation, context) {
    const checks = [];
    const photos = observation.photos || [];
    checks.push(check('every photo was checked', photos.length === expectation.photos.length, expectation.photos.length, photos.length));
    expectation.photos.forEach((expected, idx) => {
      const photo = photos[idx];
      if (!photo) return;
      const tag = expected.name;
      checks.push(...verifyProvenance(photo.hook, context).map((c) => ({ ...c, name: `${tag}: ${c.name}` })));
      const payload = photo.hook?.payload;
      if (!payload) return;
      checks.push(check(`${tag}: result is for the selected photo`, payload.photo === expected.preset, expected.preset, payload.photo));
      checks.push(check(`${tag}: rule pack chosen in the UI`, photo.ui?.pack === expectation.pack, expectation.pack, photo.ui?.pack ?? 'none'));
      const judged = new Map((payload.rules || []).map((r) => [r.id, r]));
      const shown = new Map((photo.ui?.rules || []).map((r) => [r.id, r.status]));
      const failed = expectation.rules.filter((id) => !judged.get(id)?.passed);
      if (expected.compliant) {
        checks.push(check(`${tag}: passes every rule of the pack`, failed.length === 0, 'no failed rule', failed.join(', ') || 'none'));
        checks.push(check(`${tag}: UI verdict`, photo.ui?.verdict === 'COMPLIANT', 'COMPLIANT', photo.ui?.verdict ?? 'none'));
      } else {
        const rule = judged.get(expected.failingRule);
        checks.push(check(`${tag}: fails "${expected.failingRule}"`, Boolean(rule) && rule.passed === false, 'failed', rule ? (rule.passed ? 'passed' : 'failed') : 'not judged'));
        const settled = Boolean(photo.ui?.verdict) && !/EVALUATING|LOADING/.test(photo.ui.verdict);
        checks.push(check(`${tag}: UI verdict`, settled && photo.ui.verdict !== 'COMPLIANT', 'not COMPLIANT', photo.ui?.verdict ?? 'none'));
        checks.push(check(`${tag}: UI marks "${expected.failingRule}" as failed`, shown.get(expected.failingRule) === 'fail', 'fail', shown.get(expected.failingRule) ?? 'none'));
      }
      // The rule cards must show what the evaluation found.
      const disagree = expectation.rules.filter((id) => (shown.get(id) === 'pass') !== Boolean(judged.get(id)?.passed));
      checks.push(check(`${tag}: UI rule cards match the evaluation`, disagree.length === 0, 'all agree', disagree.join(', ') || 'all agree'));
    });
    return checks;
  },

  /**
   * Dino Game: every decision the model made fits the scene it was made for,
   * the scenes the runner set up were all decided, and the UI shows the action.
   */
  gameDecisions(observation, expectation, context) {
    const hooks = observation.hooks || [];
    const checks = [];
    checks.push(check('decisions published by the app', hooks.length > 0, '> 0', hooks.length));
    if (hooks.length === 0) return checks;
    checks.push(...verifyProvenance(hooks[hooks.length - 1], context));
    const otherMode = hooks.filter((h) => h.payload?.inputMode !== expectation.inputMode).length;
    checks.push(check('input mode', otherMode === 0, expectation.inputMode, otherMode === 0 ? expectation.inputMode : `${otherMode} decisions in another mode`));

    const wrong = [];
    const decided = new Set();
    for (const hook of hooks) {
      const payload = hook.payload || {};
      const rule = expectation.sceneActions.find((entry) => entry.pattern.test(payload.scene || ''));
      if (!rule) {
        wrong.push(`unknown scene "${payload.scene}"`);
      } else if (payload.action !== rule.action) {
        wrong.push(`${rule.scene}: ${payload.action} instead of ${rule.action}`);
      } else {
        decided.add(payload.action);
      }
    }
    checks.push(check('every decision fits its scene', wrong.length === 0, 'all fit', [...new Set(wrong)].join('; ') || `all ${hooks.length} fit`));
    for (const action of expectation.requiredActions) {
      const rule = expectation.sceneActions.find((entry) => entry.action === action);
      const happened = hooks.some((h) => h.payload?.action === action && rule.pattern.test(h.payload?.scene || ''));
      checks.push(check(`decided ${action} for ${rule.scene}`, happened, 'at least once', happened ? 'yes' : 'never'));
    }
    const last = hooks[hooks.length - 1].payload || {};
    checks.push(check('UI shows the latest action', observation.ui?.action === last.action, last.action, observation.ui?.action ?? 'none'));
    return checks;
  },

  /** Speech recognition: the transcript says what is said in the clip, and the UI shows that transcript. */
  transcript(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    checks.push(check('running mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    checks.push(check('transcription finished', payload.final === true, 'final', payload.final ? 'final' : 'not final'));
    const transcript = payload.transcript || '';
    if (expectation.maxWordErrorRate !== undefined) {
      const wer = wordErrorRate(expectation.reference, transcript);
      checks.push(
        check(
          'word error rate against what is said in the clip',
          wer <= expectation.maxWordErrorRate,
          `<= ${expectation.maxWordErrorRate} ("${expectation.reference}")`,
          `${fmt(wer)} ("${transcript}")`
        )
      );
    }
    if (expectation.minWordRecall !== undefined) {
      const recall = wordRecall(expectation.reference, transcript);
      checks.push(
        check(
          'share of the words in the clip that were heard, in order',
          recall >= expectation.minWordRecall,
          `>= ${expectation.minWordRecall} ("${expectation.reference}")`,
          `${fmt(recall)} ("${transcript}")`
        )
      );
    }
    checks.push(check('UI shows the transcript', (observation.ui || '') === transcript, transcript || '(empty)', observation.ui || '(empty)'));
    return checks;
  },

  /** Text to speech: the audio is a plausible, audible, valid rendering of the sentence, and the UI reports its length. */
  speech(observation, expectation, context) {
    const checks = verifyProvenance(observation.hook, context);
    const payload = observation.hook?.payload;
    if (!payload) return checks;
    checks.push(check('synthesis mode', payload.mode === expectation.mode, expectation.mode, payload.mode));
    checks.push(check('audio is for the typed sentence', payload.text === expectation.text, expectation.text, payload.text));
    checks.push(check('synthesis completed', payload.cancelled === false, 'not cancelled', payload.cancelled ? 'cancelled' : 'not cancelled'));
    checks.push(check('audio chunks received', payload.chunkCount >= 1, '>= 1', payload.chunkCount));
    const [minRate, maxRate] = expectation.sampleRateHz;
    checks.push(check('sample rate (Hz)', inRange(payload.sampleRateHz, minRate, maxRate), `${minRate} to ${maxRate}`, payload.sampleRateHz));
    const [minSec, maxSec] = expectation.durationSec;
    checks.push(check('audio length (s)', inRange(payload.durationSec, minSec, maxSec), `${minSec} to ${maxSec}`, payload.durationSec));
    const expectedSamples = Math.round(payload.durationSec * payload.sampleRateHz);
    checks.push(check('sample count matches the length', payload.sampleCount === expectedSamples, expectedSamples, payload.sampleCount));
    checks.push(check('samples are finite', payload.nonFiniteCount === 0, 0, payload.nonFiniteCount));
    checks.push(check('audio is audible (RMS level)', payload.rms >= expectation.minRms, `>= ${expectation.minRms}`, payload.rms));
    checks.push(check('audio is not clipped (peak level)', inRange(payload.peak, 0, expectation.maxPeak), `<= ${expectation.maxPeak}`, payload.peak));
    const shown = typeof payload.durationSec === 'number' ? `${payload.durationSec.toFixed(2)} s` : 'none';
    checks.push(check('UI shows the audio length', observation.ui?.audio === shown, shown, observation.ui?.audio ?? 'none'));
    return checks;
  },

  /**
   * Live audio classification: of the chunks classified while the clip plays
   * through the microphone, enough are labelled correctly and almost none as
   * anything else, and the UI shows that label.
   */
  audioStream(observation, expectation, context) {
    const hooks = observation.hooks || [];
    const checks = [];
    checks.push(check('live results published by the app', hooks.length > 0, '> 0', hooks.length));
    if (hooks.length === 0) return checks;
    checks.push(...verifyProvenance(hooks[hooks.length - 1], context));
    const wrongMode = hooks.filter((h) => h.payload?.mode !== expectation.mode).length;
    checks.push(check('running mode', wrongMode === 0, expectation.mode, wrongMode === 0 ? expectation.mode : `${wrongMode} results in another mode`));
    const tops = hooks.map((h) => topCategory(h.payload?.results?.[0]?.categories)).filter(Boolean);
    const matching = tops.filter((t) => t.label === expectation.label && t.score >= expectation.minScore).length;
    checks.push(check(`chunks labelled "${expectation.label}"`, matching >= expectation.minMatches, `>= ${expectation.minMatches}`, matching));
    const fraction = tops.length > 0 ? matching / tops.length : 0;
    const others = [...new Set(tops.filter((t) => t.label !== expectation.label).map((t) => t.label))];
    checks.push(
      check(
        `share of labelled chunks that are "${expectation.label}"`,
        fraction >= expectation.minFraction,
        `>= ${expectation.minFraction}`,
        `${fmt(fraction)}${others.length ? ` (also: ${others.join(', ')})` : ''}`
      )
    );
    const uiLabels = [...new Set((observation.uiLabels || []).filter((label) => label && label !== '--' && label !== 'No results'))];
    checks.push(check('UI shows the label while recording', uiLabels.includes(expectation.label), expectation.label, uiLabels.join(', ') || 'none'));
    return checks;
  },

  /**
   * Interactive segmentation: after each stroke drawn in the UI, the mask is
   * where the strokes say it should be.
   */
  maskSteps(observation, expectation, context) {
    const checks = [];
    const steps = observation.steps || [];
    checks.push(check('every stroke produced a result', steps.length === expectation.steps.length, expectation.steps.length, steps.length));
    let previous = null;
    expectation.steps.forEach((expected, idx) => {
      const step = steps[idx];
      if (!step) return;
      const tag = (name) => (expectation.steps.length > 1 ? `after stroke ${idx + 1}: ${name}` : name);
      checks.push(...verifyProvenance(step.hook, context).map((c) => ({ ...c, name: tag(c.name) })));
      const payload = step.hook?.payload;
      if (!payload) return;
      const stats = payload.maskStats;
      checks.push(
        check(
          tag('mask produced'),
          payload.hasMask === true && Boolean(stats) && !stats?.error,
          'a mask with statistics',
          stats?.error || (stats ? 'ok' : 'none')
        )
      );
      if (!stats || stats.error) return;

      checks.push(check(tag('mask values are finite'), stats.nonFiniteCount === 0, 0, stats.nonFiniteCount));
      const [minFg, maxFg] = expected.foregroundFraction;
      checks.push(
        check(tag('share of the image selected'), inRange(stats.foregroundFraction, minFg, maxFg), `${minFg} to ${maxFg}`, stats.foregroundFraction)
      );
      if (expected.shrinksBy !== undefined) {
        const drop = previous ? previous.foregroundFraction - stats.foregroundFraction : NaN;
        checks.push(
          check(tag('selection shrank after the negative stroke'), drop >= expected.shrinksBy, `by >= ${expected.shrinksBy} of the image`, drop)
        );
      }
      checks.push(
        check(tag('strokes sent to the model'), (stats.strokes || []).length === expected.strokes.length, expected.strokes.length, (stats.strokes || []).length)
      );
      expected.strokes.forEach((expectedStroke, strokeIdx) => {
        const stroke = stats.strokes?.[strokeIdx];
        if (!stroke) return;
        const name = `stroke ${strokeIdx + 1}`;
        checks.push(check(tag(`${name} brush mode`), stroke.brushMode === expectedStroke.brushMode, expectedStroke.brushMode, stroke.brushMode));
        if (expectedStroke.onMask) {
          const [lo, hi] = expectedStroke.onMask;
          checks.push(check(tag(`${name} share lying on the mask`), inRange(stroke.foregroundFraction, lo, hi), `${lo} to ${hi}`, stroke.foregroundFraction));
        }
      });
      if (expected.centroidWithin) {
        const c = stats.foregroundCentroid;
        const b = expected.centroidWithin;
        const inside = Boolean(c) && inRange(c.x, b.minX, b.maxX) && inRange(c.y, b.minY, b.maxY);
        checks.push(
          check(tag('centre of the selection'), inside, `x ${b.minX}-${b.maxX}, y ${b.minY}-${b.maxY}`, c ? `x ${fmt(c.x)}, y ${fmt(c.y)}` : 'none')
        );
      }
      previous = stats;
    });
    return checks;
  },
};

/**
 * Verifies one case. `expectation` is null when no ground truth is defined for
 * the case, which yields UNVERIFIED so that it is never reported as a pass.
 */
export function verifyCase(expectation, observation, context = {}) {
  if (!expectation) {
    return { status: VERIFICATION.UNVERIFIED, reason: 'no output expectation is defined for this case', checks: [] };
  }
  if (expectation.unverified) {
    return { status: VERIFICATION.UNVERIFIED, reason: expectation.unverified, checks: [] };
  }
  const verifier = VERIFIERS[expectation.type];
  if (!verifier) throw new Error(`Unknown expectation type: ${expectation.type}`);
  const checks = verifier(observation || {}, expectation, context);
  const failed = checks.filter((c) => !c.pass);
  return {
    status: failed.length === 0 ? VERIFICATION.VERIFIED : VERIFICATION.FAILED,
    reason: failed.map((c) => `${c.name}: expected ${c.expected}, got ${c.actual}`).join('; '),
    checks,
  };
}
