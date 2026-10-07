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
 * Dino Game: a minimal "dino run" game driven by the MediaPipe DecisionMaker
 * task (@mediapipe/tasks-decision).
 *
 * The model never sees distances. Whenever what's ahead of the dino changes,
 * the game describes it in words ("Low flying obstacle ahead: pterodactyl.")
 * and asks a Choice question: jump, duck or wait. The game code only handles
 * timing: it performs the chosen move when the obstacle gets close.
 *
 * The simulation can be paused and advanced one tick at a time so users can
 * inspect each decision. The model is shared with the Decision Maker page
 * (see decision-runtime.ts), so loading it on either page is enough.
 */

import template from '../templates/dino-game.html?raw';
import { InferenceTimer } from '../components/inference-timer';
import { decisionRuntime } from '../components/decision-runtime';
import { mountDecisionModelPanel } from '../components/decision-model-panel';

// ---------------------------------------------------------------------------
// Constants (all units are pixels and ticks; one tick = 1/60 s at 1x speed)
// ---------------------------------------------------------------------------

const WIDTH = 800;
const HEIGHT = 220;
const GROUND_Y = 180;

const DINO_X = 60;
const DINO_W = 24;
const DINO_H = 28;
const DUCK_W = 32;
const DUCK_H = 16;

const GRAVITY = 0.6;
const JUMP_VELOCITY = 11;
const AIR_TIME = (2 * JUMP_VELOCITY) / GRAVITY; // ticks spent in the air
const APEX_TIME = JUMP_VELOCITY / GRAVITY; // ticks to reach the top of a jump

const START_SPEED = 5;
const MAX_SPEED = 10;
const ACCELERATION = 0.001;

const TICK_MS = 1000 / 60;

/**
 * Time the model gets to answer: an obstacle comes into view only this many
 * ticks before the dino must start its jump (6 ticks = 100 ms at 1x).
 */
const DECISION_BUDGET_TICKS = 6;

/** The model is asked about the next obstacle once it is this close (grows with speed). */
function lookDistance(speed: number): number {
  return speed * (APEX_TIME + DECISION_BUDGET_TICKS);
}
/** Start ducking this many ticks before a flying obstacle reaches the dino. */
const DUCK_LEAD_TICKS = 6;

/** Flying obstacles: between ducking height and standing head height. */
const BIRD_BOTTOM = 18;

/**
 * The Choice question asked about the scene text. This wording was picked
 * from a sweep with laya_s256 (13/13 correct on the game's sentences).
 */
const DINO_QUESTION = {
  instructions: 'What should the running dino do?',
  options: [
    { label: 'jump', description: 'an obstacle on the ground' },
    { label: 'duck', description: 'an obstacle flying low' },
    { label: 'wait', description: 'no obstacle, the path is clear' },
  ],
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Action = 'JUMP' | 'DUCK' | 'WAIT';
const ACTIONS: Action[] = ['JUMP', 'DUCK', 'WAIT'];

interface ObstacleKind {
  name: string;
  flying: boolean;
  width: number;
  height: number;
  /** Ground obstacles: number of cacti drawn side by side. */
  count?: number;
}

const OBSTACLE_KINDS: ObstacleKind[] = [
  { name: 'small cactus', flying: false, width: 14, height: 26 },
  { name: 'cactus', flying: false, width: 18, height: 36 },
  { name: 'tall cactus', flying: false, width: 18, height: 48 },
  { name: 'two cacti', flying: false, width: 34, height: 32, count: 2 },
  { name: 'three cacti', flying: false, width: 48, height: 28, count: 3 },
  { name: 'bird', flying: true, width: 26, height: 16 },
  { name: 'pterodactyl', flying: true, width: 26, height: 16 },
];

interface Obstacle {
  id: number;
  kind: ObstacleKind;
  x: number;
  passed: boolean;
}

interface World {
  tick: number;
  speed: number;
  dinoY: number; // height above the ground
  dinoVy: number;
  ducking: boolean;
  obstacles: Obstacle[];
  nextId: number;
  nextSpawnIn: number;
  scroll: number; // total distance travelled, for ground animation
  cleared: number;
  gameOver: boolean;
}

/** What the agent can "see" each tick. */
interface Observation {
  onGround: boolean;
  speed: number;
  /** The next obstacle, if it is within lookDistance(speed). */
  target: Obstacle | null;
  distance: number | null; // gap between the dino and the target
}

interface Decision {
  action: Action;
  reason: string;
  /** Obstacle id this decision is about (null = clear path). */
  target: number | null;
  /** Text sent to the model, if it was called. */
  prompt?: string;
  probabilities?: Record<Action, number>;
  inferenceTime?: number;
  /** Placeholder while an async model call is in flight. */
  thinking?: boolean;
}

// ---------------------------------------------------------------------------
// Scene text and timing
// ---------------------------------------------------------------------------

/** The text the model reads: what's ahead, in words only. */
function describe(obs: Observation): string {
  if (!obs.target) return 'The path ahead is clear.';
  const { name, flying } = obs.target.kind;
  return flying ? `Low flying obstacle ahead: ${name}.` : `Ground obstacle ahead: ${name}.`;
}

/** Distance at which a jump puts the top of the arc above the middle of the obstacle. */
function jumpDistance(obs: Observation): number {
  return obs.speed * APEX_TIME - (obs.target!.kind.width + DINO_W) / 2;
}

/** The moves to perform this tick for the current decision (timing only). */
function controls(obs: Observation, decision: Decision): { jump: boolean; duck: boolean } {
  const none = { jump: false, duck: false };
  if (!obs.target || decision.target !== obs.target.id || decision.thinking || !obs.onGround) return none;
  if (decision.action === 'JUMP') return { jump: obs.distance! <= jumpDistance(obs), duck: false };
  if (decision.action === 'DUCK') return { jump: false, duck: obs.distance! <= obs.speed * DUCK_LEAD_TICKS };
  return none;
}

// ---------------------------------------------------------------------------
// World simulation
// ---------------------------------------------------------------------------

/** Small seeded PRNG so a run is reproducible while stepping. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createWorld(): World {
  return {
    tick: 0,
    speed: START_SPEED,
    dinoY: 0,
    dinoVy: 0,
    ducking: false,
    obstacles: [],
    nextId: 1,
    nextSpawnIn: 300,
    scroll: 0,
    cleared: 0,
    gameOver: false,
  };
}

function observe(world: World): Observation {
  const next = world.obstacles.find((o) => !o.passed);
  const distance = next ? Math.max(0, next.x - (DINO_X + DINO_W)) : null;
  const visible = next && distance! <= lookDistance(world.speed);
  return {
    onGround: world.dinoY === 0,
    speed: world.speed,
    target: visible ? next : null,
    distance: visible ? distance : null,
  };
}

/** Advance the world by one tick. */
function stepWorld(world: World, input: { jump: boolean; duck: boolean }, rand: () => number) {
  world.tick++;
  world.scroll += world.speed;

  // Dino physics.
  if (input.jump && world.dinoY === 0) world.dinoVy = JUMP_VELOCITY;
  world.dinoVy -= GRAVITY;
  world.dinoY = Math.max(0, world.dinoY + world.dinoVy);
  if (world.dinoY === 0) world.dinoVy = 0;
  world.ducking = input.duck && world.dinoY === 0;

  // Move obstacles; count the ones the dino got past.
  for (const o of world.obstacles) {
    o.x -= world.speed;
    if (!o.passed && o.x + o.kind.width < DINO_X) {
      o.passed = true;
      world.cleared++;
    }
  }
  world.obstacles = world.obstacles.filter((o) => o.x + o.kind.width > -10);

  // Spawn obstacles with a gap that is always long enough to recover in between.
  world.nextSpawnIn -= world.speed;
  if (world.nextSpawnIn <= 0) {
    const kind = OBSTACLE_KINDS[Math.floor(rand() * OBSTACLE_KINDS.length)];
    world.obstacles.push({ id: world.nextId++, kind, x: WIDTH, passed: false });
    const minGap = world.speed * AIR_TIME * 1.3;
    world.nextSpawnIn = minGap + rand() * 400;
  }

  world.speed = Math.min(MAX_SPEED, world.speed + ACCELERATION);

  // Collision (axis-aligned boxes, measured upward from the ground).
  const dinoW = world.ducking ? DUCK_W : DINO_W;
  const dinoH = world.ducking ? DUCK_H : DINO_H;
  for (const o of world.obstacles) {
    const bottom = o.kind.flying ? BIRD_BOTTOM : 0;
    const overlapX = DINO_X + dinoW > o.x + 2 && DINO_X < o.x + o.kind.width - 2;
    const overlapY = world.dinoY < bottom + o.kind.height - 2 && world.dinoY + dinoH > bottom + 2;
    if (overlapX && overlapY) {
      world.gameOver = true;
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Sprites
// ---------------------------------------------------------------------------

const DINO_COLOR = '#535353';
const OBSTACLE_COLOR = '#202124';
const PIXEL = 2; // each sprite pixel is 2x2 canvas px

/** Pixel-art T-rex body (rows 0-11), facing right. '.' = empty, 'o' = eye. */
const DINO_BODY = [
  '......XXXXX.',
  '.....XXoXXXX',
  '.....XXXXXXX',
  '.....XXXXXXX',
  '.....XXXX...',
  '.....XXXXXX.',
  'X...XXXXX...',
  'XX.XXXXXXXX.',
  'XXXXXXXXX.X.',
  '.XXXXXXXX...',
  '..XXXXXXX...',
  '...XXXXX....',
];

/** Leg rows (12-13): standing/jumping plus two running frames. */
const DINO_LEGS = {
  stand: ['...XX.XX....', '...XX.XX....'],
  run1: ['...XX.XX....', '...XX.......'],
  run2: ['...XX.XX....', '......XX....'],
};

/** Ducking T-rex (16x6 body + 2 leg rows = DUCK_W x DUCK_H). */
const DINO_DUCK_BODY = [
  '..........XXXXX.',
  'X.XXXXXXXXXoXXXX',
  'XXXXXXXXXXXXXXXX',
  '.XXXXXXXXXXXXXXX',
  '..XXXXXXXXXXX...',
  '...XXXXXXXX.X...',
];
const DINO_DUCK_LEGS = {
  run1: ['....XX..XX......', '....X....XX.....'],
  run2: ['....XX..XX......', '.....XX..X......'],
};

/** Pterodactyl-style bird facing left, wings up / down (13x8 = 26x16 px). */
const BIRD_FRAMES = [
  [
    '.....X.......',
    '.....XX......',
    '.XX..XXX.....',
    'XXXXXXXXXXXXX',
    '....XXXXXXX..',
    '.....XXX.....',
    '.............',
    '.............',
  ],
  [
    '.............',
    '.............',
    '.XX..........',
    'XXXXXXXXXXXXX',
    '....XXXXXXX..',
    '.....XXX.....',
    '.....XX......',
    '.....X.......',
  ],
];

function drawSprite(ctx: CanvasRenderingContext2D, rows: string[], x: number, y: number, crashed = false) {
  rows.forEach((row, r) => {
    for (let c = 0; c < row.length; c++) {
      const ch = crashed && row[c] === 'o' ? 'x' : row[c];
      if (ch === '.') continue;
      ctx.fillStyle = ch === 'o' ? '#fff' : ch === 'x' ? '#d93025' : DINO_COLOR;
      ctx.fillRect(x + c * PIXEL, y + r * PIXEL, PIXEL, PIXEL);
    }
  });
}

/** A black pixel-art cactus (trunk + two arms) at x with the given size. */
function drawCactus(ctx: CanvasRenderingContext2D, x: number, width: number, height: number) {
  const cols = Math.max(5, Math.round(width / PIXEL));
  const rows = Math.round(height / PIXEL);
  const top = GROUND_Y - rows * PIXEL;
  const px = (c: number, r: number, w: number, h: number) =>
    ctx.fillRect(x + c * PIXEL, top + r * PIXEL, w * PIXEL, h * PIXEL);

  ctx.fillStyle = OBSTACLE_COLOR;

  // Trunk with a rounded top.
  const trunkW = Math.max(3, Math.round(cols * 0.34));
  const trunkL = Math.floor((cols - trunkW) / 2);
  px(trunkL + 1, 0, trunkW - 2, 1);
  px(trunkL, 1, trunkW, rows - 1);

  // Arms: an elbow on each side, the left one lower than the right.
  const side = trunkL;
  if (side < 2) return;
  const armW = Math.min(2, side - 1);
  const arm = (col: number, armTop: number, armBottom: number, joinFrom: number, joinTo: number) => {
    px(col, armTop + 1, armW, armBottom - armTop); // upright part
    if (armW > 1)
      px(col + 1, armTop, armW - 1, 1); // rounded tip
    else px(col, armTop, 1, 1);
    px(joinFrom, armBottom - armW + 1, joinTo - joinFrom, armW); // joint to trunk
  };
  arm(0, Math.round(rows * 0.3), Math.round(rows * 0.62), armW, trunkL);
  arm(cols - armW, Math.round(rows * 0.18), Math.round(rows * 0.48), trunkL + trunkW, cols - armW);
}

function drawObstacle(ctx: CanvasRenderingContext2D, o: Obstacle, tick: number) {
  const k = o.kind;
  if (k.flying) {
    const frame = BIRD_FRAMES[Math.floor(tick / 10) % 2];
    ctx.save();
    // Same sprite helper as the dino, but in the obstacle color.
    frame.forEach((row, r) => {
      for (let c = 0; c < row.length; c++) {
        if (row[c] === '.') continue;
        ctx.fillStyle = OBSTACLE_COLOR;
        ctx.fillRect(o.x + c * PIXEL, GROUND_Y - BIRD_BOTTOM - k.height + r * PIXEL, PIXEL, PIXEL);
      }
    });
    ctx.restore();
    return;
  }
  const n = k.count ?? 1;
  const each = Math.floor(k.width / n);
  for (let i = 0; i < n; i++) {
    // Vary heights a little within a group.
    const h = k.height - (i % 2) * 6;
    drawCactus(ctx, o.x + i * each, each, h);
  }
}

/** Ground line with scrolling pebbles. */
function drawGround(ctx: CanvasRenderingContext2D, scroll: number) {
  ctx.fillStyle = DINO_COLOR;
  ctx.fillRect(0, GROUND_Y, WIDTH, 1);
  const spacing = 37;
  const offset = scroll % spacing;
  for (let i = 0; i * spacing - offset < WIDTH + spacing; i++) {
    const x = i * spacing - offset;
    const seed = (i + Math.floor(scroll / spacing)) * 9301;
    const dy = 4 + (seed % 7);
    const w = 2 + (seed % 3);
    ctx.fillRect(x, GROUND_Y + dy, w, 1);
  }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

class DinoGameTask {
  private world = createWorld();
  private rand = mulberry32(Date.now());
  /** The model's current decision about what's ahead; performed by the game when timing is right. */
  private plan: Decision = { action: 'WAIT', reason: 'Press Play or Next Step.', target: null };
  private modelCalls = 0;

  private running = false;
  private busy = false;
  /** Shared "Inference Time" badge + history graph, same as the other tasks. */
  private inferenceTimer = new InferenceTimer();
  /** Bumped on reset/model change so stale async answers are ignored. */
  private epoch = 0;
  private inFlight = false;
  private simSpeed = 1;
  private pauseOnDecision = true;
  private disposers: (() => void)[] = [];

  private ctx!: CanvasRenderingContext2D;
  private el: Record<string, HTMLElement> = {};

  constructor(private container: HTMLElement) {}

  init() {
    this.container.innerHTML = template;
    this.container.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));
    this.container.style.setProperty('--dm-width', `${WIDTH}px`);
    this.container.style.setProperty('--dm-aspect', `${WIDTH} / ${HEIGHT}`);
    this.el['dm-question-text'].textContent = DINO_QUESTION.instructions;
    this.el['dm-options'].innerHTML = DINO_QUESTION.options
      .map((o) => `<li><b>${o.label}</b> — ${o.description}</li>`)
      .join('');

    const canvas = this.el['dm-canvas'] as HTMLCanvasElement;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    this.ctx = canvas.getContext('2d')!;
    this.ctx.scale(dpr, dpr);

    this.el['dm-play'].addEventListener('click', () => this.setRunning(!this.running));
    this.el['dm-step'].addEventListener('click', () => this.step());
    this.el['dm-reset'].addEventListener('click', () => this.reset());
    this.el['dm-speed'].addEventListener('input', (e) => {
      this.simSpeed = parseFloat((e.target as HTMLInputElement).value);
      this.el['dm-speed-value'].textContent = `${this.simSpeed}x`;
    });
    this.el['dm-pause-on-decision'].addEventListener('change', (e) => {
      this.pauseOnDecision = (e.target as HTMLInputElement).checked;
    });
    window.addEventListener('keydown', this.onKeyDown);

    // Model controls and state are shared with the Decision Maker page.
    this.disposers.push(mountDecisionModelPanel(this.el['dm-model-panel'], this.el['dm-delegate-panel']));
    this.disposers.push(
      decisionRuntime.subscribe((event) => {
        if (event.type === 'loading') {
          // A new model is loading: stop and forget answers from the old one.
          this.setRunning(false);
          this.epoch++;
          this.inFlight = false;
          this.inferenceTimer.resetRollingWindow();
        } else if (event.type === 'ready') {
          this.refreshPlan();
        } else if (event.type === 'status') {
          this.setStatus(event.text);
        } else if (event.type === 'error') {
          this.setStatus(`Error: ${event.error}`);
          this.setRunning(false);
        } else if (event.type === 'state') {
          this.updateButtons();
        }
      })
    );

    this.render();
    this.updatePanel();
    this.inferenceTimer.mount();
    if (decisionRuntime.ready) {
      this.refreshPlan();
    } else if (decisionRuntime.loading) {
      this.setStatus(`Loading ${decisionRuntime.loadingLabel}...`);
    } else {
      this.setStatus('Select a model and press "Load Model" to begin');
    }
  }

  cleanup() {
    this.running = false;
    this.epoch++;
    window.removeEventListener('keydown', this.onKeyDown);
    this.inferenceTimer.cleanup();
    // The worker and model stay loaded in decisionRuntime for the next page.
    for (const dispose of this.disposers) dispose();
    this.disposers = [];
  }

  // -------------------------------------------------------------------------
  // Model (shared runtime, runs in a worker)
  // -------------------------------------------------------------------------

  /** True if what's ahead changed since the last decision (or it is still pending). */
  private needsDecision(obs: Observation) {
    return this.plan.target !== (obs.target?.id ?? null) || (this.plan.thinking === true && !this.inFlight);
  }

  /** Asks the model about the scene: one Choice call. */
  private async modelDecide(obs: Observation): Promise<Decision> {
    const target = obs.target?.id ?? null;
    if (!decisionRuntime.ready) return { action: 'WAIT', reason: 'Model not ready.', target };

    const prompt = describe(obs);
    this.modelCalls++;
    const msg = await decisionRuntime.ask(prompt, 'choice', DINO_QUESTION);
    if (msg.type !== 'DECIDE_RESULT') return { action: 'WAIT', reason: `Model error: ${msg.error}`, target, prompt };

    const { selectedKey, probabilities } = msg.result as {
      selectedKey: string;
      probabilities: Record<string, number>;
    };
    this.inferenceTimer.record(msg.inferenceTime, decisionRuntime.delegate);
    const action = (selectedKey?.toUpperCase() as Action) ?? 'WAIT';
    const probs = Object.fromEntries(ACTIONS.map((a) => [a, probabilities?.[a.toLowerCase()] ?? 0])) as Record<
      Action,
      number
    >;
    const what = obs.target ? `“${obs.target.kind.name}”` : 'a clear path';
    return {
      action: ACTIONS.includes(action) ? action : 'WAIT',
      reason: `Model read ${what} → ${selectedKey} (${Math.round(probs[action] * 100)}%).`,
      target,
      prompt,
      probabilities: probs,
      inferenceTime: msg.inferenceTime,
    };
  }

  /**
   * While playing, ask the model in the background so the game keeps its speed.
   * An obstacle comes into view only DECISION_BUDGET_TICKS before the dino has
   * to act, so the model must answer within that budget (~100 ms).
   */
  private askModelInBackground(obs: Observation) {
    this.inFlight = true;
    const epoch = this.epoch;
    this.modelDecide(obs).then((decision) => {
      this.inFlight = false;
      if (epoch !== this.epoch) return;
      // Ignore answers about something that's no longer ahead.
      if (decision.target !== this.plan.target) return;
      this.plan = decision;
      if (this.running && this.pauseOnDecision && decision.action !== 'WAIT') this.setRunning(false);
      this.render();
      this.updatePanel();
    });
  }

  /** Blocking decision for the current frame (used when paused / stepping). */
  private async refreshPlan() {
    if (this.busy) return;
    this.busy = true;
    try {
      this.plan = await this.modelDecide(observe(this.world));
    } finally {
      this.busy = false;
    }
    this.render();
    this.updatePanel();
  }

  // -------------------------------------------------------------------------
  // Simulation control
  // -------------------------------------------------------------------------

  private onKeyDown = (e: KeyboardEvent) => {
    const target = e.target as HTMLElement;
    if (
      target &&
      ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName) &&
      target.getAttribute('type') !== 'checkbox'
    )
      return;
    if (e.code === 'Space') {
      e.preventDefault();
      this.setRunning(!this.running);
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      this.step();
    } else if (e.code === 'KeyR') {
      this.reset();
    }
  };

  private canStep() {
    return !this.world.gameOver && decisionRuntime.ready;
  }

  private setRunning(running: boolean) {
    if (running && !this.canStep()) return;
    const wasRunning = this.running;
    this.running = running;
    if (running && !wasRunning) this.runLoop();
    if (!running && wasRunning && this.plan.thinking && !this.inFlight) this.refreshPlan();
    this.updateButtons();
  }

  /** Plays at a steady speed; model answers arrive asynchronously. */
  private async runLoop() {
    while (this.running) {
      const start = performance.now();
      this.tick();
      const wait = TICK_MS / this.simSpeed - (performance.now() - start);
      await new Promise((r) => setTimeout(r, Math.max(0, wait)));
    }
  }

  /** One tick while playing (never waits for the model). */
  private tick() {
    if (this.busy || !this.canStep()) return;
    stepWorld(this.world, controls(observe(this.world), this.plan), this.rand);

    const obs = observe(this.world);
    if (this.needsDecision(obs)) {
      const target = obs.target?.id ?? null;
      if (target !== this.plan.target)
        this.plan = { action: 'WAIT', reason: 'Model is reading what’s ahead…', target, thinking: true };
      if (!this.inFlight) this.askModelInBackground(obs);
    }

    if (this.world.gameOver) this.setRunning(false);
    this.render();
    this.updatePanel();
  }

  /** One manual step: act, then (if what's ahead changed) wait for the model. */
  private async step() {
    if (this.running || this.busy || !this.canStep()) return;
    this.busy = true;
    try {
      stepWorld(this.world, controls(observe(this.world), this.plan), this.rand);
      const obs = observe(this.world);
      if (this.needsDecision(obs)) this.plan = await this.modelDecide(obs);
    } finally {
      this.busy = false;
    }
    this.render();
    this.updatePanel();
  }

  private async reset() {
    this.setRunning(false);
    this.epoch++;
    this.inFlight = false;
    this.world = createWorld();
    this.rand = mulberry32(Date.now());
    this.modelCalls = 0;
    await this.refreshPlan();
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  private setStatus(text: string) {
    this.el['status-message'].textContent = text;
    this.inferenceTimer.syncStatusVisibility(text);
    this.updateButtons();
  }

  private updateButtons() {
    const play = this.el['dm-play'] as HTMLButtonElement;
    play.querySelector('.material-icons')!.textContent = this.running ? 'pause' : 'play_arrow';
    play.querySelector('.dm-btn-label')!.textContent = this.running ? 'Pause' : 'Play';
    play.disabled = !this.canStep();
    (this.el['dm-step'] as HTMLButtonElement).disabled = this.running || !this.canStep();
  }

  private updatePanel() {
    const d = this.plan;
    this.el['dm-obs-tick'].textContent = `${this.world.tick}`;
    this.el['dm-obs-speed'].textContent = `${this.world.speed.toFixed(2)} px/t`;
    this.el['dm-obs-cleared'].textContent = `${this.world.cleared}`;
    this.el['dm-obs-calls'].textContent = `${this.modelCalls}`;

    const action = this.el['dm-action'];
    const over = this.world.gameOver;
    action.textContent = over ? 'CRASH' : d.thinking ? '…' : d.action;
    action.className = `dm-action ${over ? 'crash' : d.thinking ? '' : d.action.toLowerCase()}`;
    this.el['dm-reason'].textContent = over ? 'Hit an obstacle. Press Reset.' : d.reason;

    for (const a of ACTIONS) {
      const p = d.thinking ? undefined : d.probabilities?.[a];
      (this.el[`dm-prob-${a}`] as HTMLElement).style.width = p === undefined ? '0' : `${p * 100}%`;
      this.el[`dm-prob-value-${a}`].textContent = p === undefined ? '-' : p.toFixed(2);
    }
    if (d.prompt) this.el['dm-prompt'].textContent = d.prompt;

    if (decisionRuntime.ready) {
      this.el['status-message'].textContent = over ? 'Game over' : this.running ? 'Running' : 'Paused';
    }
    this.updateButtons();
  }

  private render() {
    const ctx = this.ctx;
    const w = this.world;
    const obs = observe(w);

    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // The zone the model "looks" at: obstacles in here are described to it.
    const look = lookDistance(w.speed);
    ctx.fillStyle = 'rgba(0, 127, 139, 0.06)';
    ctx.fillRect(DINO_X + DINO_W, 20, look, GROUND_Y - 20);
    ctx.strokeStyle = 'rgba(0, 127, 139, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(DINO_X + DINO_W + look, 20);
    ctx.lineTo(DINO_X + DINO_W + look, GROUND_Y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#007f8b';
    ctx.font = '11px Roboto, sans-serif';
    const budgetMs = Math.round((DECISION_BUDGET_TICKS * TICK_MS) / this.simSpeed);
    ctx.textAlign = 'right';
    ctx.fillText('model looks here', DINO_X + DINO_W + look - 4, 32);
    ctx.fillText(`${budgetMs} ms to decide`, DINO_X + DINO_W + look - 4, 45);
    ctx.textAlign = 'left';

    drawGround(ctx, w.scroll);

    // Obstacles with their text label; the current target also shows the model's choice.
    for (const o of w.obstacles) {
      drawObstacle(ctx, o, w.tick);
      const top = GROUND_Y - (o.kind.flying ? BIRD_BOTTOM : 0) - o.kind.height;
      const isTarget = obs.target?.id === o.id && this.plan.target === o.id;
      const label = isTarget && !this.plan.thinking ? `${o.kind.name} → ${this.plan.action}` : o.kind.name;
      ctx.font = `${isTarget ? 'bold ' : ''}11px Roboto, sans-serif`;
      ctx.fillStyle = isTarget ? '#007f8b' : '#5f6368';
      ctx.textAlign = 'center';
      ctx.fillText(label, o.x + o.kind.width / 2, top - 6);
      ctx.textAlign = 'left';
    }

    // Dino (legs alternate every 6 ticks while running).
    const frame = Math.floor(w.tick / 6) % 2;
    if (w.ducking && !w.gameOver) {
      const legs = frame ? DINO_DUCK_LEGS.run1 : DINO_DUCK_LEGS.run2;
      drawSprite(ctx, [...DINO_DUCK_BODY, ...legs], DINO_X, GROUND_Y - DUCK_H);
    } else {
      const legs = w.dinoY > 0 || w.gameOver ? DINO_LEGS.stand : frame ? DINO_LEGS.run1 : DINO_LEGS.run2;
      drawSprite(ctx, [...DINO_BODY, ...legs], DINO_X, GROUND_Y - w.dinoY - DINO_H, w.gameOver);
    }

    if (w.gameOver) {
      ctx.fillStyle = DINO_COLOR;
      ctx.font = 'bold 20px Roboto Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('GAME OVER', WIDTH / 2, HEIGHT / 2);
      ctx.textAlign = 'left';
    }
  }
}

let activeTask: DinoGameTask | null = null;

export async function setupDinoGame(container: HTMLElement) {
  activeTask = new DinoGameTask(container);
  activeTask.init();
}

export function cleanupDinoGame() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
