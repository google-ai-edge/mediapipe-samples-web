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
 * Supports two input modalities:
 * - Scene Image (Vision): captures the full 768×192 game scene (dino on the
 *   left, shaded near action zone, and all near & far obstacles at their
 *   continuous track positions—with no text labels) at the vision encoder's
 *   exact native 48×48 pooled-patch multiple (768×192 = 16×4 pooled tokens,
 *   bypassing CPU stbir_resize) and compares it against 6 canonical
 *   single-obstacle reference prototypes (`options[].imageBytes`) in
 *   DecisionMaker so the model generalizes across clear track, far obstacles
 *   (`wait`), near ground/flying obstacles (`jump` / `duck`), and multi-obstacle
 *   scenes without enumerating combined states.
 * - Text State: describes what is ahead in words ("Low flying obstacle ahead:
 *   pterodactyl.") and asks a Choice question: jump, duck, or wait.
 */

import template from '../templates/dino-game.html?raw';
import { InferenceTimer } from '../components/inference-timer';
import { DECISION_MODELS, decisionRuntime } from '../components/decision-runtime';
import { mountDecisionModelPanel, type DecisionModelPanelHandle } from '../components/decision-model-panel';

// ---------------------------------------------------------------------------
// Constants (all units are pixels and ticks; one tick = 1/60 s at 1x speed)
// ---------------------------------------------------------------------------

const WIDTH = 800;
const HEIGHT = 220;
const GROUND_Y = 180;

/**
 * Vision scene dimensions: 768×192 is an exact 16×4 multiple of the ViT vision
 * encoder's 48×48 pooled-patch size (`patch_width=16 * pooling_kernel=3 = 48`),
 * so `GetAspectRatioPreservingSize(768, 192)` returns `768×192` and completely
 * bypasses CPU `stbir_resize` (saving ~55 ms per frame and preserving crisp
 * 1:1 pixel-art sprites).
 */
const SCENE_WIDTH = 768;
const SCENE_HEIGHT = 192;
const SCENE_GROUND_Y = 164;
const SCENE_ZONE_W = 196;
const SCENE_PROTO_NEAR_X = 198;
const SCENE_PROTO_FAR_X = 404;
const SCENE_NEAR_ENTRY_X = 200;

const DINO_X = 60;
const DINO_W = 24;
const DINO_H = 28;
const DUCK_W = 32;
const DUCK_H = 16;

const GRAVITY = 0.6;
const JUMP_VELOCITY = 11;
const APEX_TIME = JUMP_VELOCITY / GRAVITY; // ticks to reach the top of a jump

const START_SPEED = 5;
const MAX_SPEED = 10;
const ACCELERATION = 0.001;

const TICK_MS = 1000 / 60;

export type InputMode = 'image' | 'text';

/**
 * Lead ticks before the jump apex when an obstacle enters the near action zone.
 * In image mode, `10.5` lead ticks triggers the background vision inference as
 * the obstacle enters the near action zone (`sceneX ≈ 198px`, ~15–23 ticks
 * before the jump/duck point) so the dino keeps running smoothly while awaiting
 * the worker's decision.
 */
const DECISION_BUDGET_TICKS_IMAGE = 10.5;
const DECISION_BUDGET_TICKS_TEXT = 6;

function budgetTicksForMode(mode: InputMode): number {
  return mode === 'image' ? DECISION_BUDGET_TICKS_IMAGE : DECISION_BUDGET_TICKS_TEXT;
}

/** The model is asked about a near obstacle once it is this close (grows with speed). */
function lookDistance(speed: number, mode: InputMode = 'image'): number {
  return speed * (APEX_TIME + budgetTicksForMode(mode));
}
/** Start ducking this many ticks before a flying obstacle reaches the dino. */
const DUCK_LEAD_TICKS = 6;

/** Flying obstacles: between ducking height and standing head height. */
const BIRD_BOTTOM = 22;

/**
 * The Choice question asked about the scene text. This wording was picked
 * from a sweep with laya_s256 (13/13 correct on the game's sentences).
 */
const DINO_TEXT_QUESTION = {
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
  { name: 'small cactus', flying: false, width: 18, height: 30 },
  { name: 'cactus', flying: false, width: 18, height: 36 },
  { name: 'tall cactus', flying: false, width: 18, height: 44 },
  { name: 'two cacti', flying: false, width: 38, height: 34, count: 2 },
  { name: 'three cacti', flying: false, width: 58, height: 34, count: 3 },
  { name: 'bird', flying: true, width: 26, height: 16 },
  { name: 'pterodactyl', flying: true, width: 26, height: 16 },
];

function findObstacleKind(name: string): ObstacleKind {
  return OBSTACLE_KINDS.find((k) => k.name === name) ?? OBSTACLE_KINDS[1];
}

interface PrototypeObstacleSpec {
  kindName: string;
  x: number;
}

interface ScenePrototypeSpec {
  id: string;
  action: Action;
  shortTitle: string;
  obstacles: PrototypeObstacleSpec[];
}

/**
 * 6 canonical single-obstacle full-game scene prototypes (768×192) encoded as
 * `options[].imageBytes`. Every prototype contains at most 1 obstacle (zero
 * multi-obstacle combinations); the vision model generalizes across obstacle
 * sizes, counts, continuous positions, and multi-obstacle (near + far) scenes.
 */
const SCENE_PROTOTYPE_SPECS: ScenePrototypeSpec[] = [
  { id: 'wait_clear', action: 'WAIT', shortTitle: 'clear track', obstacles: [] },
  {
    id: 'wait_far_cactus',
    action: 'WAIT',
    shortTitle: 'far ground obstacle',
    obstacles: [{ kindName: 'cactus', x: SCENE_PROTO_FAR_X }],
  },
  {
    id: 'wait_far_bird',
    action: 'WAIT',
    shortTitle: 'far flying obstacle',
    obstacles: [{ kindName: 'bird', x: SCENE_PROTO_FAR_X }],
  },
  {
    id: 'jump_near_cactus',
    action: 'JUMP',
    shortTitle: 'near cactus',
    obstacles: [{ kindName: 'cactus', x: SCENE_PROTO_NEAR_X }],
  },
  {
    id: 'jump_near_group',
    action: 'JUMP',
    shortTitle: 'near cactus group',
    obstacles: [{ kindName: 'three cacti', x: SCENE_PROTO_NEAR_X }],
  },
  {
    id: 'duck_near_bird',
    action: 'DUCK',
    shortTitle: 'near flying obstacle',
    obstacles: [{ kindName: 'bird', x: SCENE_PROTO_NEAR_X }],
  },
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
  /** The obstacle inside the near lookahead action zone, if any. */
  target: Obstacle | null;
  distance: number | null; // gap between the dino and `target`
  /** In image mode, a far obstacle on the track outside the near action zone. */
  farTarget: Obstacle | null;
  farDistance: number | null;
  /** Key identifying the current decision scenario ('clear', 'far:<id>', 'near:<id>'). */
  sceneKey: string;
}

interface Decision {
  action: Action;
  reason: string;
  /** Near obstacle id this decision will act on (null = clear or far-only). */
  target: number | null;
  /** Scene key this decision was computed for ('clear', 'far:<id>', 'near:<id>'). */
  sceneKey: string;
  /** Prototype ID matched in image mode. */
  matchedProtoId?: string;
  /** Text sent to the model, if it was called in text mode. */
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
    nextSpawnIn: 20,
    scroll: 0,
    cleared: 0,
    gameOver: false,
  };
}

function normalizedSceneX(worldX: number, speed: number): number {
  const zoneX = DINO_X + DINO_W;
  const scale = (SCENE_NEAR_ENTRY_X - zoneX) / lookDistance(speed, 'image');
  return Math.round(zoneX + Math.max(0, worldX - zoneX) * scale);
}

function observe(world: World, mode: InputMode = 'image'): Observation {
  const maxSceneX = mode === 'image' ? SCENE_WIDTH - 4 : WIDTH - 4;
  const unpassed = world.obstacles.filter((o) => !o.passed && normalizedSceneX(o.x, world.speed) < maxSceneX);
  const next = unpassed[0] ?? null;
  const distance = next ? Math.max(0, next.x - (DINO_X + DINO_W)) : null;
  const look = lookDistance(world.speed, mode);
  const inNearZone = Boolean(next && distance !== null && distance <= look);

  if (mode === 'text') {
    const target = inNearZone ? next : null;
    return {
      onGround: world.dinoY === 0,
      speed: world.speed,
      target,
      distance: target ? distance : null,
      farTarget: null,
      farDistance: null,
      sceneKey: target ? `near:${target.id}` : 'clear',
    };
  }

  if (inNearZone && next) {
    const second =
      unpassed.find((o) => o.id !== next.id && normalizedSceneX(o.x, world.speed) > DINO_X + DINO_W + SCENE_ZONE_W) ??
      null;
    const secondDist = second ? Math.max(0, second.x - (DINO_X + DINO_W)) : null;
    return {
      onGround: world.dinoY === 0,
      speed: world.speed,
      target: next,
      distance,
      farTarget: second,
      farDistance: secondDist,
      sceneKey: `near:${next.id}`,
    };
  }

  const farCandidate = next;
  const farDist = farCandidate ? Math.max(0, farCandidate.x - (DINO_X + DINO_W)) : null;
  return {
    onGround: world.dinoY === 0,
    speed: world.speed,
    target: null,
    distance: null,
    farTarget: farCandidate,
    farDistance: farDist,
    sceneKey: farCandidate ? `far:${farCandidate.id}` : 'clear',
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

  // Spawn obstacles with a ~435px scene gap so:
  // - When an obstacle is evaluated in the far zone (~404px), it is alone on track.
  // - When it enters the near action zone (~198px), the next obstacle is visible on the far track (~630px).
  world.nextSpawnIn -= world.speed;
  if (world.nextSpawnIn <= 0) {
    const kind = OBSTACLE_KINDS[Math.floor(rand() * OBSTACLE_KINDS.length)];
    const zoneX = DINO_X + DINO_W;
    const scale = (SCENE_NEAR_ENTRY_X - zoneX) / lookDistance(world.speed, 'image');
    const spawnX = Math.max(WIDTH, Math.round(zoneX + (SCENE_WIDTH - zoneX) / scale));
    world.obstacles.push({ id: world.nextId++, kind, x: spawnX, passed: false });
    const gapScene = 428 + rand() * 16;
    world.nextSpawnIn = gapScene / scale;
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

/** A pixel-art cactus (trunk + two arms) at x with the given size. */
function drawCactus(
  ctx: CanvasRenderingContext2D,
  x: number,
  width: number,
  height: number,
  groundY = GROUND_Y,
  color = OBSTACLE_COLOR
) {
  const cols = Math.max(9, Math.round(width / PIXEL));
  const rows = Math.round(height / PIXEL);
  const top = groundY - rows * PIXEL;
  const px = (c: number, r: number, w: number, h: number) =>
    ctx.fillRect(x + c * PIXEL, top + r * PIXEL, w * PIXEL, h * PIXEL);

  ctx.fillStyle = color;

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

function drawObstacle(
  ctx: CanvasRenderingContext2D,
  o: { kind: ObstacleKind; x: number },
  tick: number,
  groundY = GROUND_Y,
  birdBottom = BIRD_BOTTOM,
  overrideX?: number,
  color = OBSTACLE_COLOR
) {
  const k = o.kind;
  const drawX = overrideX ?? o.x;
  if (k.flying) {
    const frame = BIRD_FRAMES[Math.floor(tick / 10) % 2];
    ctx.save();
    // Same sprite helper as the dino, but in the obstacle color.
    frame.forEach((row, r) => {
      for (let c = 0; c < row.length; c++) {
        if (row[c] === '.') continue;
        ctx.fillStyle = color;
        ctx.fillRect(drawX + c * PIXEL, groundY - birdBottom - k.height + r * PIXEL, PIXEL, PIXEL);
      }
    });
    ctx.restore();
    return;
  }
  const n = k.count ?? 1;
  const gap = n > 1 ? 2 : 0;
  const each = Math.floor((k.width - gap * (n - 1)) / n);
  for (let i = 0; i < n; i++) {
    // Vary heights a little within a group.
    const h = k.height - (i % 2) * 6;
    drawCactus(ctx, drawX + i * (each + gap), each, h, groundY, color);
  }
}

/** Flying obstacle altitude in the 768×192 full-scene vision view (`y ∈ [114, 130]`, cleanly above ground cacti `y ∈ [120, 164]`). */
const FULL_SCENE_BIRD_BOTTOM = 34;
const TGA_HEADER_BYTES = 18;

let cachedBaseTgaBytes: Uint8Array | null = null;
let cachedBaseSceneCanvas: HTMLCanvasElement | null = null;
let cachedMistOverlayCanvas: HTMLCanvasElement | null = null;

/**
 * Builds and caches the static 768×192 24-bit uncompressed TGA buffer
 * (white sky, stippled near action zone, #007f8b border, ground line, dino
 * sprite, and far-track horizon mist). Encoding a live frame only copies this
 * 442 KB buffer (`~0.03 ms`) and stamps the 1–2 obstacle sprites (`~0.01 ms`).
 */
function getBaseSceneTgaBytes(): Uint8Array {
  if (cachedBaseTgaBytes) return cachedBaseTgaBytes;

  const pixelBytes = SCENE_WIDTH * SCENE_HEIGHT * 3;
  const tga = new Uint8Array(TGA_HEADER_BYTES + pixelBytes);
  tga[2] = 2; // uncompressed true-color
  tga[12] = SCENE_WIDTH & 0xff;
  tga[13] = (SCENE_WIDTH >> 8) & 0xff;
  tga[14] = SCENE_HEIGHT & 0xff;
  tga[15] = (SCENE_HEIGHT >> 8) & 0xff;
  tga[16] = 24; // 24-bit BGR
  tga[17] = 0x20; // top-left origin
  tga.fill(255, TGA_HEADER_BYTES);

  const setPixel = (x: number, y: number, r: number, g: number, b: number) => {
    const off = TGA_HEADER_BYTES + (y * SCENE_WIDTH + x) * 3;
    tga[off] = b;
    tga[off + 1] = g;
    tga[off + 2] = r;
  };
  const fillRect = (x0: number, y0: number, w: number, h: number, r: number, g: number, b: number) => {
    const xStart = Math.max(0, x0);
    const xEnd = Math.min(SCENE_WIDTH, x0 + w);
    const yStart = Math.max(0, y0);
    const yEnd = Math.min(SCENE_HEIGHT, y0 + h);
    for (let y = yStart; y < yEnd; y++) {
      for (let x = xStart; x < xEnd; x++) {
        setPixel(x, y, r, g, b);
      }
    }
  };

  const zoneX = DINO_X + DINO_W;
  const zoneEnd = zoneX + SCENE_ZONE_W;
  for (let y = 16; y < SCENE_GROUND_Y; y++) {
    const yTile = Math.floor(y / 4);
    for (let x = zoneX; x < zoneEnd; x++) {
      const even = (Math.floor(x / 4) + yTile) % 2 === 0;
      setPixel(x, y, even ? 182 : 166, even ? 236 : 222, even ? 244 : 230);
    }
  }
  fillRect(zoneEnd, 16, 2, SCENE_GROUND_Y - 16, 0, 127, 139);
  fillRect(0, SCENE_GROUND_Y, SCENE_WIDTH, 2, 83, 83, 83);

  const dinoRows = [...DINO_BODY, ...DINO_LEGS.stand];
  for (let r = 0; r < dinoRows.length; r++) {
    const row = dinoRows[r];
    for (let c = 0; c < row.length; c++) {
      const ch = row[c];
      if (ch === '.') continue;
      const v = ch === 'o' ? 255 : 83;
      fillRect(DINO_X + c * PIXEL, SCENE_GROUND_Y - DINO_H + r * PIXEL, PIXEL, PIXEL, v, v, v);
    }
  }

  for (let y = 16; y < SCENE_GROUND_Y; y++) {
    const yTile = Math.floor(y / 4);
    for (let x = zoneEnd + 2; x < SCENE_WIDTH; x++) {
      const alpha = x <= 445 ? 0.2 : 0.2 + Math.min(1, (x - 445) / 95) * 0.8;
      const fogVal = (Math.floor(x / 4) + yTile) % 2 === 0 ? 248 : 232;
      const c = Math.trunc(255 * (1 - alpha) + fogVal * alpha);
      setPixel(x, y, c, c, c);
    }
  }

  cachedBaseTgaBytes = tga;
  return tga;
}

/**
 * Synchronously encodes the 768×192 full-scene image as a 24-bit uncompressed
 * TGA buffer (`< 0.05 ms`, zero PNG compression/decompression overhead, and
 * bypasses `stbir_resize` inside `stb_image_preprocessor.cc`).
 */
function renderFullSceneTgaBytes(
  obstacles: Array<{ kind: ObstacleKind; x: number }>,
  zoneW = SCENE_ZONE_W
): Uint8Array {
  const tga = getBaseSceneTgaBytes().slice();
  const zoneEnd = DINO_X + DINO_W + zoneW;

  const fillObsRect = (x0: number, y0: number, w: number, h: number) => {
    const xStart = Math.max(0, x0);
    const xEnd = Math.min(SCENE_WIDTH, x0 + w);
    const yStart = Math.max(0, y0);
    const yEnd = Math.min(SCENE_HEIGHT, y0 + h);
    for (let y = yStart; y < yEnd; y++) {
      const yTile = Math.floor(y / 4);
      const inMistY = y >= 16 && y < SCENE_GROUND_Y;
      for (let x = xStart; x < xEnd; x++) {
        let r = 32;
        let g = 33;
        let b = 36;
        if (inMistY && x >= zoneEnd + 2) {
          const alpha = x <= 445 ? 0.2 : 0.2 + Math.min(1, (x - 445) / 95) * 0.8;
          const fogVal = (Math.floor(x / 4) + yTile) % 2 === 0 ? 248 : 232;
          r = Math.trunc(r * (1 - alpha) + fogVal * alpha);
          g = Math.trunc(g * (1 - alpha) + fogVal * alpha);
          b = Math.trunc(b * (1 - alpha) + fogVal * alpha);
        }
        const off = TGA_HEADER_BYTES + (y * SCENE_WIDTH + x) * 3;
        tga[off] = b;
        tga[off + 1] = g;
        tga[off + 2] = r;
      }
    }
  };

  const drawCactusTga = (x: number, width: number, height: number) => {
    const cols = Math.max(9, Math.round(width / PIXEL));
    const rows = Math.round(height / PIXEL);
    const top = SCENE_GROUND_Y - rows * PIXEL;
    const px = (c: number, r: number, w: number, h: number) =>
      fillObsRect(x + c * PIXEL, top + r * PIXEL, w * PIXEL, h * PIXEL);

    const trunkW = Math.max(3, Math.round(cols * 0.34));
    const trunkL = Math.floor((cols - trunkW) / 2);
    px(trunkL + 1, 0, trunkW - 2, 1);
    px(trunkL, 1, trunkW, rows - 1);

    const side = trunkL;
    if (side < 2) return;
    const armW = Math.min(2, side - 1);
    const arm = (col: number, armTop: number, armBottom: number, joinFrom: number, joinTo: number) => {
      px(col, armTop + 1, armW, armBottom - armTop);
      if (armW > 1) px(col + 1, armTop, armW - 1, 1);
      else px(col, armTop, 1, 1);
      px(joinFrom, armBottom - armW + 1, joinTo - joinFrom, armW);
    };
    arm(0, Math.round(rows * 0.3), Math.round(rows * 0.62), armW, trunkL);
    arm(cols - armW, Math.round(rows * 0.18), Math.round(rows * 0.48), trunkL + trunkW, cols - armW);
  };

  const birdFrame = BIRD_FRAMES[0];
  for (const o of obstacles) {
    if (o.x + o.kind.width < 0 || o.x >= SCENE_WIDTH) continue;
    if (o.kind.flying) {
      for (let r = 0; r < birdFrame.length; r++) {
        const row = birdFrame[r];
        for (let c = 0; c < row.length; c++) {
          if (row[c] === '.') continue;
          fillObsRect(
            o.x + c * PIXEL,
            SCENE_GROUND_Y - FULL_SCENE_BIRD_BOTTOM - o.kind.height + r * PIXEL,
            PIXEL,
            PIXEL
          );
        }
      }
    } else {
      const n = o.kind.count ?? 1;
      const gap = n > 1 ? 2 : 0;
      const each = Math.floor((o.kind.width - gap * (n - 1)) / n);
      for (let i = 0; i < n; i++) {
        const h = o.kind.height - (i % 2) * 6;
        drawCactusTga(o.x + i * (each + gap), each, h);
      }
    }
  }

  return tga;
}

function getCachedSceneCanvases(): { baseCanvas: HTMLCanvasElement; mistCanvas: HTMLCanvasElement } {
  if (cachedBaseSceneCanvas && cachedMistOverlayCanvas) {
    return { baseCanvas: cachedBaseSceneCanvas, mistCanvas: cachedMistOverlayCanvas };
  }
  const baseCanvas = document.createElement('canvas');
  baseCanvas.width = SCENE_WIDTH;
  baseCanvas.height = SCENE_HEIGHT;
  const bctx = baseCanvas.getContext('2d')!;
  const baseImg = bctx.createImageData(SCENE_WIDTH, SCENE_HEIGHT);
  const baseData = baseImg.data;
  const baseTga = getBaseSceneTgaBytes();
  for (let i = 0; i < SCENE_WIDTH * SCENE_HEIGHT; i++) {
    const tgaOff = TGA_HEADER_BYTES + i * 3;
    const rgbaOff = i * 4;
    baseData[rgbaOff] = baseTga[tgaOff + 2];
    baseData[rgbaOff + 1] = baseTga[tgaOff + 1];
    baseData[rgbaOff + 2] = baseTga[tgaOff];
    baseData[rgbaOff + 3] = 255;
  }
  bctx.putImageData(baseImg, 0, 0);

  const mistCanvas = document.createElement('canvas');
  mistCanvas.width = SCENE_WIDTH;
  mistCanvas.height = SCENE_HEIGHT;
  const mctx = mistCanvas.getContext('2d')!;
  const mistImg = mctx.createImageData(SCENE_WIDTH, SCENE_HEIGHT);
  const mistData = mistImg.data;
  const zoneEnd = DINO_X + DINO_W + SCENE_ZONE_W;
  for (let y = 16; y < SCENE_GROUND_Y; y++) {
    const yTile = Math.floor(y / 4);
    for (let x = zoneEnd + 2; x < SCENE_WIDTH; x++) {
      const alpha = x <= 445 ? 0.2 : 0.2 + Math.min(1, (x - 445) / 95) * 0.8;
      const fogVal = (Math.floor(x / 4) + yTile) % 2 === 0 ? 248 : 232;
      const idx = (y * SCENE_WIDTH + x) * 4;
      mistData[idx] = fogVal;
      mistData[idx + 1] = fogVal;
      mistData[idx + 2] = fogVal;
      mistData[idx + 3] = Math.round(alpha * 255);
    }
  }
  mctx.putImageData(mistImg, 0, 0);

  cachedBaseSceneCanvas = baseCanvas;
  cachedMistOverlayCanvas = mistCanvas;
  return { baseCanvas, mistCanvas };
}

/**
 * Renders the full 768×192 game scene onto a preview canvas using cached
 * offscreen layers (`drawImage`) so 60 FPS rendering never triggers CPU
 * `getImageData` / `putImageData` pipeline stalls.
 */
function renderFullSceneCanvas(
  canvas: HTMLCanvasElement,
  obstacles: Array<{ kind: ObstacleKind; x: number }>,
  _zoneW = SCENE_ZONE_W
) {
  if (canvas.width !== SCENE_WIDTH) canvas.width = SCENE_WIDTH;
  if (canvas.height !== SCENE_HEIGHT) canvas.height = SCENE_HEIGHT;
  const sctx = canvas.getContext('2d')!;
  const { baseCanvas, mistCanvas } = getCachedSceneCanvases();
  sctx.drawImage(baseCanvas, 0, 0);

  for (const o of obstacles) {
    if (o.x + o.kind.width < 0 || o.x >= SCENE_WIDTH) continue;
    drawObstacle(sctx, o, 0, SCENE_GROUND_Y, FULL_SCENE_BIRD_BOTTOM, undefined, OBSTACLE_COLOR);
  }

  sctx.drawImage(mistCanvas, 0, 0);
}

function getWorldSceneObstacles(w: World): Array<{ kind: ObstacleKind; x: number }> {
  return w.obstacles
    .filter((o) => !o.passed)
    .map((o) => ({ kind: o.kind, x: normalizedSceneX(o.x, w.speed) }))
    .filter((o) => o.x + o.kind.width > 0 && o.x < SCENE_WIDTH);
}

/**
 * Renders the live world onto the 768×192 full-scene preview canvas using each
 * obstacle's continuous speed-normalized track coordinate.
 */
function renderFullSceneFromWorld(canvas: HTMLCanvasElement, w: World) {
  renderFullSceneCanvas(canvas, getWorldSceneObstacles(w), SCENE_ZONE_W);
}

function renderFullSceneTgaFromWorld(w: World): Uint8Array {
  return renderFullSceneTgaBytes(getWorldSceneObstacles(w), SCENE_ZONE_W);
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
  private plan: Decision = {
    action: 'WAIT',
    reason: 'Press Play or Next Step.',
    target: null,
    sceneKey: 'init',
  };
  private modelCalls = 0;
  private inputMode: InputMode = 'image';

  private running = false;
  private busy = false;
  /** Shared "Inference Time" badge + history graph, same as the other tasks. */
  private inferenceTimer = new InferenceTimer({ mode: 'single', labelA: 'Decision' });
  /** Bumped on reset/model change so stale async answers are ignored. */
  private epoch = 0;
  private inFlight = false;
  private simSpeed = 1;
  private pauseOnDecision = false;
  private disposers: (() => void)[] = [];
  private modelPanelHandle?: DecisionModelPanelHandle;

  private ctx!: CanvasRenderingContext2D;
  private sceneCanvas!: HTMLCanvasElement;
  private el: Record<string, HTMLElement> = {};
  private protoCards = new Map<string, HTMLElement>();
  private pendingDecisionRequest?: { obs: Observation; snapshotTga?: Uint8Array };
  private imageQuestionPromise?: Promise<{
    instructions: string;
    temperature: number;
    options: Array<{ label: string; description: string; imageBytes: Uint8Array }>;
  }>;

  constructor(private container: HTMLElement) {}

  init() {
    this.container.innerHTML = template;
    this.container.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));
    this.container.style.setProperty('--dm-width', `${WIDTH}px`);
    this.container.style.setProperty('--dm-aspect', `${WIDTH} / ${HEIGHT}`);
    this.el['dm-question-text'].textContent = DINO_TEXT_QUESTION.instructions;
    this.el['dm-options'].innerHTML = DINO_TEXT_QUESTION.options
      .map((o) => `<li><b>${o.label}</b> — ${o.description}</li>`)
      .join('');

    const canvas = this.el['dm-canvas'] as HTMLCanvasElement;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    this.ctx = canvas.getContext('2d')!;
    this.ctx.scale(dpr, dpr);

    this.sceneCanvas = this.el['dm-scene-canvas'] as HTMLCanvasElement;
    renderFullSceneFromWorld(this.sceneCanvas, this.world);
    this.mountPrototypeGrid();
    void this.ensureImageQuestion();

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
    this.el['dm-mode-image']?.addEventListener('click', () => this.setInputMode('image'));
    this.el['dm-mode-text']?.addEventListener('click', () => this.setInputMode('text'));
    window.addEventListener('keydown', this.onKeyDown);

    // Model controls and state are shared with the Decision Maker page.
    this.modelPanelHandle = mountDecisionModelPanel(this.el['dm-model-panel'], 'vision');
    this.disposers.push(this.modelPanelHandle);
    this.disposers.push(
      decisionRuntime.subscribe((event) => {
        if (event.type === 'loading') {
          // A new model is loading: stop and forget answers from the old one.
          this.setRunning(false);
          this.epoch++;
          this.inFlight = false;
          this.inferenceTimer.resetRollingWindow();
        } else if (event.type === 'ready') {
          void this.refreshPlan();
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

    this.syncInputModeUI();
    this.render();
    this.updatePanel();
    this.inferenceTimer.mount();

    if (decisionRuntime.ready) {
      if (this.inputMode === 'image' && !this.isLoadedModelVisionCapable()) {
        decisionRuntime.select('embeddinggemma2_text_vision_440m');
        this.modelPanelHandle?.setModality('vision');
        void decisionRuntime.load();
      } else {
        void this.refreshPlan();
      }
    } else if (decisionRuntime.loading) {
      this.setStatus(`Loading ${decisionRuntime.loadingLabel}...`);
    } else {
      this.setStatus('Load a vision model to begin');
    }
  }

  /** Renders miniature full-scene prototype cards in the Model Input section. */
  private mountPrototypeGrid() {
    const grid = this.el['dm-proto-grid'];
    if (!grid) return;
    grid.innerHTML = '';
    this.protoCards.clear();

    for (const spec of SCENE_PROTOTYPE_SPECS) {
      const card = document.createElement('div');
      card.className = 'dm-proto-card';
      card.dataset.protoId = spec.id;

      const thumb = document.createElement('canvas');
      thumb.width = SCENE_WIDTH;
      thumb.height = SCENE_HEIGHT;
      thumb.className = 'dm-proto-thumb';
      const obstacles = spec.obstacles.map((o) => ({ kind: findObstacleKind(o.kindName), x: o.x }));
      renderFullSceneCanvas(thumb, obstacles, SCENE_ZONE_W);

      const meta = document.createElement('div');
      meta.className = 'dm-proto-meta';

      const badge = document.createElement('span');
      badge.className = `dm-proto-badge ${spec.action.toLowerCase()}`;
      badge.textContent = spec.action;

      const title = document.createElement('span');
      title.className = 'dm-proto-title';
      title.title = spec.shortTitle;
      title.textContent = spec.shortTitle;

      meta.append(badge, title);
      card.append(thumb, meta);
      grid.appendChild(card);
      this.protoCards.set(spec.id, card);
    }
  }

  /** Builds and caches the multimodal ChoiceQuestion with 6 encoded full-scene prototypes. */
  private ensureImageQuestion() {
    if (!this.imageQuestionPromise) {
      this.imageQuestionPromise = Promise.resolve().then(() => {
        const options: Array<{ label: string; description: string; imageBytes: Uint8Array }> = [];
        for (const spec of SCENE_PROTOTYPE_SPECS) {
          const obstacles = spec.obstacles.map((o) => ({ kind: findObstacleKind(o.kindName), x: o.x }));
          const imageBytes = renderFullSceneTgaBytes(obstacles, SCENE_ZONE_W);
          options.push({
            label: spec.id,
            description: '',
            imageBytes,
          });
        }
        return {
          instructions: DINO_TEXT_QUESTION.instructions,
          temperature: 0.01,
          options,
        };
      });
    }
    return this.imageQuestionPromise;
  }

  private isLoadedModelVisionCapable(): boolean {
    if (!decisionRuntime.ready) return false;
    if (decisionRuntime.customModel) return true;
    return Boolean(decisionRuntime.loadedValue && DECISION_MODELS[decisionRuntime.loadedValue]?.supportsVision);
  }

  private setInputMode(mode: InputMode) {
    if (this.inputMode === mode) return;
    this.setRunning(false);
    this.epoch++;
    this.inFlight = false;
    this.inputMode = mode;
    this.modelPanelHandle?.setModality(mode === 'image' ? 'vision' : 'text');
    this.syncInputModeUI();
    this.render();
    this.updatePanel();

    if (mode === 'image' && decisionRuntime.ready && !this.isLoadedModelVisionCapable()) {
      decisionRuntime.select('embeddinggemma2_text_vision_440m');
      this.modelPanelHandle?.setModality('vision');
      void decisionRuntime.load();
      return;
    }
    if (decisionRuntime.ready) {
      void this.refreshPlan();
    }
  }

  private syncInputModeUI() {
    const isImage = this.inputMode === 'image';
    this.el['dm-mode-image']?.classList.toggle('active', isImage);
    this.el['dm-mode-text']?.classList.toggle('active', !isImage);
    if (this.el['dm-help']) {
      this.el['dm-help'].innerHTML = isImage
        ? 'The game captures the <b>full 768×192 game scene</b> (dino on the left, shaded near action zone, and all near &amp; far obstacles on the track—with no text labels) and compares those pixels against encoded reference full-scene prototypes in <code>DecisionMaker</code> to choose: <b>jump</b>, <b>duck</b>, or <b>wait</b>.'
        : 'The model never sees the game. When something new comes into view, it gets a one-line text description and makes a Choice: <b>jump</b>, <b>duck</b>, or <b>wait</b>.';
    }
    if (this.el['dm-input-label']) {
      this.el['dm-input-label'].textContent = isImage
        ? 'Model input (full 768×192 game scene)'
        : 'Model input (scene text description)';
    }
    if (this.el['dm-input-badge']) {
      this.el['dm-input-badge'].textContent = isImage ? 'Vision · { imageBytes }' : 'Text · string';
    }
    if (this.el['dm-scene-preview-wrap']) {
      this.el['dm-scene-preview-wrap'].style.display = isImage ? 'flex' : 'none';
    }
    if (this.el['dm-proto-section']) {
      this.el['dm-proto-section'].style.display = isImage ? 'block' : 'none';
    }
    if (this.el['dm-options']) {
      this.el['dm-options'].style.display = isImage ? 'none' : 'block';
    }
    if (this.el['dm-prompt']) {
      this.el['dm-prompt'].style.display = isImage ? 'none' : 'block';
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

  /**
   * True if the scene state changed since the last decision (or a decision is
   * still pending). In Scene Image mode, we evaluate:
   * - 'clear' when no obstacles are on screen,
   * - 'far:<id>' when an obstacle is on the far mid-track (~400–408px) while
   *   the near action zone is clear and the dino is on the ground, and
   * - 'near:<id>' as soon as an obstacle enters the near lookahead window
   *   (~198px, ~15–23 ticks before the jump/duck point) so inference completes
   *   in the background while the dino continues running smoothly.
   */
  private needsDecision(obs: Observation): boolean {
    if (this.plan.thinking && !this.inFlight) return true;
    if (this.plan.sceneKey === obs.sceneKey) return false;

    if (this.inputMode === 'image' && obs.sceneKey.startsWith('far:')) {
      if (!obs.onGround || !obs.farTarget) return false;
      const normX = normalizedSceneX(obs.farTarget.x, obs.speed);
      return normX >= SCENE_PROTO_FAR_X - 4 && normX <= SCENE_PROTO_FAR_X + 4;
    }
    return true;
  }

  /** Asks the model about the scene: one Choice call (either full-scene image or text state). */
  private async modelDecide(obs: Observation, snapshotTga?: Uint8Array): Promise<Decision> {
    const target = obs.target?.id ?? null;
    const sceneKey = obs.sceneKey;
    if (!decisionRuntime.ready) return { action: 'WAIT', reason: 'Model not ready.', target, sceneKey };

    if (this.inputMode === 'image') {
      if (!this.isLoadedModelVisionCapable()) {
        return {
          action: 'WAIT',
          reason: 'Select and load a vision-capable model (440M or 740M).',
          target,
          sceneKey,
        };
      }
      renderFullSceneFromWorld(this.sceneCanvas, this.world);
      const imageBytes = snapshotTga ?? renderFullSceneTgaFromWorld(this.world);
      const imageQuestion = await this.ensureImageQuestion();
      if (this.el['dm-scene-caption']) {
        this.el['dm-scene-caption'].textContent =
          `768×192 native-patch scene · ${(imageBytes.byteLength / 1024).toFixed(0)} KB`;
      }
      this.modelCalls++;
      const msg = await decisionRuntime.ask({ imageBytes }, 'choice', imageQuestion);
      if (msg.type !== 'DECIDE_RESULT') {
        return { action: 'WAIT', reason: `Model error: ${msg.error}`, target, sceneKey };
      }
      const { selectedKey, probabilities } = msg.result as {
        selectedKey: string;
        probabilities: Record<string, number>;
      };
      this.inferenceTimer.record(msg.inferenceTime, 'GPU');

      // Aggregate prototype probabilities by action class using max-pooling so
      // classes with more reference prototypes are not artificially boosted.
      const maxByAction: Record<Action, number> = { JUMP: 0, DUCK: 0, WAIT: 0 };
      let bestSpec = SCENE_PROTOTYPE_SPECS[0];
      let bestRawProb = -1;
      for (const spec of SCENE_PROTOTYPE_SPECS) {
        const p = probabilities?.[spec.id] ?? 0;
        if (p > maxByAction[spec.action]) {
          maxByAction[spec.action] = p;
        }
        if (p > bestRawProb) {
          bestRawProb = p;
          bestSpec = spec;
        }
      }
      const matchedSpec = SCENE_PROTOTYPE_SPECS.find((s) => s.id === selectedKey) ?? bestSpec;
      const action: Action = matchedSpec.action;
      const sumMax = maxByAction.JUMP + maxByAction.DUCK + maxByAction.WAIT || 1;
      const probs: Record<Action, number> = {
        JUMP: maxByAction.JUMP / sumMax,
        DUCK: maxByAction.DUCK / sumMax,
        WAIT: maxByAction.WAIT / sumMax,
      };

      let sceneDesc = 'clear track';
      if (obs.target && obs.farTarget) {
        sceneDesc = `near ${obs.target.kind.name} in action zone + far ${obs.farTarget.kind.name} ahead`;
      } else if (obs.target) {
        sceneDesc = `near ${obs.target.kind.name} in action zone`;
      } else if (obs.farTarget) {
        sceneDesc = `far ${obs.farTarget.kind.name} outside action zone`;
      }

      return {
        action,
        reason: `Scene: ${sceneDesc} → matched “${matchedSpec.shortTitle}” → ${action.toLowerCase()} (${Math.round(probs[action] * 100)}%).`,
        target,
        sceneKey,
        matchedProtoId: matchedSpec.id,
        probabilities: probs,
        inferenceTime: msg.inferenceTime,
      };
    }

    const prompt = describe(obs);
    this.modelCalls++;
    const msg = await decisionRuntime.ask(prompt, 'choice', DINO_TEXT_QUESTION);
    if (msg.type !== 'DECIDE_RESULT') {
      return { action: 'WAIT', reason: `Model error: ${msg.error}`, target, sceneKey, prompt };
    }

    const { selectedKey, probabilities } = msg.result as {
      selectedKey: string;
      probabilities: Record<string, number>;
    };
    this.inferenceTimer.record(msg.inferenceTime, 'GPU');
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
      sceneKey,
      prompt,
      probabilities: probs,
      inferenceTime: msg.inferenceTime,
    };
  }

  /**
   * Asks the model asynchronously in the worker while the game loop and dino
   * physics continue running uninterrupted.
   */
  private askModelInBackground(req: { obs: Observation; snapshotTga?: Uint8Array }) {
    this.inFlight = true;
    const epoch = this.epoch;
    this.modelDecide(req.obs, req.snapshotTga).then((decision) => {
      this.inFlight = false;
      if (epoch !== this.epoch) return;
      // If a newer scene state arrived while this inference was in flight,
      // immediately kick off a decision using the snapshot captured when that
      // scene state was entered.
      if (decision.sceneKey !== this.plan.sceneKey) {
        if (this.running && this.pendingDecisionRequest?.obs.sceneKey === this.plan.sceneKey) {
          const nextReq = this.pendingDecisionRequest;
          this.pendingDecisionRequest = undefined;
          this.askModelInBackground(nextReq);
        }
        return;
      }
      this.pendingDecisionRequest = undefined;
      this.plan = decision;
      if (this.running && this.pauseOnDecision && decision.action !== 'WAIT') this.setRunning(false);
      this.render();
      this.updatePanel();
    });
  }

  /** Blocking decision for the current frame (used on model load / reset). */
  private async refreshPlan() {
    if (this.busy) return;
    this.busy = true;
    this.updateButtons();
    if (this.inputMode === 'image' && this.modelCalls === 0) {
      this.plan = {
        action: 'WAIT',
        reason: 'Prewarming 6 reference full-scene prototypes on GPU…',
        target: null,
        sceneKey: 'init',
        thinking: true,
      };
      this.updatePanel();
    }
    try {
      this.plan = await this.modelDecide(observe(this.world, this.inputMode));
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
    return !this.world.gameOver && decisionRuntime.ready && !this.busy;
  }

  private setRunning(running: boolean) {
    if (running && !this.canStep()) return;
    const wasRunning = this.running;
    this.running = running;
    if (running && !wasRunning) this.runLoop();
    if (!running && wasRunning && this.plan.thinking && !this.inFlight) void this.refreshPlan();
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

  /** One tick while playing. Physics and dino movement run continuously while inference executes in the worker. */
  private tick() {
    if (this.busy || !this.canStep()) return;

    stepWorld(this.world, controls(observe(this.world, this.inputMode), this.plan), this.rand);

    const obs = observe(this.world, this.inputMode);
    if (this.needsDecision(obs)) {
      const target = obs.target?.id ?? null;
      if (obs.sceneKey !== this.plan.sceneKey) {
        const snapshotTga = this.inputMode === 'image' ? renderFullSceneTgaFromWorld(this.world) : undefined;
        this.pendingDecisionRequest = { obs, snapshotTga };
        this.plan = {
          action: 'WAIT',
          reason:
            this.inputMode === 'image' ? 'Model is inspecting full scene image…' : 'Model is reading what’s ahead…',
          target,
          sceneKey: obs.sceneKey,
          thinking: true,
        };
      }
      if (!this.inFlight) {
        const req = this.pendingDecisionRequest ?? {
          obs,
          snapshotTga: this.inputMode === 'image' ? renderFullSceneTgaFromWorld(this.world) : undefined,
        };
        this.pendingDecisionRequest = undefined;
        this.askModelInBackground(req);
      }
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
      stepWorld(this.world, controls(observe(this.world, this.inputMode), this.plan), this.rand);
      const obs = observe(this.world, this.inputMode);
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
    this.pendingDecisionRequest = undefined;
    this.world = createWorld();
    this.rand = mulberry32(Date.now());
    this.modelCalls = 0;
    renderFullSceneFromWorld(this.sceneCanvas, this.world);
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

    for (const [id, card] of this.protoCards.entries()) {
      card.classList.toggle('matched', !d.thinking && d.matchedProtoId === id);
    }

    if (decisionRuntime.ready) {
      this.el['status-message'].textContent = over
        ? 'Game over'
        : this.busy
          ? 'Prewarming scene prototypes…'
          : this.running
            ? 'Running'
            : 'Paused';
    }
    this.updateButtons();
  }

  private render() {
    const ctx = this.ctx;
    const w = this.world;
    const obs = observe(w, this.inputMode);

    if (this.inputMode === 'image' && this.sceneCanvas) {
      renderFullSceneFromWorld(this.sceneCanvas, w);
    }

    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // Near action zone in front of the dino.
    const look = lookDistance(w.speed, this.inputMode);
    const zoneW =
      this.inputMode === 'image' ? Math.round((SCENE_ZONE_W / (SCENE_NEAR_ENTRY_X - (DINO_X + DINO_W))) * look) : look;
    ctx.fillStyle = 'rgba(0, 127, 139, 0.10)';
    ctx.fillRect(DINO_X + DINO_W, 20, zoneW, GROUND_Y - 20);
    ctx.strokeStyle = 'rgba(0, 127, 139, 0.65)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(DINO_X + DINO_W + zoneW, 20);
    ctx.lineTo(DINO_X + DINO_W + zoneW, GROUND_Y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#007f8b';
    ctx.font = '11px Roboto, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(
      this.inputMode === 'image' ? 'near action zone (jump / duck)' : 'model looks here',
      DINO_X + DINO_W + zoneW - 6,
      32
    );

    if (this.inputMode === 'image') {
      ctx.fillStyle = '#5f6368';
      ctx.textAlign = 'left';
      ctx.fillText('far track (wait until obstacle enters action zone) →', DINO_X + DINO_W + zoneW + 10, 32);
    }
    ctx.textAlign = 'left';

    drawGround(ctx, w.scroll);

    // Obstacles: show the model's decision on the active near or far target.
    for (const o of w.obstacles) {
      drawObstacle(ctx, o, w.tick);
      const top = GROUND_Y - (o.kind.flying ? BIRD_BOTTOM : 0) - o.kind.height;
      const isNearTarget = obs.target?.id === o.id && this.plan.sceneKey === `near:${o.id}`;
      const isFarTarget = !obs.target && obs.farTarget?.id === o.id && this.plan.sceneKey === `far:${o.id}`;
      let label = '';
      if (this.inputMode === 'image') {
        if (isNearTarget && !this.plan.thinking) {
          label = `near → ${this.plan.action}`;
        } else if (isFarTarget && !this.plan.thinking) {
          label = `far → ${this.plan.action}`;
        }
      } else {
        label = isNearTarget && !this.plan.thinking ? `${o.kind.name} → ${this.plan.action}` : o.kind.name;
      }
      if (label) {
        ctx.font = `${isNearTarget || isFarTarget ? 'bold ' : ''}11px Roboto, sans-serif`;
        ctx.fillStyle = isNearTarget ? '#007f8b' : '#5f6368';
        ctx.textAlign = 'center';
        ctx.fillText(label, o.x + o.kind.width / 2, top - 6);
        ctx.textAlign = 'left';
      }
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
