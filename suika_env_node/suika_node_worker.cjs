const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const readline = require("node:readline");

const WALL_PAD = 64;
const LOSE_HEIGHT = 84;
const STATUS_BAR_HEIGHT = 48;
const WIDTH = 640;
const HEIGHT = 960;
const MAX_TOP10 = 10;
const MAX_BOARD = 40;
const DT_MS = 1000 / 60;
const READY_DELAY_MS = 500;
const READY_TIMEOUT_MS = 2000;
const READY_POLL_MS = 20;
const STABLE_POLLS_REQUIRED = 5;

const FRICTION = {
  friction: 0.006,
  frictionStatic: 0.006,
  frictionAir: 0,
  restitution: 0.1,
};

const GAME_STATES = {
  MENU: 0,
  READY: 1,
  DROP: 2,
  LOSE: 3,
};

const FRUIT_SIZES = [
  { radius: 24, scoreValue: 1 },
  { radius: 32, scoreValue: 3 },
  { radius: 40, scoreValue: 6 },
  { radius: 56, scoreValue: 10 },
  { radius: 64, scoreValue: 15 },
  { radius: 72, scoreValue: 21 },
  { radius: 84, scoreValue: 28 },
  { radius: 96, scoreValue: 36 },
  { radius: 128, scoreValue: 45 },
  { radius: 160, scoreValue: 55 },
  { radius: 192, scoreValue: 66 },
];

function mulberry32(a) {
  return function rand() {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function loadMatter() {
  const matterPath = path.join(__dirname, "suika-game", "matter.js");
  const source = fs.readFileSync(matterPath, "utf8");
  const sandbox = {
    module: { exports: {} },
    exports: {},
    window: {},
    self: {},
    global: {},
    globalThis: {},
    console,
    setTimeout,
    clearTimeout,
    Math,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "matter.js" });
  return sandbox.module.exports || sandbox.exports || sandbox.window.Matter || sandbox.Matter;
}

class SuikaCore {
  constructor(seed = 42) {
    this.Matter = loadMatter();
    const { Engine, Bodies, Composite, Events } = this.Matter;
    this.Engine = Engine;
    this.Bodies = Bodies;
    this.Composite = Composite;
    this.Events = Events;
    this.seed = Number(seed) || 42;
    this.rand = mulberry32(this.seed);
    this.engine = null;
    this.gameStatics = null;
    this._stepCollisionPairs = 0;
    this._stepLoseHits = 0;
    this._stepFirstLoseEvent = null;
    this._loseHitsTotal = 0;
    this.reset(this.seed);
  }

  _teardownEngine() {
    if (!this.engine) return;
    try {
      // Remove callbacks attached to the old engine to break references.
      this.Events.off(this.engine);
    } catch (_) {}
    try {
      // Remove all bodies/constraints/composites in the old world.
      this.Composite.clear(this.engine.world, false, true);
    } catch (_) {}
    try {
      // Clear broadphase and internal engine caches.
      this.Engine.clear(this.engine);
    } catch (_) {}
    this.previewBall = null;
    this.gameStatics = null;
    this.engine = null;
  }

  _sampleSpawnType() {
    return Math.floor(this.rand() * 5);
  }

  _calcScore() {
    const score = this.fruitsMerged.reduce((total, count, sizeIndex) => {
      return total + FRUIT_SIZES[sizeIndex].scoreValue * count;
    }, 0);
    this.score = score;
  }

  _generateFruitBody(x, y, sizeIndex, extraConfig = {}) {
    const size = FRUIT_SIZES[sizeIndex];
    const circle = this.Bodies.circle(x, y, size.radius, {
      ...FRICTION,
      ...extraConfig,
    });
    circle.sizeIndex = sizeIndex;
    return circle;
  }

  _setupCollisionHandler() {
    this.Events.on(this.engine, "collisionStart", (e) => {
      for (let i = 0; i < e.pairs.length; i += 1) {
        const { bodyA, bodyB } = e.pairs[i];
        if (!bodyA || !bodyB) continue;
        if (bodyA.isStatic || bodyB.isStatic) continue;
        if (!Number.isFinite(bodyA.sizeIndex) || !Number.isFinite(bodyB.sizeIndex)) continue;

        this._stepCollisionPairs += 1;
        const aY = bodyA.position.y + bodyA.circleRadius;
        const bY = bodyB.position.y + bodyB.circleRadius;
        if (aY < LOSE_HEIGHT || bY < LOSE_HEIGHT) {
          this._stepLoseHits += 1;
          this._loseHitsTotal += 1;
          if (this._stepFirstLoseEvent === null) {
            this._stepFirstLoseEvent = {
              aY: Number(aY),
              bY: Number(bY),
              aX: Number(bodyA.position.x),
              bX: Number(bodyB.position.x),
              aSize: Number(bodyA.sizeIndex),
              bSize: Number(bodyB.sizeIndex),
            };
          }
          this._loseGame();
          return;
        }

        if (bodyA.sizeIndex !== bodyB.sizeIndex) continue;

        let newSize = bodyA.sizeIndex + 1;
        if (bodyA.circleRadius >= FRUIT_SIZES[FRUIT_SIZES.length - 1].radius) {
          newSize = 0;
        }

        this.fruitsMerged[bodyA.sizeIndex] += 1;
        const midPosX = (bodyA.position.x + bodyB.position.x) / 2;
        const midPosY = (bodyA.position.y + bodyB.position.y) / 2;

        this.Composite.remove(this.engine.world, [bodyA, bodyB]);
        this.Composite.add(this.engine.world, this._generateFruitBody(midPosX, midPosY, newSize));
        this._calcScore();
      }
    });
  }

  _createStatics() {
    const wallProps = {
      isStatic: true,
      ...FRICTION,
    };
    return [
      this.Bodies.rectangle(-(WALL_PAD / 2), HEIGHT / 2, WALL_PAD, HEIGHT, wallProps),
      this.Bodies.rectangle(WIDTH + (WALL_PAD / 2), HEIGHT / 2, WALL_PAD, HEIGHT, wallProps),
      this.Bodies.rectangle(
        WIDTH / 2,
        HEIGHT + (WALL_PAD / 2) - STATUS_BAR_HEIGHT,
        WIDTH,
        WALL_PAD,
        wallProps
      ),
    ];
  }

  reset(seed = null) {
    if (seed !== null && seed !== undefined) {
      this.seed = Number(seed) || 42;
      this.rand = mulberry32(this.seed);
    }

    this._teardownEngine();
    this.engine = this.Engine.create();
    this.gameStatics = this._createStatics();
    this.Composite.add(this.engine.world, this.gameStatics);
    this._setupCollisionHandler();

    this.stateIndex = GAME_STATES.READY;
    this.score = 0;
    this.fruitsMerged = Array(FRUIT_SIZES.length).fill(0);
    this.currentFruitSize = 0;
    this.nextFruitSize = 0;
    this.currentFruitX = 0.5;
    this.previewX = WIDTH / 2;
    // Match browser JS startup behavior:
    // currentFruitSize=0, nextFruitSize=0 at start, then randomized only after drops.
    this.previewBall = this._generateFruitBody(this.previewX, 0, this.currentFruitSize, {
      isStatic: true,
      collisionFilter: { mask: 0x0040 },
    });
    this.Composite.add(this.engine.world, this.previewBall);

    return this._snapshot();
  }

  _setNextFruitSize() {
    this.nextFruitSize = this._sampleSpawnType();
  }

  _loseGame() {
    this.stateIndex = GAME_STATES.LOSE;
  }

  _advance(ms) {
    const n = Math.max(1, Math.ceil(ms / DT_MS));
    for (let i = 0; i < n; i += 1) {
      this.Engine.update(this.engine, DT_MS);
      if (this.stateIndex === GAME_STATES.LOSE) break;
    }
  }

  addFruit(pixelX) {
    if (this.stateIndex !== GAME_STATES.READY) return false;

    this.stateIndex = GAME_STATES.DROP;
    const x = clamp(Number(pixelX) || 0, 0, WIDTH);
    const latestFruit = this._generateFruitBody(x, 0, this.currentFruitSize);
    this.Composite.add(this.engine.world, latestFruit);

    this.currentFruitSize = this.nextFruitSize;
    this._setNextFruitSize();
    this._calcScore();

    if (this.previewBall) {
      this.Composite.remove(this.engine.world, this.previewBall);
    }
    this.previewBall = this._generateFruitBody(this.previewX, 0, this.currentFruitSize, {
      isStatic: true,
      collisionFilter: { mask: 0x0040 },
    });

    // Match browser env behavior:
    // 1) JS timer returns DROP->READY after 500ms.
    // 2) Python side waits a bit more until score is stable.
    let elapsed = 0;
    let previewShown = false;
    let stablePolls = 0;
    let lastScore = null;
    while (elapsed < READY_TIMEOUT_MS) {
      this._advance(READY_POLL_MS);
      elapsed += READY_POLL_MS;

      if (!previewShown && elapsed >= READY_DELAY_MS && this.stateIndex === GAME_STATES.DROP) {
        this.Composite.add(this.engine.world, this.previewBall);
        this.stateIndex = GAME_STATES.READY;
        previewShown = true;
      }

      if (this.stateIndex !== GAME_STATES.DROP) {
        if (lastScore !== null && this.score === lastScore) {
          stablePolls += 1;
        } else {
          stablePolls = 0;
        }
        lastScore = this.score;
        if (stablePolls >= STABLE_POLLS_REQUIRED) {
          break;
        }
      } else {
        stablePolls = 0;
        lastScore = this.score;
      }
    }

    if (!previewShown && this.stateIndex === GAME_STATES.DROP) {
      this.Composite.add(this.engine.world, this.previewBall);
      this.stateIndex = GAME_STATES.READY;
    }
    return true;
  }

  step(actionCentered) {
    this._stepCollisionPairs = 0;
    this._stepLoseHits = 0;
    this._stepFirstLoseEvent = null;

    const prevScore = this.score;
    const xCentered = clamp(Number(actionCentered) || 0, -0.5, 0.5);
    const xNorm = xCentered + 0.5;
    this.currentFruitX = xNorm;
    this.previewX = xNorm * WIDTH;

    const actionPixelX = Math.floor(xNorm * WIDTH);
    this.addFruit(actionPixelX);

    const terminated = this.stateIndex === GAME_STATES.LOSE;
    const snap = this._snapshot();
    const fruitCount = Number.isFinite(snap.fruit_count) ? Number(snap.fruit_count) : 0.0;
    const stepPenalty = 0.5 * fruitCount;
    const reward = terminated ? (-200.0 - stepPenalty) : ((this.score - prevScore) - stepPenalty);
    return {
      ...snap,
      reward,
      terminated,
      truncated: false,
      info: {
        score: this.score,
        collision_pairs_step: this._stepCollisionPairs,
        lose_height_hits_step: this._stepLoseHits,
        lose_height_hits_total: this._loseHitsTotal,
        lose_height_triggered: this._stepLoseHits > 0 ? 1 : 0,
        lose_height: LOSE_HEIGHT,
        lose_event: this._stepFirstLoseEvent,
      },
    };
  }

  _snapshot() {
    const bodies = this.engine?.world?.bodies || [];
    const fruits = bodies
      .filter((b) => !b.isStatic && Number.isFinite(b?.position?.x) && Number.isFinite(b?.position?.y) && Number.isFinite(b?.sizeIndex))
      .slice()
      .sort((a, b) => a.position.y - b.position.y);

    const topFruits = fruits.slice(0, MAX_TOP10);
    const boardFruits = fruits.slice(0, MAX_BOARD);

    const top10 = [];
    const top10Types = [];
    const top10Mask = [];
    for (const b of topFruits) {
      top10.push(clamp(b.position.x / WIDTH, 0, 1), clamp(b.position.y / HEIGHT, 0, 1));
      top10Types.push(clamp(Math.floor(b.sizeIndex), 0, 10));
      top10Mask.push(1.0);
    }
    while (top10.length < 20) top10.push(0.0);
    while (top10Types.length < 10) top10Types.push(0.0);
    while (top10Mask.length < 10) top10Mask.push(0.0);

    const boardXY = [];
    const boardRadius = [];
    const boardMass = [];
    const boardType = [];
    const boardMask = [];
    for (const b of boardFruits) {
      boardXY.push(clamp(b.position.x / WIDTH, 0, 1), clamp(b.position.y / HEIGHT, 0, 1));
      boardRadius.push(clamp((Number(b.circleRadius) || 0) / 100.0, 0, 1));
      boardMass.push(clamp((Number(b.mass) || 0) / 1000.0, 0, 1));
      boardType.push(clamp(Math.floor(b.sizeIndex), 0, 10));
      boardMask.push(1.0);
    }
    while (boardXY.length < 80) boardXY.push(0.0);
    while (boardRadius.length < 40) boardRadius.push(0.0);
    while (boardMass.length < 40) boardMass.push(0.0);
    while (boardType.length < 40) boardType.push(0.0);
    while (boardMask.length < 40) boardMask.push(0.0);

    const largestFruitType = fruits.reduce((m, b) => Math.max(m, Number.isFinite(b.sizeIndex) ? b.sizeIndex : 0), 0);
    const minY = fruits.length > 0 ? Math.min(...fruits.map((b) => b.position.y)) : HEIGHT;
    const maxHeight = clamp((HEIGHT - minY) / HEIGHT, 0, 1);
    const dangerCount = fruits.filter((b) => b.position.y <= LOSE_HEIGHT).length;

    return {
      status: this.stateIndex,
      score: this.score,
      current_fruit_type: this.currentFruitSize,
      next_fruit_type: this.nextFruitSize,
      current_fruit_x: this.currentFruitX,
      stage_top10_xy: top10,
      top10_fruit_types: top10Types,
      top10_mask: top10Mask,
      max_height: maxHeight,
      danger_count: dangerCount,
      largest_fruit_type: largestFruitType,
      fruit_count: fruits.length,
      board_fruit_xy: boardXY,
      board_fruit_radius: boardRadius,
      board_fruit_mass: boardMass,
      board_fruit_type: boardType,
      board_fruit_mask: boardMask,
    };
  }
}

let core = new SuikaCore(42);

function respond(obj) {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

function handle(req) {
  const cmd = req?.cmd;
  if (cmd === "reset") {
    const out = core.reset(req.seed ?? null);
    respond({ ok: true, ...out, info: {} });
    return;
  }
  if (cmd === "step") {
    const out = core.step(req.action ?? 0.0);
    respond({ ok: true, ...out });
    return;
  }
  if (cmd === "restart") {
    const out = core.reset(req.seed ?? null);
    respond({ ok: true, ...out, info: {} });
    return;
  }
  if (cmd === "close") {
    core._teardownEngine();
    respond({ ok: true });
    process.exit(0);
    return;
  }
  respond({ ok: false, error: `unknown cmd: ${cmd}` });
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
  terminal: false,
});

rl.on("line", (line) => {
  const txt = line.trim();
  if (!txt) return;
  try {
    handle(JSON.parse(txt));
  } catch (e) {
    respond({ ok: false, error: String(e) });
  }
});
