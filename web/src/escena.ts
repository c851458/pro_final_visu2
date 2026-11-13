// Escena 3D del centro de distribucion con three.js. Solo representa el estado que devuelve la API:
// la velocidad de animacion no cambia las reglas ni el paso de la simulacion.
// Estilo de software de gestion de flotas: AMR industrial, demarcacion de piso, muelles de despacho,
// objetivo marcado en el piso, recorrido trazado y registro de eventos (sin efectos de juego).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import type { Fotograma, Mapa, Paquete } from './tipos';

const [UP, DOWN, LEFT, RIGHT] = [0, 1, 2, 3];
const MOVES: Record<number, [number, number]> = { [UP]: [-1, 0], [DOWN]: [1, 0], [LEFT]: [0, -1], [RIGHT]: [0, 1] };
const [HEAVY, FRAGILE] = [1, 2];
export const TYPE_LABELS = ['Normal', 'Pesado', 'Frágil'];
const PKG_SIZES = [0.34, 0.4, 0.32];
const ZONE_COLORS = [0xd9534f, 0x3b82f6, 0x8b5cf6];
export const ZONE_CSS = ['#d9534f', '#3b82f6', '#8b5cf6'];
const ROUTES = ['RUTA NORTE', 'RUTA CENTRO', 'RUTA SUR'];
const STATUS_COLORS = { normal: 0x2ec4f0, carry: 0x34d399, bad: 0xef4444, low: 0xf59e0b };
const WALL_H = 0.9;
const RACK_H = 1.5;
const PLATE_Y = 0.3;
const TRAIL_LEN = 80;
const DURATIONS: Record<string, number> = {
  reset: 0, move: 1, bump: 1.2, break: 1.6, pick: 2, drop: 2, pick_fail: 1.2, drop_fail: 1.2,
  wrong_zone: 1.3, charge: 2.4, charge_fail: 1.2, rescue: 3,
};
const ALERTS = ['bump', 'break', 'pick_fail', 'drop_fail', 'wrong_zone', 'charge_fail', 'rescue'];

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const headingOf = (dr: number, dc: number) => Math.atan2(dc, dr); // el frente del robot mira hacia +z

function lerpAngle(a: number, b: number, t: number) {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

export function batteryColor(frac: number) {
  return frac > 0.5 ? 0x34d399 : frac > 0.25 ? 0xf59e0b : 0xef4444;
}

// ---------------------------------------------------------------- texturas

let maxAnisotropy = 1;

function canvasTexture(w: number, h: number, draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  repeat?: [number, number]) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  draw(canvas.getContext('2d')!, w, h);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = maxAnisotropy;
  if (repeat) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(...repeat);
  }
  return tex;
}

function speckle(ctx: CanvasRenderingContext2D, w: number, h: number, n: number, alpha: number, size = 2) {
  for (let i = 0; i < n; i++) {
    const v = Math.random() < 0.5 ? 0 : 255;
    ctx.fillStyle = `rgba(${v},${v},${v},${Math.random() * alpha})`;
    ctx.fillRect(Math.random() * w, Math.random() * h, size, size);
  }
}

function stripes(ctx: CanvasRenderingContext2D, w: number, h: number, color: string, gap: number, width: number) {
  ctx.fillStyle = color;
  for (let x = -h; x < w + h; x += gap) {
    ctx.beginPath();
    ctx.moveTo(x, h); ctx.lineTo(x + width, h); ctx.lineTo(x + width + h, 0); ctx.lineTo(x + h, 0);
    ctx.fill();
  }
}

function barcode(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, seed: number) {
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  ctx.fillStyle = '#111';
  for (let px = x; px < x + w;) {
    const bar = 1 + Math.floor(rnd() * 3);
    if (rnd() > 0.45) ctx.fillRect(px, y, bar, h);
    px += bar + 1;
  }
}

// Caja de envio: carton con guia (ruta, codigo de barras) y marcas de manipulacion segun el tipo.
const pkgTexCache = new Map<string, THREE.Texture>();
function packageTexture(type: number, zone: number, zoneNames: string[], express = false) {
  const key = `${type}-${zone}-${express}`;
  const cached = pkgTexCache.get(key);
  if (cached) return cached;
  const tex = canvasTexture(256, 256, (ctx, w, h) => {
    if (type === HEAVY) {  // caja reforzada / guacal
      ctx.fillStyle = '#8a6a43';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = '#5c4529';
      ctx.lineWidth = 14;
      ctx.strokeRect(7, 7, w - 14, h - 14);
      for (const y of [80, 170]) { ctx.fillStyle = '#6f5434'; ctx.fillRect(0, y, w, 10); }
    } else {
      ctx.fillStyle = type === FRAGILE ? '#c9a36b' : '#b8874f';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 900, 0.08);
      ctx.fillStyle = 'rgba(214,190,150,0.8)';  // cinta
      ctx.fillRect(w * 0.44, 0, w * 0.12, h);
    }
    // guia de envio
    ctx.fillStyle = '#f7f7f2';
    ctx.fillRect(30, 30, 150, 104);
    ctx.fillStyle = '#111';
    ctx.font = 'bold 18px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`MUELLE ${zoneNames[zone]}`, 38, 36);
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(ROUTES[zone], 38, 58);
    barcode(ctx, 38, 76, 132, 40, 97 + zone * 13 + type * 7);
    ctx.fillStyle = ZONE_CSS[zone];
    ctx.fillRect(30, 124, 150, 10);
    if (express) {
      ctx.fillStyle = '#c62828';
      ctx.fillRect(30, 150, 120, 30);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 18px system-ui, sans-serif';
      ctx.fillText('EXPRESS', 42, 156);
    }
    if (type === FRAGILE) {
      ctx.fillStyle = '#c62828';
      ctx.font = 'bold 26px system-ui, sans-serif';
      ctx.fillText('FRÁGIL', 40, 200);
      ctx.strokeStyle = '#c62828';
      ctx.lineWidth = 4;
      ctx.beginPath();  // simbolo de copa
      ctx.moveTo(196, 186); ctx.quadraticCurveTo(198, 214, 212, 215); ctx.quadraticCurveTo(226, 214, 228, 186);
      ctx.moveTo(212, 215); ctx.lineTo(212, 234); ctx.moveTo(202, 236); ctx.lineTo(222, 236);
      ctx.stroke();
    } else if (type === HEAVY) {
      ctx.fillStyle = '#111';
      ctx.font = 'bold 22px system-ui, sans-serif';
      ctx.fillText('CARGA PESADA', 36, 200);
    }
  });
  pkgTexCache.set(key, tex);
  return tex;
}

const shadowed = <T extends THREE.Mesh>(m: T) => { m.castShadow = true; m.receiveShadow = true; return m; };

// ---------------------------------------------------------------- robot movil autonomo (AMR)

function buildRobot() {
  const body = new THREE.MeshStandardMaterial({ color: 0x2b3036, roughness: 0.45, metalness: 0.4 });
  const shell = new THREE.MeshStandardMaterial({ color: 0xd9dde2, roughness: 0.35, metalness: 0.2 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x111316, roughness: 0.9 });
  const metal = new THREE.MeshStandardMaterial({ color: 0xaab3bd, roughness: 0.25, metalness: 0.9 });
  const amber = new THREE.MeshStandardMaterial({ color: 0xf2b200, roughness: 0.5 });
  const statusMat = new THREE.MeshStandardMaterial({ color: 0x2ec4f0, emissive: 0x2ec4f0, emissiveIntensity: 1.2 });
  const add = (parent: THREE.Object3D, geo: THREE.BufferGeometry, mat: THREE.Material, x = 0, y = 0, z = 0) => {
    const m = shadowed(new THREE.Mesh(geo, mat));
    m.position.set(x, y, z);
    parent.add(m);
    return m;
  };

  const root = new THREE.Group();
  const chassis = new THREE.Group();
  root.add(chassis);
  add(chassis, new RoundedBoxGeometry(0.66, 0.06, 0.66, 2, 0.02), rubber, 0, 0.05, 0);  // paragolpes
  add(chassis, new RoundedBoxGeometry(0.62, 0.2, 0.62, 3, 0.05), body, 0, 0.16, 0);
  add(chassis, new RoundedBoxGeometry(0.6, 0.03, 0.6, 2, 0.01), shell, 0, 0.27, 0);  // cubierta
  add(chassis, new THREE.BoxGeometry(0.5, 0.025, 0.5), metal, 0, PLATE_Y - 0.0125, 0);  // plataforma de carga
  for (const z of [-0.315, 0.315]) add(chassis, new THREE.BoxGeometry(0.4, 0.035, 0.012), amber, 0, 0.12, z);
  // franja de luz de estado alrededor del chasis
  for (const [x, z, w, d] of [[0, 0.312, 0.56, 0.01], [0, -0.312, 0.56, 0.01], [0.312, 0, 0.01, 0.56], [-0.312, 0, 0.01, 0.56]]) {
    add(chassis, new THREE.BoxGeometry(w, 0.022, d), statusMat, x, 0.21, z);
  }
  // sensor LiDAR al frente y paro de emergencia atras
  const lidar = new THREE.Group();
  lidar.position.set(0, 0.17, 0.27);
  chassis.add(lidar);
  add(lidar, new THREE.CylinderGeometry(0.045, 0.05, 0.06, 20), rubber);
  add(lidar, new THREE.BoxGeometry(0.03, 0.012, 0.01), statusMat, 0, 0.01, 0.045);
  add(chassis, new THREE.CylinderGeometry(0.025, 0.025, 0.02, 16),
    new THREE.MeshStandardMaterial({ color: 0xd32f2f, roughness: 0.4 }), 0.22, 0.29, -0.22);
  const wheels: THREE.Mesh[] = [];
  for (const x of [-0.3, 0.3]) {
    const w = add(chassis, new THREE.CylinderGeometry(0.075, 0.075, 0.05, 20), rubber, x, 0.075, 0);
    w.rotation.z = Math.PI / 2;
    wheels.push(w);
  }
  const slot = new THREE.Object3D();  // centro de la plataforma
  slot.position.set(0, PLATE_Y, 0);
  chassis.add(slot);
  const light = new THREE.PointLight(0x2ec4f0, 0.8, 1.2, 2);
  light.position.set(0, 0.2, 0);
  chassis.add(light);
  return { root, chassis, lidar, wheels, statusMat, slot, light };
}

// ---------------------------------------------------------------- escena

interface Zona { center: THREE.Vector3; width: number; minC: number; row: number; stack: number }
interface PaqueteVisual { mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>; data: Paquete; size: number;
  mode: 'shelf' | 'flying' | 'carried' | 'dropping' | 'delivered'; rest: THREE.Vector3 }
interface Anim { f: Fotograma; t: number; dur: number; from: THREE.Vector3; to: THREE.Vector3; h0: number; h1: number;
  b0: number; b1: number; dir?: THREE.Vector3; pkg?: PaqueteVisual; pkgFrom?: THREE.Vector3; moved?: boolean }

export class Escena {
  stepMs = 200;
  onFotograma: (f: Fotograma) => void = () => {};
  onFinFotograma: (f: Fotograma) => void = () => {};
  /** Registro de operacion: texto, tipo (ok, alerta, info) y paso. */
  onEvento: (texto: string, tipo: 'ok' | 'alerta' | 'info', paso: number) => void = () => {};
  /** Formato de la recompensa (puntos en v1, pesos colombianos en v2). */
  formatReward: (x: number) => string = (x) => (x >= 0 ? '+' : '') + x.toFixed(1);
  /** Recompensas menores o iguales a este valor se registran como alerta. */
  alertThreshold = -1;

  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private clock = new THREE.Clock();
  private built = false;
  private W = 15;
  private H = 11;
  private mapa!: Mapa;
  private zones: Zona[] = [];
  private racks: { bottom: THREE.Group }[] = [];
  private obstacles = new THREE.Group();
  private robot = buildRobot();
  private pkgs: PaqueteVisual[] = [];
  private queue: Fotograma[] = [];
  private anim: Anim | null = null;
  private seed = 7;
  private batteryMax = 100;
  private useBattery = true;
  private vis = { pos: [8, 7] as number[], heading: Math.PI, carrying: false, current: 0, success: false, battery: 100,
    alertUntil: 0, wheelAngle: 0, chargingUntil: 0 };
  private camTween: { t: number; p0: THREE.Vector3; t0: THREE.Vector3; p1: THREE.Vector3; t1: THREE.Vector3 } | null = null;
  private followRobot = false;
  private targetRing: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  private trail: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private trailPoints: THREE.Vector3[] = [];
  private batteryBar = new THREE.Group();
  private barFill: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private chargerPos = new THREE.Vector3();
  private chargerLed!: THREE.MeshStandardMaterial;

  constructor(private container: HTMLElement, private zoneNames: string[]) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    maxAnisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.82;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x1c2026);
    this.scene.fog = new THREE.Fog(0x1c2026, 30, 70);
    this.scene.environment = new THREE.PMREMGenerator(this.renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    this.camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 200);
    this.camera.position.set(0, 14, 10.5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 35;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.target.set(0, 0, 0.6);
    this.controls.addEventListener('start', () => { this.camTween = null; });

    // iluminacion de nave industrial: luz difusa neutra y una principal suave
    this.scene.add(new THREE.HemisphereLight(0xeef2f7, 0x40444a, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 1.35);
    sun.position.set(-6, 15, 6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -11, right: 11, top: 9, bottom: -9, near: 1, far: 40 });
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);

    this.scene.add(this.robot.root);
    this.scene.add(this.obstacles);

    // objetivo marcado en el piso y recorrido del robot
    this.targetRing = new THREE.Mesh(new THREE.RingGeometry(0.36, 0.46, 40),
      new THREE.MeshBasicMaterial({ color: 0xf2b200, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
    this.targetRing.rotation.x = -Math.PI / 2;
    this.scene.add(this.targetRing);
    this.trail = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x2ec4f0, transparent: true, opacity: 0.55 }));
    this.scene.add(this.trail);

    const barBg = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.07),
      new THREE.MeshBasicMaterial({ color: 0x0b0e13, transparent: true, opacity: 0.75 }));
    this.barFill = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.035), new THREE.MeshBasicMaterial({ color: 0x34d399 }));
    this.barFill.position.z = 0.001;
    this.batteryBar.add(barBg, this.barFill);
    this.scene.add(this.batteryBar);

    new ResizeObserver(() => this.resize()).observe(container);
    this.loop();
  }

  get ocupado() { return this.anim !== null || this.queue.length > 0; }
  get pendientes() { return this.queue.length; }

  private cellPos(r: number, c: number, y = 0) {
    return new THREE.Vector3(c - (this.W - 1) / 2, y, r - (this.H - 1) / 2);
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Prepara la escena para un nivel: obstaculos, bateria y robot en el inicio. */
  setNivel(mapa: Mapa, useBattery: boolean, batteryMax: number) {
    if (!this.built) this.buildWarehouse(mapa);
    this.mapa = mapa;
    this.useBattery = useBattery;
    this.batteryMax = batteryMax;
    this.batteryBar.visible = useBattery;
    this.obstacles.clear();
    for (const [r, c] of mapa.bloqueos) this.obstacles.add(this.buildObstacle(r, c));
    this.limpiar();
    this.vis.pos = mapa.start.slice();
    this.vis.heading = Math.PI;
    this.robot.root.position.copy(this.cellPos(mapa.start[0], mapa.start[1]));
    this.robot.root.rotation.y = Math.PI;
    this.resetTrail();
    this.buildOrder([]);
  }

  encolar(frames: Fotograma[]) { this.queue.push(...frames); }

  limpiar() {
    this.queue.length = 0;
    if (this.anim) this.finishAnim(this.anim);
    this.anim = null;
  }

  async esperar() {
    while (this.ocupado) await new Promise((r) => setTimeout(r, 40));
  }

  // ---------------------------------------------------------------- construccion

  private rand() { return (this.seed = (this.seed * 16807) % 2147483647) / 2147483647; }

  /** Piso de concreto con juntas, demarcacion de seguridad, pasillos numerados, muelles y zona de carga. */
  private floorTexture(mapa: Mapa) {
    const { W, H } = this;
    const px = 96;
    const isShelf = (r: number, c: number) => mapa.layout[r]?.[c] === 'S';
    return canvasTexture((W - 2) * px, (H - 2) * px, (ctx, w, h) => {
      const X = (c: number) => (c - 1) * px;
      const Y = (r: number) => (r - 1) * px;
      ctx.fillStyle = '#868a90';
      ctx.fillRect(0, 0, w, h);
      for (let i = 0; i < 70; i++) {  // manchas suaves del concreto pulido
        const g = ctx.createRadialGradient(Math.random() * w, Math.random() * h, 0, Math.random() * w, Math.random() * h, 60 + Math.random() * 160);
        g.addColorStop(0, `rgba(${Math.random() < 0.5 ? '70,72,76' : '200,202,205'},0.08)`);
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
      }
      speckle(ctx, w, h, 9000, 0.07);
      ctx.strokeStyle = 'rgba(40,42,46,0.35)';  // juntas de dilatacion cada 3 m
      ctx.lineWidth = 2;
      for (let c = 1; c < W - 1; c += 3) { ctx.beginPath(); ctx.moveTo(X(c), 0); ctx.lineTo(X(c), h); ctx.stroke(); }
      for (let r = 1; r < H - 1; r += 3) { ctx.beginPath(); ctx.moveTo(0, Y(r)); ctx.lineTo(w, Y(r)); ctx.stroke(); }

      // linea amarilla alrededor de cada bloque de estanterias
      ctx.strokeStyle = '#e8b600';
      ctx.lineWidth = 7;
      const blocks: [number, number, number, number][] = [];
      for (let c = 1; c < W - 1; c++) {
        if (isShelf(2, c) && !isShelf(2, c - 1)) {
          let c2 = c;
          while (isShelf(2, c2 + 1)) c2++;
          let r2 = 2;
          while (isShelf(r2 + 1, c)) r2++;
          blocks.push([2, c, r2, c2]);
        }
      }
      blocks.forEach(([r1, c1, r2, c2]) => ctx.strokeRect(X(c1) - 8, Y(r1) - 8, X(c2 + 1) - X(c1) + 16, Y(r2 + 1) - Y(r1) + 16));

      // pasillos numerados
      ctx.fillStyle = 'rgba(245,245,240,0.85)';
      ctx.font = 'bold 22px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      let n = 1;
      for (let c = 1; c < W - 1; c++) {
        if (!isShelf(2, c) && (isShelf(2, c - 1) || isShelf(2, c + 1))) {
          ctx.save();
          ctx.translate(X(c) + px / 2, Y(7) + px / 2);
          ctx.fillText(`P-${String(n++).padStart(2, '0')}`, 0, 0);
          ctx.restore();
        }
      }

      // corredor peatonal (franjas) junto a la pared superior
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, w, 18); ctx.clip();
      stripes(ctx, w, 18, 'rgba(232,182,0,0.85)', 24, 10);
      ctx.restore();

      // muelles de despacho: area de preparacion delimitada
      mapa.zone_cells.forEach((cells, z) => {
        const cols = cells.map(([, c]) => c);
        const row = cells[0][0];
        const x1 = X(Math.min(...cols));
        const x2 = X(Math.max(...cols) + 1);
        ctx.fillStyle = `${ZONE_CSS[z]}55`;
        ctx.fillRect(x1, Y(row), x2 - x1, px);
        ctx.save();
        ctx.beginPath(); ctx.rect(x1, Y(row), x2 - x1, px); ctx.clip();
        stripes(ctx, x2 - x1 + x1, px + Y(row), 'rgba(255,255,255,0.10)', 30, 8);
        ctx.restore();
        ctx.strokeStyle = '#f2f2f2';
        ctx.lineWidth = 5;
        ctx.setLineDash([18, 10]);
        ctx.strokeRect(x1 + 6, Y(row) + 6, x2 - x1 - 12, px - 12);
        ctx.setLineDash([]);
        ctx.fillStyle = ZONE_CSS[z];
        ctx.fillRect(x1 + 6, Y(row) + 6, 10, px - 12);
        ctx.fillStyle = '#f2f2f2';
        ctx.font = 'bold 22px system-ui, sans-serif';
        ctx.fillText(`MUELLE ${this.zoneNames[z]}`, (x1 + x2) / 2, Y(row) + px * 0.38);
        ctx.font = '15px system-ui, sans-serif';
        ctx.fillText(ROUTES[z], (x1 + x2) / 2, Y(row) + px * 0.66);
      });

      // estacion de carga
      const [cr, cc] = mapa.charger;
      ctx.strokeStyle = '#e8b600';
      ctx.lineWidth = 6;
      ctx.strokeRect(X(cc) + 8, Y(cr) + 8, px - 16, px - 16);
      ctx.fillStyle = '#e8b600';
      ctx.font = 'bold 16px system-ui, sans-serif';
      ctx.fillText('CARGA', X(cc) + px / 2, Y(cr) + px / 2 + 10);

      // punto de inicio del robot
      const [sr, sc] = mapa.start;
      ctx.strokeStyle = 'rgba(46,196,240,0.6)';
      ctx.lineWidth = 4;
      ctx.strokeRect(X(sc) + 14, Y(sr) + 14, px - 28, px - 28);
    });
  }

  private buildWarehouse(mapa: Mapa) {
    this.built = true;
    const layout = mapa.layout;
    this.H = layout.length;
    this.W = layout[0].length;
    const { W, H } = this;

    const wallTex = canvasTexture(256, 256, (ctx, w, h) => {  // paneles prefabricados de concreto
      ctx.fillStyle = '#a3a7ad';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 2500, 0.08);
      ctx.strokeStyle = 'rgba(60,62,66,0.45)';
      ctx.lineWidth = 3;
      ctx.strokeRect(0, 0, w, h);
    });
    const guardTex = canvasTexture(128, 32, (ctx, w, h) => {
      ctx.fillStyle = '#e8b600';
      ctx.fillRect(0, 0, w, h);
      stripes(ctx, w, h, '#1b1b1b', 32, 12);
    });
    const cardboardTex = canvasTexture(128, 128, (ctx, w, h) => {
      ctx.fillStyle = '#b08250';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 600, 0.1);
      ctx.fillStyle = 'rgba(214,190,150,0.7)';
      ctx.fillRect(w * 0.42, 0, w * 0.16, h);
      ctx.fillStyle = '#f2f2ee';
      ctx.fillRect(14, 16, 44, 30);
    });

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), new THREE.MeshStandardMaterial({ color: 0x2a2e34, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    this.scene.add(ground);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(W - 2, H - 2),
      new THREE.MeshStandardMaterial({ map: this.floorTexture(mapa), roughness: 0.6, metalness: 0.05 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // muros perimetrales con defensa amarilla en la base
    const wallCells: [number, number][] = [];
    layout.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === '#') wallCells.push([r, c]); }));
    const walls = new THREE.InstancedMesh(new THREE.BoxGeometry(1, WALL_H, 1),
      new THREE.MeshStandardMaterial({ map: wallTex, roughness: 0.9 }), wallCells.length);
    const guard = new THREE.InstancedMesh(new THREE.BoxGeometry(1.01, 0.12, 1.01),
      new THREE.MeshStandardMaterial({ map: guardTex, roughness: 0.6 }), wallCells.length);
    const m4 = new THREE.Matrix4();
    wallCells.forEach(([r, c], i) => {
      const p = this.cellPos(r, c);
      walls.setMatrixAt(i, m4.makeTranslation(p.x, WALL_H / 2, p.z));
      guard.setMatrixAt(i, m4.makeTranslation(p.x, 0.1, p.z));
    });
    for (const m of [walls, guard]) this.scene.add(shadowed(m));

    // muelles de despacho: puerta enrollable en el muro detras de cada zona
    this.zones = mapa.zone_cells.map((cells, z) => {
      const cols = cells.map(([, c]) => c);
      const minC = Math.min(...cols);
      const width = Math.max(...cols) - minC + 1;
      const row = cells[0][0];
      const center = this.cellPos(row, minC + (width - 1) / 2);
      const doorTex = canvasTexture(256 * width, 200, (ctx, w, h) => {
        ctx.fillStyle = '#c4c8cd';
        ctx.fillRect(0, 0, w, h);
        for (let y = 0; y < h; y += 12) { ctx.fillStyle = 'rgba(70,74,80,0.35)'; ctx.fillRect(0, y, w, 3); }
        ctx.fillStyle = ZONE_CSS[z];
        ctx.fillRect(0, 0, w, 34);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 24px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`MUELLE ${this.zoneNames[z]} · ${ROUTES[z]}`, w / 2, 18);
      });
      const door = new THREE.Mesh(new THREE.PlaneGeometry(width - 0.2, WALL_H - 0.1),
        new THREE.MeshStandardMaterial({ map: doorTex, roughness: 0.5, metalness: 0.3 }));
      door.position.set(center.x, (WALL_H - 0.1) / 2 + 0.02, center.z + 0.5 - 0.002);
      door.rotation.y = Math.PI;
      this.scene.add(door);
      return { center, width, minC, row, stack: 0 };
    });

    // estacion de carga montada en el muro
    this.chargerPos = this.cellPos(mapa.charger[0], mapa.charger[1]);
    const unit = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.5, 0.55, 0.16, 3, 0.03),
      new THREE.MeshStandardMaterial({ color: 0x3a4048, roughness: 0.4, metalness: 0.5 })));
    unit.position.set(this.chargerPos.x, 0.42, this.chargerPos.z - 0.42);
    this.scene.add(unit);
    const contacts = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.05, 0.12),
      new THREE.MeshStandardMaterial({ color: 0xb08d3a, roughness: 0.3, metalness: 0.9 })));
    contacts.position.set(this.chargerPos.x, 0.12, this.chargerPos.z - 0.36);
    this.scene.add(contacts);
    this.chargerLed = new THREE.MeshStandardMaterial({ color: 0x34d399, emissive: 0x34d399, emissiveIntensity: 0.6 });
    const led = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.03, 0.01), this.chargerLed);
    led.position.set(this.chargerPos.x, 0.6, this.chargerPos.z - 0.335);
    this.scene.add(led);

    // estanterias selectivas (racks) con mercancia en carton
    const postGeo = new THREE.BoxGeometry(0.06, RACK_H, 0.06);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x1f4e8c, roughness: 0.5, metalness: 0.4 });
    const beamGeo = new THREE.BoxGeometry(0.94, 0.07, 0.05);
    const beamMat = new THREE.MeshStandardMaterial({ color: 0xe0661b, roughness: 0.45, metalness: 0.4 });
    const deckGeo = new THREE.BoxGeometry(0.92, 0.02, 0.9);
    const deckMat = new THREE.MeshStandardMaterial({ color: 0x8a9096, roughness: 0.6, metalness: 0.6 });
    const palletGeo = new THREE.BoxGeometry(0.8, 0.1, 0.8);
    const palletMat = new THREE.MeshStandardMaterial({ color: 0x9a7448, roughness: 0.9 });
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxMat = new THREE.MeshStandardMaterial({ map: cardboardTex, roughness: 0.85 });
    const addBoxes = (group: THREE.Group, y: number, count: number) => {
      for (let i = 0; i < count; i++) {
        const s = 0.22 + this.rand() * 0.16;
        const b = shadowed(new THREE.Mesh(boxGeo, boxMat));
        b.scale.set(s * (0.9 + this.rand() * 0.4), s * 0.9, s);
        b.position.set((this.rand() - 0.5) * 0.45, y + (s * 0.9) / 2, (this.rand() - 0.5) * 0.45);
        group.add(b);
      }
    };
    this.racks = mapa.shelf_cells.map(([r, c]) => {
      const rack = new THREE.Group();
      rack.position.copy(this.cellPos(r, c));
      for (const [x, z] of [[-0.45, -0.45], [0.45, -0.45], [-0.45, 0.45], [0.45, 0.45]]) {
        const post = shadowed(new THREE.Mesh(postGeo, postMat));
        post.position.set(x, RACK_H / 2, z);
        rack.add(post);
      }
      for (const y of [0.55, 1.03, RACK_H - 0.04]) {
        for (const z of [-0.45, 0.45]) {
          const b = shadowed(new THREE.Mesh(beamGeo, beamMat));
          b.position.set(0, y, z);
          rack.add(b);
        }
        if (y < RACK_H - 0.1) {
          const deck = shadowed(new THREE.Mesh(deckGeo, deckMat));
          deck.position.y = y + 0.04;
          rack.add(deck);
          addBoxes(rack, y + 0.05, 1 + Math.floor(this.rand() * 3));
        }
      }
      const pallet = shadowed(new THREE.Mesh(palletGeo, palletMat));
      pallet.position.y = 0.05;
      rack.add(pallet);
      const bottom = new THREE.Group();
      addBoxes(bottom, 0.1, 1 + Math.floor(this.rand() * 2));
      rack.add(bottom);
      this.scene.add(rack);
      return { bottom };
    });
  }

  /** Obstaculo del pasillo bloqueado (nivel N3): conos, barrera y derrame. */
  private buildObstacle(r: number, c: number) {
    const g = new THREE.Group();
    g.position.copy(this.cellPos(r, c));
    const coneMat = new THREE.MeshStandardMaterial({ color: 0xff6a00, roughness: 0.5 });
    const bandMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
    for (const [x, z] of [[-0.32, -0.32], [0.32, -0.32], [-0.32, 0.32], [0.32, 0.32]]) {
      const cone = shadowed(new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.4, 16), coneMat));
      cone.position.set(x, 0.2, z);
      const band = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.078, 0.06, 16), bandMat);
      band.position.set(x, 0.24, z);
      g.add(cone, band);
    }
    const barTex = canvasTexture(256, 32, (ctx, w, h) => {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);
      stripes(ctx, w, h, '#d32f2f', 40, 20);
    });
    const bar = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.07, 0.04), new THREE.MeshStandardMaterial({ map: barTex })));
    bar.position.set(0, 0.34, 0);
    g.add(bar);
    const spill = new THREE.Mesh(new THREE.CircleGeometry(0.42, 24),
      new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.05, metalness: 0.4, transparent: true, opacity: 0.7 }));
    spill.rotation.x = -Math.PI / 2;
    spill.position.y = 0.012;
    g.add(spill);
    return g;
  }

  // ---------------------------------------------------------------- paquetes

  private shelfPkgPos(shelf: number, size: number) {
    const [r, c] = this.mapa.shelf_cells[shelf];
    return this.cellPos(r, c, 0.1 + size / 2);
  }

  private zoneSlot(z: number, size: number) {
    const zone = this.zones[z];
    const k = zone.stack++;
    return this.cellPos(zone.row, zone.minC + (k % zone.width), size / 2 + Math.floor(k / zone.width) * 0.42);
  }

  /** Casilla de pasillo desde la que se recoge un paquete (la vecina libre mas cercana al robot). */
  private pickFace(shelf: number) {
    const [r, c] = this.mapa.shelf_cells[shelf];
    const free = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]
      .filter(([rr, cc]) => !'#SX'.includes(this.mapa.layout[rr]?.[cc] ?? '#')
        && !this.mapa.bloqueos.some(([br, bc]) => br === rr && bc === cc));
    const [pr, pc] = this.vis.pos;
    free.sort((a, b) => Math.abs(a[0] - pr) + Math.abs(a[1] - pc) - Math.abs(b[0] - pr) - Math.abs(b[1] - pc));
    const [fr, fc] = free[0] ?? [r, c];
    return this.cellPos(fr, fc);
  }

  private plateWorld(size: number) {
    return this.robot.slot.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, size / 2, 0));
  }

  private buildOrder(order: Paquete[]) {
    for (const p of this.pkgs) {
      this.scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      p.mesh.material.dispose();
    }
    this.zones.forEach((z) => { z.stack = 0; });
    this.pkgs = order.map((data) => {
      const size = PKG_SIZES[data.type];
      const mesh = shadowed(new THREE.Mesh(new RoundedBoxGeometry(size, size * 0.85, size, 2, 0.015),
        new THREE.MeshStandardMaterial({ map: packageTexture(data.type, data.zone, this.zoneNames, data.service === 1),
          roughness: 0.8, emissive: 0xf2b200, emissiveIntensity: 0 })));
      this.scene.add(mesh);
      const p: PaqueteVisual = { mesh, data, size, mode: 'shelf', rest: new THREE.Vector3() };
      if (data.status === 'entregado') {
        p.mode = 'delivered';
        p.rest.copy(this.zoneSlot(data.zone, size));
        mesh.position.copy(p.rest);
      } else if (data.status === 'cargando') {
        p.mode = 'carried';
      }
      return p;
    });
    const used = new Set(order.map((d) => d.shelf));
    this.racks.forEach((rk, i) => { rk.bottom.visible = !used.has(i); });
  }

  // ---------------------------------------------------------------- recorrido

  private resetTrail() {
    this.trailPoints = [this.robot.root.position.clone().setY(0.02)];
    this.trail.geometry.setFromPoints(this.trailPoints);
  }

  private pushTrail(p: THREE.Vector3) {
    this.trailPoints.push(p.clone().setY(0.02));
    if (this.trailPoints.length > TRAIL_LEN) this.trailPoints.shift();
    this.trail.geometry.setFromPoints(this.trailPoints);
  }

  // ---------------------------------------------------------------- animacion de fotogramas

  private durationFor(f: Fotograma) {
    let k = DURATIONS[f.event] ?? 1;
    if (f.event === 'move' && this.vis.carrying && this.pkgs[this.vis.current]?.data.type === HEAVY) k = 1.4;
    const backlog = this.queue.length > 2 ? 0.5 : 1;
    return Math.max(1, this.stepMs * k * backlog);
  }

  private beginAnim(f: Fotograma): Anim {
    const vis = this.vis;
    const a: Anim = { f, t: 0, dur: this.durationFor(f), from: this.cellPos(vis.pos[0], vis.pos[1]),
      to: this.cellPos(f.pos[0], f.pos[1]), h0: vis.heading, h1: vis.heading, b0: vis.battery, b1: f.battery };
    if (f.action !== null && f.action in MOVES) {
      const [dr, dc] = MOVES[f.action];
      a.h1 = headingOf(dr, dc);
      a.dir = new THREE.Vector3(dc, 0, dr);
    }
    switch (f.event) {
      case 'reset':
        this.buildOrder(f.order);
        Object.assign(vis, { carrying: f.carrying, current: f.current, success: f.success, battery: f.battery });
        a.from = a.to;
        a.h0 = a.h1 = Math.PI;
        a.b0 = a.b1 = f.battery;
        this.robot.root.position.copy(a.to);
        this.resetTrail();
        break;
      case 'pick': {
        const p = this.pkgs[f.current];
        const [r, c] = this.mapa.shelf_cells[p.data.shelf];
        a.h1 = headingOf(r - f.pos[0], c - f.pos[1]);
        a.pkg = p;
        p.mode = 'flying';
        p.rest.copy(this.shelfPkgPos(p.data.shelf, p.size));
        break;
      }
      case 'drop': {
        const p = this.pkgs[f.current - 1];
        a.pkg = p;
        p.mode = 'dropping';
        a.pkgFrom = this.plateWorld(p.size);
        p.rest.copy(this.zoneSlot(p.data.zone, p.size));
        break;
      }
      case 'break':
        a.pkg = this.pkgs[f.current];
        break;
      case 'charge':
        vis.chargingUntil = performance.now() + a.dur;
        break;
      default:
    }
    if (ALERTS.includes(f.event)) vis.alertUntil = performance.now() + a.dur + 250;
    this.onFotograma(f);
    return a;
  }

  private applyAnim(a: Anim, t: number) {
    const { f } = a;
    const robot = this.robot;
    const p = a.from.clone().lerp(a.to, ease(t));
    if ((f.event === 'bump' || f.event === 'break') && a.dir) {
      p.copy(a.from).addScaledVector(a.dir, 0.12 * Math.sin(Math.PI * Math.min(1, t * 1.6)));
    }
    if (f.event === 'rescue') {
      // intervencion tecnica: el robot se detiene (luz roja) y un tecnico lo lleva a la estacion de carga
      p.copy(t < 0.5 ? a.from : a.to);
      robot.root.visible = t < 0.42 || t > 0.58;
      if (t >= 0.5 && !a.moved) { a.moved = true; this.resetTrail(); }
    }
    robot.root.position.copy(p);
    this.vis.heading = lerpAngle(a.h0, a.h1, clamp01(t / 0.35));
    robot.root.rotation.y = this.vis.heading;
    if (['pick_fail', 'drop_fail', 'wrong_zone', 'charge_fail'].includes(f.event)) {
      robot.chassis.rotation.y = Math.sin(t * Math.PI * 4) * 0.08 * (1 - t);
    }
    if (f.event === 'pick' && a.pkg) {
      const u = ease(clamp01((t - 0.3) / 0.7));
      a.pkg.mesh.position.copy(a.pkg.rest).lerp(this.plateWorld(a.pkg.size), u);
      a.pkg.mesh.position.y += Math.sin(Math.PI * u) * 0.12;
      a.pkg.mesh.rotation.y = lerpAngle(0, this.vis.heading, u);
    }
    if (f.event === 'drop' && a.pkg && a.pkgFrom) {
      const u = ease(clamp01((t - 0.15) / 0.85));
      a.pkg.mesh.position.copy(a.pkgFrom).lerp(a.pkg.rest, u);
      a.pkg.mesh.rotation.y = lerpAngle(this.vis.heading, 0, u);
    }
    if (f.event === 'charge') this.vis.battery = a.b0 + (a.b1 - a.b0) * ease(t);
    else if (f.event === 'rescue') this.vis.battery = t > 0.5 ? a.b1 : Math.max(0, a.b0 - 1);
    else this.vis.battery = a.b0 + (a.b1 - a.b0) * t;
  }

  private finishAnim(a: Anim) {
    const { f } = a;
    this.vis.pos = f.pos.slice();
    this.robot.chassis.rotation.y = 0;
    this.robot.root.visible = true;
    this.robot.root.position.copy(this.cellPos(f.pos[0], f.pos[1]));
    Object.assign(this.vis, { carrying: f.carrying, current: f.current, success: f.success, battery: f.battery });
    if (f.event === 'move') this.pushTrail(this.robot.root.position);
    const money = this.formatReward(f.reward);
    const log = (text: string, kind: 'ok' | 'alerta' | 'info') => this.onEvento(text, kind, f.steps);
    switch (f.event) {
      case 'pick':
        if (a.pkg) {
          a.pkg.mode = 'carried';
          log(`Recogido paquete ${f.current + 1} (${TYPE_LABELS[a.pkg.data.type].toLowerCase()}) · ${money}`, 'ok');
        }
        break;
      case 'drop':
        if (a.pkg) {
          a.pkg.mode = 'delivered';
          a.pkg.mesh.position.copy(a.pkg.rest);
          const late = f.order[f.current - 1]?.late;
          log(`Despachado en muelle ${this.zoneNames[a.pkg.data.zone]}${late ? ' FUERA DE PLAZO' : ''} · ${money}`, late ? 'alerta' : 'ok');
        }
        if (f.success) log('Orden completa despachada', 'ok');
        break;
      case 'break':
        a.pkg && (a.pkg.mode = 'shelf');
        log(`Daño: paquete frágil roto por colisión · ${money}`, 'alerta');
        break;
      case 'bump':
        log(`Colisión en (${f.pos[0]}, ${f.pos[1]}) · ${money}`, 'alerta');
        break;
      case 'charge':
        log(`Carga de batería completa · ${money}`, 'info');
        break;
      case 'rescue':
        log(`Intervención técnica: batería agotada · ${money}`, 'alerta');
        break;
      case 'wrong_zone':
        log(`Muelle equivocado: no se despacha · ${money}`, 'alerta');
        break;
      case 'pick_fail':
      case 'drop_fail':
      case 'charge_fail':
        log(`Acción no válida (${f.event === 'pick_fail' ? 'recoger' : f.event === 'drop_fail' ? 'despachar' : 'cargar'}) · ${money}`, 'alerta');
        break;
      default:
        if (f.reward <= this.alertThreshold) log(`Costo ${money}`, 'alerta');
    }
    // por si en un mismo paso ocurrieron dos cosas (p. ej. se rompio un fragil y se agoto la bateria)
    this.pkgs.forEach((p, i) => {
      p.data = f.order[i];
      if (p.data.status === 'pendiente') p.mode = 'shelf';
    });
    this.onFinFotograma(f);
  }

  private updateRobot(dt: number, now: number) {
    const { robot, vis } = this;
    const moving = this.anim && ['move', 'bump', 'break'].includes(this.anim.f.event);
    if (moving && this.anim) vis.wheelAngle += dt * (1000 / this.anim.dur) * 8;
    robot.wheels.forEach((w) => { w.rotation.x = vis.wheelAngle; });
    robot.lidar.rotation.y += dt * 6;

    const frac = vis.battery / this.batteryMax;
    const low = this.useBattery && frac <= 0.25;
    const mood = now < vis.alertUntil ? 'bad' : vis.carrying ? 'carry' : low ? 'low' : 'normal';
    const color = STATUS_COLORS[mood];
    const blink = mood === 'bad' && now % 400 < 200 ? 0.3 : 1.2;
    robot.statusMat.color.setHex(color);
    robot.statusMat.emissive.setHex(color);
    robot.statusMat.emissiveIntensity = blink;
    robot.light.color.setHex(color);

    this.batteryBar.position.copy(robot.root.position).add(new THREE.Vector3(0, 0.95, 0));
    this.batteryBar.quaternion.copy(this.camera.quaternion);
    this.barFill.scale.x = Math.max(0.001, frac);
    this.barFill.position.x = -0.23 * (1 - frac);
    this.barFill.material.color.setHex(batteryColor(frac));
  }

  private updateWorld(now: number) {
    const { vis } = this;
    this.pkgs.forEach((p, i) => {
      if (p.mode === 'shelf') {
        p.mesh.position.copy(this.shelfPkgPos(p.data.shelf, p.size));
        p.mesh.rotation.y = 0;
      } else if (p.mode === 'carried') {
        p.mesh.position.copy(this.plateWorld(p.size));
        p.mesh.rotation.y = this.robot.root.rotation.y;
      }
      const isTarget = i === vis.current && p.mode === 'shelf';
      p.mesh.material.emissiveIntensity = isTarget ? 0.08 + 0.06 * Math.sin(now / 300) : 0;
    });

    // objetivo marcado en el piso: estanteria del paquete actual o muelle de destino
    const cur = this.pkgs[vis.current];
    const ring = this.targetRing;
    ring.visible = Boolean(cur) && !vis.success;
    if (cur && ring.visible) {
      const target = vis.carrying ? this.zones[cur.data.zone].center : this.pickFace(cur.data.shelf);
      ring.position.set(target.x, 0.02, target.z);
      const scale = vis.carrying ? 1.5 : 1;
      ring.scale.setScalar(scale * (1 + 0.04 * Math.sin(now / 350)));
      ring.material.color.setHex(vis.carrying ? ZONE_COLORS[cur.data.zone] : 0xf2b200);
      ring.material.opacity = 0.55 + 0.25 * Math.sin(now / 350);
    }

    const charging = now < vis.chargingUntil;
    this.chargerLed.emissiveIntensity = charging ? 1.5 + 0.5 * Math.sin(now / 80) : 0.6;
    this.chargerLed.emissive.setHex(charging ? 0xf2b200 : 0x34d399);
  }

  // ---------------------------------------------------------------- camara

  setCamara(name: 'iso' | 'top' | 'follow' | 'side') {
    this.followRobot = name === 'follow';
    const presets = { iso: [[0, 14, 10.5], [0, 0, 0.6]], top: [[0, 19, 0.01], [0, 0, 0]], side: [[10, 9, 10], [0, 0, 0.5]] };
    const root = this.robot.root.position;
    const [pos, target] = name === 'follow'
      ? [root.clone().add(new THREE.Vector3(0, 4.5, 5)), root.clone()]
      : [new THREE.Vector3(...(presets[name][0] as [number, number, number])), new THREE.Vector3(...(presets[name][1] as [number, number, number]))];
    this.camTween = { t: 0, p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: target };
  }

  /** Accion de movimiento segun hacia donde mira la camara ("arriba" = hacia el fondo de la pantalla). */
  accionRelativa(dir: 'up' | 'down' | 'left' | 'right') {
    const fwd = new THREE.Vector3();
    this.camera.getWorldDirection(fwd);
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
    fwd.normalize();
    const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
    const d = { up: fwd, down: fwd.clone().negate(), right, left: right.clone().negate() }[dir];
    if (Math.abs(d.x) > Math.abs(d.z)) return d.x > 0 ? RIGHT : LEFT;
    return d.z > 0 ? DOWN : UP;
  }

  private updateCamera(dt: number) {
    if (this.camTween) {
      const tw = this.camTween;
      tw.t = Math.min(1, tw.t + dt * 1.6);
      const e = ease(tw.t);
      this.camera.position.lerpVectors(tw.p0, tw.p1, e);
      this.controls.target.lerpVectors(tw.t0, tw.t1, e);
      if (tw.t >= 1) this.camTween = null;
    } else if (this.followRobot) {
      const delta = this.robot.root.position.clone().sub(this.controls.target).multiplyScalar(Math.min(1, dt * 4));
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    this.controls.update();
  }

  private loop = () => {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const now = performance.now();
    if (this.anim) {
      this.anim.t = Math.min(1, this.anim.t + (dt * 1000) / this.anim.dur);
      this.applyAnim(this.anim, this.anim.t);
      if (this.anim.t >= 1) {
        const done = this.anim;
        this.anim = null;
        this.finishAnim(done);
      }
    }
    if (!this.anim && this.queue.length && this.built) {
      this.anim = this.beginAnim(this.queue.shift()!);
      this.applyAnim(this.anim, 0);
    }
    if (this.built) {
      this.updateRobot(dt, now);
      this.updateWorld(now);
    }
    this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.loop);
  };
}
