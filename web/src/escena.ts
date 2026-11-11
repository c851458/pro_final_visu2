// Escena 3D de la bodega con three.js. Solo representa el estado que devuelve la API:
// la velocidad de animación no cambia las reglas ni el paso de la simulación.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import type { Fotograma, Mapa, Paquete } from './tipos';

const [UP, DOWN, LEFT, RIGHT] = [0, 1, 2, 3];
const MOVES: Record<number, [number, number]> = { [UP]: [-1, 0], [DOWN]: [1, 0], [LEFT]: [0, -1], [RIGHT]: [0, 1] };
const [HEAVY, FRAGILE] = [1, 2];
export const TYPE_LABELS = ['Normal', 'Pesado', 'Frágil'];
const PKG_SIZES = [0.36, 0.4, 0.34];
const ZONE_COLORS = [0xe5484d, 0x3b82f6, 0xa855f7];
export const ZONE_CSS = ['#e5484d', '#3b82f6', '#a855f7'];
const EYE_COLORS = { normal: 0x00e5ff, carry: 0x3dff8b, bad: 0xff3b30, low: 0xffa000 };
const WALL_H = 0.7;
const RACK_H = 1.5;
const DURATIONS: Record<string, number> = {
  reset: 0, move: 1, bump: 1.3, break: 1.8, pick: 2.2, drop: 2.2, pick_fail: 1.3, drop_fail: 1.3,
  wrong_zone: 1.5, charge: 2.5, charge_fail: 1.3, rescue: 4,
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
  return frac > 0.5 ? 0x4fd18b : frac > 0.25 ? 0xffc531 : 0xff5a4e;
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

function speckle(ctx: CanvasRenderingContext2D, w: number, h: number, n: number, alpha: number) {
  for (let i = 0; i < n; i++) {
    const v = Math.random() < 0.5 ? 0 : 255;
    ctx.fillStyle = `rgba(${v},${v},${v},${Math.random() * alpha})`;
    ctx.fillRect(Math.random() * w, Math.random() * h, 2, 2);
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

function bolt(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number) {
  ctx.beginPath();
  ctx.moveTo(cx + 0.1 * s, cy - 0.5 * s);
  ctx.lineTo(cx - 0.25 * s, cy + 0.05 * s);
  ctx.lineTo(cx - 0.02 * s, cy + 0.05 * s);
  ctx.lineTo(cx - 0.1 * s, cy + 0.5 * s);
  ctx.lineTo(cx + 0.25 * s, cy - 0.08 * s);
  ctx.lineTo(cx + 0.02 * s, cy - 0.08 * s);
  ctx.closePath();
  ctx.fill();
}

const pkgTexCache = new Map<string, THREE.Texture>();
function packageTexture(type: number, zone: number, zoneNames: string[], express = false) {
  const key = `${type}-${zone}-${express}`;
  const cached = pkgTexCache.get(key);
  if (cached) return cached;
  const tex = canvasTexture(256, 256, (ctx, w, h) => {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (type === HEAVY) {
      ctx.fillStyle = '#59616d';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 900, 0.18);
      ctx.strokeStyle = '#3a4048';
      ctx.lineWidth = 16;
      ctx.strokeRect(8, 8, w - 16, h - 16);
      ctx.beginPath(); ctx.moveTo(16, 16); ctx.lineTo(w - 16, h - 16); ctx.moveTo(w - 16, 16); ctx.lineTo(16, h - 16); ctx.stroke();
      ctx.fillStyle = '#ffd23f';
      ctx.fillRect(48, 150, w - 96, 56);
      ctx.fillStyle = '#1b1b1b';
      ctx.font = 'bold 40px system-ui, sans-serif';
      ctx.fillText('500 kg', w / 2, 180);
    } else if (type === FRAGILE) {
      ctx.fillStyle = '#efe9dc';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 500, 0.08);
      ctx.strokeStyle = '#d42f2f';
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.moveTo(98, 60); ctx.quadraticCurveTo(100, 130, 128, 132); ctx.quadraticCurveTo(156, 130, 158, 60);
      ctx.moveTo(128, 132); ctx.lineTo(128, 170); ctx.moveTo(104, 172); ctx.lineTo(152, 172);
      ctx.stroke();
      ctx.fillStyle = '#d42f2f';
      ctx.font = 'bold 42px system-ui, sans-serif';
      ctx.fillText('FRÁGIL', w / 2, 212);
    } else {
      ctx.fillStyle = '#d6a14a';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 900, 0.12);
      ctx.fillStyle = 'rgba(245,225,180,0.85)';
      ctx.fillRect(w * 0.42, 0, w * 0.16, h);
      ctx.strokeStyle = 'rgba(90,55,20,0.6)';
      ctx.lineWidth = 8;
      ctx.strokeRect(4, 4, w - 8, h - 8);
    }
    ctx.fillStyle = ZONE_CSS[zone];
    ctx.beginPath(); ctx.arc(w - 58, 58, 40, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 50px system-ui, sans-serif';
    ctx.fillText(zoneNames[zone], w - 58, 61);
    if (express) {
      ctx.fillStyle = '#d42f2f';
      ctx.fillRect(0, 0, 130, 44);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 30px system-ui, sans-serif';
      ctx.fillText('EXPRESS', 65, 24);
    }
  });
  pkgTexCache.set(key, tex);
  return tex;
}

const shadowed = <T extends THREE.Mesh>(m: T) => { m.castShadow = true; m.receiveShadow = true; return m; };

// ---------------------------------------------------------------- robot

function buildRobot() {
  const white = new THREE.MeshStandardMaterial({ color: 0xf1f3f6, roughness: 0.3, metalness: 0.15 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x272c36, roughness: 0.55, metalness: 0.5 });
  const accent = new THREE.MeshStandardMaterial({ color: 0xff7a1a, roughness: 0.35, metalness: 0.3 });
  const metal = new THREE.MeshStandardMaterial({ color: 0xaab3bd, roughness: 0.2, metalness: 0.95 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x121418, roughness: 0.9 });
  const visorMat = new THREE.MeshStandardMaterial({ color: 0x05070b, roughness: 0.05, metalness: 0.9 });
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0x00e5ff, emissive: 0x00e5ff, emissiveIntensity: 0.9 });
  const tipMat = new THREE.MeshStandardMaterial({ color: 0xff3b30, emissive: 0xff3b30, emissiveIntensity: 2 });
  const add = (parent: THREE.Object3D, geo: THREE.BufferGeometry, mat: THREE.Material, x = 0, y = 0, z = 0) => {
    const m = shadowed(new THREE.Mesh(geo, mat));
    m.position.set(x, y, z);
    parent.add(m);
    return m;
  };

  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  add(body, new RoundedBoxGeometry(0.6, 0.15, 0.62, 3, 0.05), dark, 0, 0.15, 0);
  add(body, new RoundedBoxGeometry(0.62, 0.05, 0.08, 2, 0.02), accent, 0, 0.17, 0.3);
  const wheels: THREE.Group[] = [];
  const tireGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.08, 24);
  const hubGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.09, 12);
  for (const x of [-0.33, 0.33]) {
    for (const z of [-0.2, 0.2]) {
      const w = new THREE.Group();
      w.position.set(x, 0.1, z);
      add(w, tireGeo, rubber).rotation.z = Math.PI / 2;
      add(w, hubGeo, metal).rotation.z = Math.PI / 2;
      body.add(w);
      wheels.push(w);
    }
  }
  add(body, new RoundedBoxGeometry(0.46, 0.36, 0.38, 4, 0.07), white, 0, 0.42, 0);
  add(body, new RoundedBoxGeometry(0.47, 0.06, 0.39, 2, 0.02), accent, 0, 0.33, 0);
  const chestMat = eyeMat.clone();
  add(body, new RoundedBoxGeometry(0.16, 0.08, 0.02, 2, 0.01), chestMat, 0, 0.47, 0.19);
  add(body, new THREE.CylinderGeometry(0.06, 0.07, 0.1, 16), metal, 0, 0.64, 0);
  const head = new THREE.Group();
  head.position.y = 0.79;
  body.add(head);
  add(head, new RoundedBoxGeometry(0.4, 0.25, 0.32, 4, 0.08), white);
  add(head, new RoundedBoxGeometry(0.33, 0.13, 0.04, 3, 0.03), visorMat, 0, 0.0, 0.15);
  const eyeGeo = new THREE.SphereGeometry(0.032, 16, 12);
  const eyes = [-0.075, 0.075].map((x) => {
    const e = add(head, eyeGeo, eyeMat, x, 0.005, 0.172);
    e.scale.set(1, 1.15, 0.5);
    return e;
  });
  for (const s of [-1, 1]) {
    add(head, new THREE.CylinderGeometry(0.06, 0.06, 0.04, 20), accent, s * 0.21, 0, 0).rotation.z = Math.PI / 2;
  }
  add(head, new THREE.CylinderGeometry(0.008, 0.008, 0.18, 8), metal, 0.1, 0.21, -0.05);
  add(head, new THREE.SphereGeometry(0.03, 12, 10), tipMat, 0.1, 0.31, -0.05);
  const arms = [-1, 1].map((side) => {
    const shoulder = new THREE.Group();
    shoulder.position.set(side * 0.275, 0.55, 0);
    body.add(shoulder);
    add(shoulder, new THREE.SphereGeometry(0.06, 16, 12), metal);
    add(shoulder, new RoundedBoxGeometry(0.075, 0.2, 0.075, 2, 0.03), white, 0, -0.11, 0);
    const elbow = new THREE.Group();
    elbow.position.y = -0.21;
    shoulder.add(elbow);
    add(elbow, new THREE.SphereGeometry(0.045, 12, 10), metal);
    add(elbow, new RoundedBoxGeometry(0.065, 0.17, 0.065, 2, 0.025), accent, 0, -0.09, 0);
    add(elbow, new THREE.BoxGeometry(0.05, 0.05, 0.09), dark, 0, -0.19, 0);
    return { shoulder, elbow, side };
  });
  const slot = new THREE.Object3D();
  slot.position.set(0, 0.45, 0.4);
  body.add(slot);
  const light = new THREE.PointLight(0x00e5ff, 1.5, 1.6, 2);
  light.position.set(0, 0.8, 0.45);
  body.add(light);
  return { root, body, head, wheels, eyes, eyeMat, chestMat, tipMat, arms, slot, light };
}

// ---------------------------------------------------------------- escena

interface Zona { center: THREE.Vector3; width: number; minC: number; row: number; mat: THREE.MeshStandardMaterial;
  lampMat: THREE.MeshStandardMaterial; light: THREE.PointLight; stack: number }
interface PaqueteVisual { mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>; data: Paquete; size: number;
  mode: 'shelf' | 'flying' | 'carried' | 'dropping' | 'delivered'; rest: THREE.Vector3 }
interface Anim { f: Fotograma; t: number; dur: number; from: THREE.Vector3; to: THREE.Vector3; h0: number; h1: number;
  b0: number; b1: number; dir?: THREE.Vector3; pkg?: PaqueteVisual; pkgFrom?: THREE.Vector3 }
type Popup = { el: HTMLDivElement; pos: THREE.Vector3; age: number };
type Particle = { m: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>; v: THREE.Vector3; spin: THREE.Vector3;
  age: number; life: number; gravity: number };

export class Escena {
  stepMs = 200;
  onFotograma: (f: Fotograma) => void = () => {};
  onFinFotograma: (f: Fotograma) => void = () => {};
  /** Formato de la recompensa en los avisos flotantes (puntos en v1, pesos colombianos en v2). */
  formatReward: (x: number) => string = (x) => (x >= 0 ? '+' : '') + x.toFixed(1);
  /** Recompensas menores o iguales a este valor se muestran como aviso negativo. */
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
  private particles: Particle[] = [];
  private popups: Popup[] = [];
  private particleGeo = new THREE.BoxGeometry(0.05, 0.05, 0.02);
  private seed = 7;
  private batteryMax = 100;
  private useBattery = true;
  private vis = { pos: [8, 7] as number[], heading: Math.PI, carrying: false, current: 0, success: false, battery: 100,
    armLift: 0, alertUntil: 0, wheelAngle: 0, chargingUntil: 0 };
  private camTween: { t: number; p0: THREE.Vector3; t0: THREE.Vector3; p1: THREE.Vector3; t1: THREE.Vector3 } | null = null;
  private followRobot = false;
  private marker = new THREE.Group();
  private markerMat = new THREE.MeshStandardMaterial({ color: 0xffc531, emissive: 0xffb300, emissiveIntensity: 1.2 });
  private cone = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.32, 4), this.markerMat);
  private beamMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 2.4, 24, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xffc531, transparent: true, opacity: 0.08, depthWrite: false,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending }));
  private batteryBar = new THREE.Group();
  private barFill: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private rescueCable = new THREE.Group();
  private chargerPos = new THREE.Vector3();
  private chargerLight = new THREE.PointLight(0xffd23f, 0, 3, 2);
  private chargerPadMat!: THREE.MeshStandardMaterial;
  private screenMat!: THREE.MeshStandardMaterial;

  constructor(private container: HTMLElement, private popupLayer: HTMLElement, private zoneNames: string[]) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    maxAnisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.95;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x151a22);
    this.scene.fog = new THREE.Fog(0x151a22, 28, 60);
    this.scene.environment = new THREE.PMREMGenerator(this.renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    this.camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 200);
    this.camera.position.set(0, 14, 10.5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 35;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.target.set(0, 0, 0.6);
    this.controls.addEventListener('start', () => { this.camTween = null; });

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3226, 0.55));
    const sun = new THREE.DirectionalLight(0xfff3e0, 2.2);
    sun.position.set(-7, 14, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -11, right: 11, top: 9, bottom: -9, near: 1, far: 40 });
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);

    this.scene.add(this.robot.root);
    this.scene.add(this.obstacles);

    // marcador del objetivo, barra de bateria y cable de la grua de rescate
    this.cone.rotation.x = Math.PI;
    this.beamMesh.position.y = -1.2;
    this.marker.add(this.cone, this.beamMesh);
    this.scene.add(this.marker);
    const barBg = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.11),
      new THREE.MeshBasicMaterial({ color: 0x0b0e13, transparent: true, opacity: 0.8 }));
    this.barFill = new THREE.Mesh(new THREE.PlaneGeometry(0.56, 0.06), new THREE.MeshBasicMaterial({ color: 0x4fd18b }));
    this.barFill.position.z = 0.001;
    this.batteryBar.add(barBg, this.barFill);
    this.scene.add(this.batteryBar);
    const cable = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 8, 6),
      new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.8, roughness: 0.3 }));
    cable.position.y = 5;
    const hook = new THREE.Mesh(new THREE.TorusGeometry(0.08, 0.02, 8, 16),
      new THREE.MeshStandardMaterial({ color: 0xf2c018, metalness: 0.6 }));
    hook.position.y = 1.0;
    this.rescueCable.add(cable, hook);
    this.rescueCable.visible = false;
    this.scene.add(this.rescueCable);

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

  /** Prepara la escena para un nivel: obstáculos, batería y robot en el inicio. */
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

  private buildWarehouse(mapa: Mapa) {
    this.built = true;
    const layout = mapa.layout;
    this.H = layout.length;
    this.W = layout[0].length;
    const { W, H } = this;

    const floorTex = canvasTexture(256, 256, (ctx, w, h) => {
      ctx.fillStyle = '#7b8087';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 2500, 0.12);
      ctx.strokeStyle = 'rgba(30,32,36,0.35)';
      ctx.lineWidth = 3;
      ctx.strokeRect(0, 0, w, h);
    }, [W - 2, H - 2]);
    const wallTex = canvasTexture(256, 256, (ctx, w, h) => {
      ctx.fillStyle = '#6d727b';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 1500, 0.15);
      ctx.strokeStyle = 'rgba(25,27,32,0.55)';
      ctx.lineWidth = 3;
      for (let i = 0; i <= 4; i++) {
        const y = (i * h) / 4;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        for (let x = i % 2 ? w / 2 : 0; x < w; x += w / 2) {
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + h / 4); ctx.stroke();
        }
      }
    });
    const hazardTex = canvasTexture(128, 32, (ctx, w, h) => {
      ctx.fillStyle = '#f2c018';
      ctx.fillRect(0, 0, w, h);
      stripes(ctx, w, h, '#1b1b1b', 32, 16);
    });
    const cardboardTex = canvasTexture(128, 128, (ctx, w, h) => {
      ctx.fillStyle = '#b8874f';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 600, 0.12);
      ctx.fillStyle = 'rgba(230,205,160,0.75)';
      ctx.fillRect(w * 0.42, 0, w * 0.16, h);
      ctx.strokeStyle = 'rgba(80,50,20,0.5)';
      ctx.lineWidth = 4;
      ctx.strokeRect(0, 0, w, h);
    });

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), new THREE.MeshStandardMaterial({ color: 0x1b2028, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    this.scene.add(ground);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(W - 2, H - 2),
      new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.85, metalness: 0.05 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // paredes: cada casilla '#' es un bloque con franja de seguridad
    const wallCells: [number, number][] = [];
    layout.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === '#') wallCells.push([r, c]); }));
    const walls = new THREE.InstancedMesh(new THREE.BoxGeometry(1, WALL_H, 1),
      new THREE.MeshStandardMaterial({ map: wallTex, roughness: 0.9 }), wallCells.length);
    const hazard = new THREE.InstancedMesh(new THREE.BoxGeometry(1.012, 0.16, 1.012),
      new THREE.MeshStandardMaterial({ map: hazardTex, roughness: 0.6 }), wallCells.length);
    const caps = new THREE.InstancedMesh(new THREE.BoxGeometry(1.02, 0.06, 1.02),
      new THREE.MeshStandardMaterial({ color: 0x3d424b, roughness: 0.5, metalness: 0.6 }), wallCells.length);
    const m4 = new THREE.Matrix4();
    wallCells.forEach(([r, c], i) => {
      const p = this.cellPos(r, c);
      walls.setMatrixAt(i, m4.makeTranslation(p.x, WALL_H / 2, p.z));
      hazard.setMatrixAt(i, m4.makeTranslation(p.x, 0.12, p.z));
      caps.setMatrixAt(i, m4.makeTranslation(p.x, WALL_H + 0.03, p.z));
    });
    for (const m of [walls, hazard, caps]) this.scene.add(shadowed(m));

    // zonas de entrega
    this.zones = mapa.zone_cells.map((cells, z) => {
      const cols = cells.map(([, c]) => c);
      const minC = Math.min(...cols);
      const width = Math.max(...cols) - minC + 1;
      const row = cells[0][0];
      const center = this.cellPos(row, minC + (width - 1) / 2);
      const tex = canvasTexture(256 * width, 256, (ctx, w, h) => {
        ctx.fillStyle = ZONE_CSS[z];
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.fillRect(0, 0, w, h);
        stripes(ctx, w, h, 'rgba(255,255,255,0.10)', 48, 22);
        ctx.setLineDash([22, 14]);
        ctx.strokeStyle = '#f4f7f2';
        ctx.lineWidth = 10;
        ctx.strokeRect(10, 10, w - 20, h - 20);
        ctx.fillStyle = '#f4f7f2';
        ctx.font = 'bold 68px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`ZONA ${this.zoneNames[z]}`, w / 2, h / 2);
      });
      const mat = new THREE.MeshStandardMaterial({ map: tex, emissive: ZONE_COLORS[z], emissiveIntensity: 0.1, roughness: 0.6 });
      const pad = new THREE.Mesh(new THREE.PlaneGeometry(width, 1), mat);
      pad.rotation.x = -Math.PI / 2;
      pad.position.set(center.x, 0.012, center.z);
      pad.receiveShadow = true;
      this.scene.add(pad);
      const frameMat = new THREE.MeshStandardMaterial({ color: ZONE_COLORS[z], roughness: 0.4, metalness: 0.5 });
      for (const s of [-1, 1]) {
        const post = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.5, 0.08), frameMat));
        post.position.set(center.x + s * (width / 2 + 0.02), 0.75, center.z + 0.45);
        this.scene.add(post);
      }
      const beam = shadowed(new THREE.Mesh(new THREE.BoxGeometry(width + 0.12, 0.12, 0.12), frameMat));
      beam.position.set(center.x, 1.5, center.z + 0.45);
      this.scene.add(beam);
      const lampMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: ZONE_COLORS[z], emissiveIntensity: 0.6 });
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(width - 0.4, 0.04, 0.08), lampMat);
      lamp.position.set(center.x, 1.43, center.z + 0.45);
      this.scene.add(lamp);
      const light = new THREE.PointLight(ZONE_COLORS[z], 0, 4, 2);
      light.position.set(center.x, 1.2, center.z);
      this.scene.add(light);
      return { center, width, minC, row, mat, lampMat, light, stack: 0 };
    });

    // estacion de carga
    this.chargerPos = this.cellPos(mapa.charger[0], mapa.charger[1]);
    const chargerTex = canvasTexture(256, 256, (ctx, w, h) => {
      ctx.fillStyle = '#23272f';
      ctx.fillRect(0, 0, w, h);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.rect(26, 26, w - 52, h - 52); ctx.clip('evenodd');
      ctx.fillStyle = '#f2c018';
      ctx.fillRect(0, 0, w, h);
      stripes(ctx, w, h, '#1b1b1b', 40, 20);
      ctx.restore();
      ctx.fillStyle = '#ffd23f';
      bolt(ctx, w / 2, h / 2, 150);
    });
    this.chargerPadMat = new THREE.MeshStandardMaterial({ map: chargerTex, emissive: 0xffc400, emissiveIntensity: 0.05, roughness: 0.6 });
    const chargerPad = new THREE.Mesh(new THREE.PlaneGeometry(0.96, 0.96), this.chargerPadMat);
    chargerPad.rotation.x = -Math.PI / 2;
    chargerPad.position.set(this.chargerPos.x, 0.012, this.chargerPos.z);
    this.scene.add(chargerPad);
    const pillar = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.56, 1.0, 0.2, 3, 0.05),
      new THREE.MeshStandardMaterial({ color: 0x2f353f, roughness: 0.4, metalness: 0.6 })));
    pillar.position.set(this.chargerPos.x, 0.5, this.chargerPos.z - 0.52);
    this.scene.add(pillar);
    const screenTex = canvasTexture(128, 128, (ctx, w, h) => {
      ctx.fillStyle = '#062a12';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#3dff8b';
      bolt(ctx, w / 2, h / 2, 90);
    });
    this.screenMat = new THREE.MeshStandardMaterial({ map: screenTex, emissive: 0x3dff8b, emissiveMap: screenTex, emissiveIntensity: 0.6 });
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.34, 0.34), this.screenMat);
    screen.position.set(this.chargerPos.x, 0.72, this.chargerPos.z - 0.415);
    this.scene.add(screen);
    this.chargerLight.position.set(this.chargerPos.x, 0.9, this.chargerPos.z);
    this.scene.add(this.chargerLight);

    // estanterias
    const postGeo = new THREE.BoxGeometry(0.06, RACK_H, 0.06);
    const postMat = new THREE.MeshStandardMaterial({ color: 0xe8681a, roughness: 0.45, metalness: 0.4 });
    const plankGeo = new THREE.BoxGeometry(0.94, 0.04, 0.94);
    const plankMat = new THREE.MeshStandardMaterial({ color: 0x2b5ea8, roughness: 0.4, metalness: 0.55 });
    const beamGeo = new THREE.BoxGeometry(0.94, 0.06, 0.05);
    const palletGeo = new THREE.BoxGeometry(0.8, 0.1, 0.8);
    const palletMat = new THREE.MeshStandardMaterial({ color: 0x9a7448, roughness: 0.9 });
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxMat = new THREE.MeshStandardMaterial({ map: cardboardTex, roughness: 0.85 });
    const addBoxes = (group: THREE.Group, y: number, count: number) => {
      for (let i = 0; i < count; i++) {
        const s = 0.22 + this.rand() * 0.18;
        const b = shadowed(new THREE.Mesh(boxGeo, boxMat));
        b.scale.set(s * (0.9 + this.rand() * 0.4), s, s);
        b.position.set((this.rand() - 0.5) * 0.45, y + s / 2, (this.rand() - 0.5) * 0.45);
        b.rotation.y = (this.rand() - 0.5) * 0.4;
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
      for (const y of [0.55, 1.03]) {
        const plank = shadowed(new THREE.Mesh(plankGeo, plankMat));
        plank.position.y = y;
        rack.add(plank);
        addBoxes(rack, y + 0.02, 1 + Math.floor(this.rand() * 3));
      }
      for (const [z, rot] of [[-0.45, 0], [0.45, 0], [-0.45, 1], [0.45, 1]]) {
        const b = shadowed(new THREE.Mesh(beamGeo, plankMat));
        b.position.set(rot ? z : 0, RACK_H - 0.03, rot ? 0 : z);
        b.rotation.y = (rot * Math.PI) / 2;
        rack.add(b);
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

  /** Obstáculo del pasillo bloqueado (nivel N3): conos, barrera y cajas caídas. */
  private buildObstacle(r: number, c: number) {
    const g = new THREE.Group();
    g.position.copy(this.cellPos(r, c));
    const coneMat = new THREE.MeshStandardMaterial({ color: 0xff6a00, roughness: 0.5 });
    const stripeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
    for (const [x, z] of [[-0.32, -0.32], [0.32, -0.32], [-0.32, 0.32], [0.32, 0.32]]) {
      const cone = shadowed(new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.42, 16), coneMat));
      cone.position.set(x, 0.21, z);
      const ring = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.085, 0.06, 16), stripeMat);
      ring.position.set(x, 0.25, z);
      g.add(cone, ring);
    }
    const barTex = canvasTexture(256, 32, (ctx, w, h) => {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);
      stripes(ctx, w, h, '#e0262b', 40, 20);
    });
    const bar = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.08, 0.05), new THREE.MeshStandardMaterial({ map: barTex })));
    bar.position.set(0, 0.36, 0);
    g.add(bar);
    const spill = new THREE.Mesh(new THREE.CircleGeometry(0.45, 24),
      new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.05, metalness: 0.4, transparent: true, opacity: 0.75 }));
    spill.rotation.x = -Math.PI / 2;
    spill.position.y = 0.014;
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

  private slotWorld() { return this.robot.slot.getWorldPosition(new THREE.Vector3()); }

  private buildOrder(order: Paquete[]) {
    for (const p of this.pkgs) {
      this.scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      p.mesh.material.dispose();
    }
    this.zones.forEach((z) => { z.stack = 0; });
    this.pkgs = order.map((data) => {
      const size = PKG_SIZES[data.type];
      const mesh = shadowed(new THREE.Mesh(new RoundedBoxGeometry(size, size, size, 2, 0.03),
        new THREE.MeshStandardMaterial({ map: packageTexture(data.type, data.zone, this.zoneNames, data.service === 1),
          roughness: data.type === HEAVY ? 0.35 : 0.55, metalness: data.type === HEAVY ? 0.6 : 0,
          emissive: 0xffb300, emissiveIntensity: 0 })));
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

  // ---------------------------------------------------------------- efectos

  private burst(pos: THREE.Vector3, colors: number[], count: number, speed: number, life: number, gravity = 6) {
    for (let i = 0; i < count; i++) {
      const m = new THREE.Mesh(this.particleGeo, new THREE.MeshBasicMaterial({ color: colors[i % colors.length], transparent: true }));
      m.position.copy(pos);
      const v = new THREE.Vector3(this.rand() - 0.5, this.rand() * 0.8 + 0.4, this.rand() - 0.5).normalize()
        .multiplyScalar(speed * (0.5 + this.rand()));
      this.particles.push({ m, v, gravity, spin: new THREE.Vector3(this.rand(), this.rand(), this.rand()).multiplyScalar(10), age: 0, life });
      this.scene.add(m);
    }
  }

  private popup(text: string, cls: string, worldPos: THREE.Vector3) {
    const el = document.createElement('div');
    el.className = `popup ${cls}`;
    el.textContent = text;
    this.popupLayer.appendChild(el);
    this.popups.push({ el, pos: worldPos.clone(), age: 0 });
  }

  private updateEffects(dt: number) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.age += dt;
      p.v.y -= p.gravity * dt;
      p.m.position.addScaledVector(p.v, dt);
      if (p.m.position.y < 0.02) { p.m.position.y = 0.02; p.v.multiplyScalar(0.4); }
      p.m.rotation.x += p.spin.x * dt;
      p.m.rotation.y += p.spin.y * dt;
      p.m.material.opacity = 1 - p.age / p.life;
      if (p.age >= p.life) {
        this.scene.remove(p.m);
        p.m.material.dispose();
        this.particles.splice(i, 1);
      }
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    for (let i = this.popups.length - 1; i >= 0; i--) {
      const p = this.popups[i];
      p.age += dt;
      v.copy(p.pos).add(new THREE.Vector3(0, p.age * 0.8, 0)).project(this.camera);
      p.el.style.left = `${rect.left + (v.x * 0.5 + 0.5) * rect.width}px`;
      p.el.style.top = `${rect.top + (-v.y * 0.5 + 0.5) * rect.height}px`;
      p.el.style.opacity = String(1 - p.age / 1.4);
      if (p.age > 1.4) { p.el.remove(); this.popups.splice(i, 1); }
    }
  }

  // ---------------------------------------------------------------- animacion de fotogramas

  private durationFor(f: Fotograma) {
    let k = DURATIONS[f.event] ?? 1;
    if (f.event === 'move' && this.vis.carrying && this.pkgs[this.vis.current]?.data.type === HEAVY) k = 1.5;
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
        a.pkgFrom = this.slotWorld();
        p.rest.copy(this.zoneSlot(p.data.zone, p.size));
        break;
      }
      case 'break':
        a.pkg = this.pkgs[f.current];
        break;
      case 'charge':
        vis.chargingUntil = performance.now() + a.dur;
        break;
      case 'rescue':
        this.rescueCable.visible = true;
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
      p.copy(a.from).addScaledVector(a.dir, 0.22 * Math.sin(Math.PI * Math.min(1, t * 1.6)));
    }
    if (f.event === 'rescue') {
      const lift = t < 0.3 ? ease(t / 0.3) : t > 0.75 ? ease((1 - t) / 0.25) : 1;
      p.copy(a.from).lerp(a.to, ease(clamp01((t - 0.3) / 0.45)));
      p.y = lift * 2.2;
      robot.root.rotation.z = Math.sin(t * Math.PI * 6) * 0.08 * lift;
      this.rescueCable.position.copy(p);
    }
    robot.root.position.copy(p);
    this.vis.heading = lerpAngle(a.h0, a.h1, clamp01(t / 0.35));
    robot.root.rotation.y = this.vis.heading;
    if (f.event === 'move' || f.event === 'bump') robot.body.position.y = Math.abs(Math.sin(t * Math.PI * 2)) * 0.02;
    if (['pick_fail', 'drop_fail', 'wrong_zone', 'charge_fail'].includes(f.event)) {
      robot.head.rotation.y = Math.sin(t * Math.PI * 4) * 0.45 * (1 - t);
    }
    if (f.event === 'pick' && a.pkg) {
      const u = ease(clamp01((t - 0.3) / 0.7));
      a.pkg.mesh.position.copy(a.pkg.rest).lerp(this.slotWorld(), u);
      a.pkg.mesh.position.y += Math.sin(Math.PI * u) * 0.35;
      a.pkg.mesh.rotation.y = lerpAngle(0, this.vis.heading, u);
    }
    if (f.event === 'drop' && a.pkg && a.pkgFrom) {
      const u = ease(clamp01((t - 0.15) / 0.85));
      a.pkg.mesh.position.copy(a.pkgFrom).lerp(a.pkg.rest, u);
      a.pkg.mesh.rotation.y = lerpAngle(this.vis.heading, 0, u);
    }
    if (f.event === 'charge') this.vis.battery = a.b0 + (a.b1 - a.b0) * ease(t);
    else if (f.event === 'rescue') this.vis.battery = t > 0.9 ? a.b1 : Math.max(0, a.b0 - 1);
    else this.vis.battery = a.b0 + (a.b1 - a.b0) * t;
  }

  private finishAnim(a: Anim) {
    const { f } = a;
    const robot = this.robot;
    this.vis.pos = f.pos.slice();
    robot.head.rotation.y = 0;
    robot.body.position.y = 0;
    robot.root.rotation.z = 0;
    robot.root.position.copy(this.cellPos(f.pos[0], f.pos[1]));
    this.rescueCable.visible = false;
    Object.assign(this.vis, { carrying: f.carrying, current: f.current, success: f.success, battery: f.battery });
    const head = this.cellPos(f.pos[0], f.pos[1], 1.25);
    switch (f.event) {
      case 'pick':
        if (a.pkg) a.pkg.mode = 'carried';
        this.popup(`${this.formatReward(f.reward)} ${TYPE_LABELS[a.pkg?.data.type ?? 0].toLowerCase()} recogido`, 'gold', head);
        break;
      case 'drop':
        if (a.pkg) {
          a.pkg.mode = 'delivered';
          a.pkg.mesh.position.copy(a.pkg.rest);
        }
        if (f.success) {
          this.popup(`${this.formatReward(f.reward)} ¡pedido completo!`, 'pos', head);
          this.burst(this.cellPos(f.pos[0], f.pos[1], 0.6), [0x3dff8b, 0xffc531, 0xffffff, 0x2fa8ff, 0xff5a8a], 90, 3.4, 1.8);
        } else {
          const late = f.order[f.current - 1]?.late;
          this.popup(`${this.formatReward(f.reward)} ${late ? 'entregado TARDE' : 'entregado'}`, late ? 'neg' : 'pos', head);
          this.burst(this.cellPos(f.pos[0], f.pos[1], 0.6), [ZONE_COLORS[a.pkg?.data.zone ?? 0], 0xffffff], 30, 2.4, 1.2);
        }
        break;
      case 'break':
        if (a.dir) this.burst(this.cellPos(f.pos[0], f.pos[1], 0.45).addScaledVector(a.dir, 0.5), [0xffffff, 0xe8e2d4, 0xd42f2f], 40, 2.2, 1.2);
        this.popup(`${this.formatReward(f.reward)} ¡se rompió!`, 'neg', head);
        break;
      case 'bump':
        if (a.dir) this.burst(this.cellPos(f.pos[0], f.pos[1], 0.35).addScaledVector(a.dir, 0.5), [0xffb02e, 0xff6a00, 0xfff1a0], 14, 1.6, 0.5);
        this.popup(this.formatReward(f.reward), 'neg', head);
        break;
      case 'charge':
        this.popup('⚡ batería llena', 'gold', head);
        break;
      case 'rescue':
        this.popup(`${this.formatReward(f.reward)} rescate: sin batería`, 'neg', head);
        break;
      case 'wrong_zone':
        this.popup(`${this.formatReward(f.reward)} zona equivocada`, 'neg', head);
        break;
      default:
        if (f.reward <= this.alertThreshold) this.popup(this.formatReward(f.reward), 'neg', head);
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
    const target = vis.carrying || this.anim?.f.event === 'pick' ? 1 : 0;
    vis.armLift += (target - vis.armLift) * Math.min(1, dt * 8);
    for (const arm of robot.arms) {
      arm.shoulder.rotation.x = -0.15 - vis.armLift * 1.25 + Math.sin(now / 600 + arm.side) * 0.03 * (1 - vis.armLift);
      arm.shoulder.rotation.z = arm.side * (0.12 - vis.armLift * 0.16);
      arm.elbow.rotation.x = -0.25 - vis.armLift * 0.35;
    }
    const moving = this.anim && ['move', 'bump', 'break'].includes(this.anim.f.event);
    if (moving && this.anim) vis.wheelAngle += dt * (1000 / this.anim.dur) * 10;
    robot.wheels.forEach((w) => { w.rotation.x = vis.wheelAngle; });
    if (!this.anim) robot.head.position.y = 0.79 + Math.sin(now / 500) * 0.008;

    const frac = vis.battery / this.batteryMax;
    const low = this.useBattery && frac <= 0.25;
    const mood = now < vis.alertUntil ? 'bad' : vis.carrying ? 'carry' : low ? 'low' : 'normal';
    const color = EYE_COLORS[mood];
    robot.eyeMat.color.setHex(color);
    robot.eyeMat.emissive.setHex(color);
    robot.chestMat.emissive.setHex(color);
    robot.light.color.setHex(color);
    const blink = now % 3200 < 120 ? 0.15 : 1;
    robot.eyes.forEach((e) => { e.scale.y = 1.15 * blink; });
    const period = low ? 300 : 1000;
    robot.tipMat.emissiveIntensity = now % period < period / 2 ? 2.5 : 0.3;

    this.batteryBar.position.copy(robot.root.position).add(new THREE.Vector3(0, 1.28, 0));
    this.batteryBar.quaternion.copy(this.camera.quaternion);
    this.barFill.scale.x = Math.max(0.001, frac);
    this.barFill.position.x = -0.28 * (1 - frac);
    this.barFill.material.color.setHex(batteryColor(frac));
  }

  private updateWorld(now: number) {
    const { vis } = this;
    this.pkgs.forEach((p, i) => {
      if (p.mode === 'shelf') {
        p.mesh.position.copy(this.shelfPkgPos(p.data.shelf, p.size));
        p.mesh.rotation.y = 0;
      } else if (p.mode === 'carried') {
        p.mesh.position.copy(this.slotWorld());
        p.mesh.rotation.y = this.robot.root.rotation.y;
      }
      const isTarget = i === vis.current && p.mode === 'shelf';
      p.mesh.material.emissiveIntensity = isTarget ? 0.15 + 0.15 * Math.sin(now / 250) : 0;
    });
    const cur = this.pkgs[vis.current];
    let goal: THREE.Vector3 | null = null;
    let color = 0xffc531;
    if (cur && !vis.success) {
      if (vis.carrying) {
        goal = this.zones[cur.data.zone].center.clone().setY(1.95);
        color = ZONE_COLORS[cur.data.zone];
      } else {
        goal = this.shelfPkgPos(cur.data.shelf, cur.size).setY(RACK_H + 0.55);
      }
    }
    this.marker.visible = goal !== null;
    if (goal) {
      if (this.marker.position.lengthSq() === 0) this.marker.position.copy(goal);
      this.marker.position.lerp(goal, 0.08);
      this.cone.position.y = Math.sin(now / 300) * 0.08;
      this.marker.rotation.y = now / 700;
      this.markerMat.color.setHex(color);
      this.markerMat.emissive.setHex(color);
      this.beamMesh.material.color.setHex(color);
    }
    this.zones.forEach((z, i) => {
      const active = vis.carrying && cur && cur.data.zone === i;
      z.light.intensity = active ? 4 + Math.sin(now / 200) * 1.5 : 0;
      z.mat.emissiveIntensity = active ? 0.35 + Math.sin(now / 200) * 0.15 : 0.08;
      z.lampMat.emissiveIntensity = active ? 2 : 0.5;
    });
    const charging = now < vis.chargingUntil;
    const low = this.useBattery && vis.battery / this.batteryMax <= 0.25;
    this.chargerLight.intensity = charging ? 5 + Math.sin(now / 60) * 2 : low ? 2 + Math.sin(now / 150) * 1.5 : 0;
    this.chargerPadMat.emissiveIntensity = charging ? 0.6 : low ? 0.2 + Math.sin(now / 150) * 0.15 : 0.05;
    this.screenMat.emissiveIntensity = charging ? 2 : 0.6;
    if (charging && Math.random() < 0.35) {
      this.burst(this.robot.root.position.clone().add(new THREE.Vector3(0, 0.6, 0)), [0xffd23f, 0xfff6b0, 0x3dff8b], 2, 1.2, 0.6, 1);
    }
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

  /** Acción de movimiento según hacia dónde mira la cámara ("arriba" = hacia el fondo de la pantalla). */
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
    this.updateEffects(dt);
    this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.loop);
  };
}
