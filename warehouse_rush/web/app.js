// Visualizacion 3D de Warehouse Rush v1 con three.js.
// Python ejecuta el entorno; esta pagina recibe fotogramas por la API y los anima.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const [UP, DOWN, LEFT, RIGHT, PICK, DROP, CHARGE] = [0, 1, 2, 3, 4, 5, 6];
const MOVES = { [UP]: [-1, 0], [DOWN]: [1, 0], [LEFT]: [0, -1], [RIGHT]: [0, 1] };
const [NORMAL, HEAVY, FRAGILE] = [0, 1, 2];
const TYPE_LABELS = ['Normal', 'Pesado', 'Frágil'];
const PKG_SIZES = [0.36, 0.4, 0.34];
const ZONE_COLORS = [0xe5484d, 0x3b82f6, 0xa855f7];
const ZONE_CSS = ['#e5484d', '#3b82f6', '#a855f7'];

const WALL_H = 0.7;
const RACK_H = 1.5;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

const api = {
  async get(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res.json();
  },
  async post(path, body = {}) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res.json();
  },
};

function showError(msg) {
  $('error').hidden = false;
  $('error').textContent = msg;
}

let config;
try {
  config = await api.get('/api/config');
} catch (e) {
  showError('No se pudo conectar con el servidor de Python. Ejecuta play3d.py o train3d.py.');
  throw e;
}

const ACTION_NAMES = config.action_names;
const ZONE_NAMES = config.zone_names;
const BAT_MAX = config.battery_max;
const layout = config.layout;
const H = layout.length;
const W = layout[0].length;
const cellPos = (r, c, y = 0) => new THREE.Vector3(c - (W - 1) / 2, y, r - (H - 1) / 2);
const headingOf = (dr, dc) => Math.atan2(dc, dr); // el frente del robot mira hacia +z

// ---------------------------------------------------------------- escena

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
$('scene').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x151a22);
scene.fog = new THREE.Fog(0x151a22, 28, 60);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 14, 10.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.5;
controls.maxDistance = 35;
controls.maxPolarAngle = Math.PI * 0.47;
controls.target.set(0, 0, 0.5);

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3226, 0.55));
const sun = new THREE.DirectionalLight(0xfff3e0, 2.2);
sun.position.set(-7, 14, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -11, right: 11, top: 9, bottom: -9, near: 1, far: 40 });
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.02;
scene.add(sun);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------- texturas

function canvasTexture(w, h, draw, repeat) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  draw(canvas.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  if (repeat) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(...repeat);
  }
  return tex;
}

function speckle(ctx, w, h, n, alpha) {
  for (let i = 0; i < n; i++) {
    const v = Math.random() < 0.5 ? 0 : 255;
    ctx.fillStyle = `rgba(${v},${v},${v},${Math.random() * alpha})`;
    ctx.fillRect(Math.random() * w, Math.random() * h, 2, 2);
  }
}

function stripes(ctx, w, h, color, gap, width) {
  ctx.fillStyle = color;
  for (let x = -h; x < w + h; x += gap) {
    ctx.beginPath();
    ctx.moveTo(x, h); ctx.lineTo(x + width, h); ctx.lineTo(x + width + h, 0); ctx.lineTo(x + h, 0);
    ctx.fill();
  }
}

function bolt(ctx, cx, cy, s) {
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
  const rows = 4;
  for (let i = 0; i <= rows; i++) {
    const y = (i * h) / rows;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    const off = i % 2 ? w / 2 : 0;
    for (let x = off; x < w; x += w / 2) {
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + h / rows); ctx.stroke();
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

// textura de cada paquete segun su tipo, con una etiqueta del color de su zona de destino
const pkgTexCache = {};
function packageTexture(type, zone) {
  const key = `${type}-${zone}`;
  if (pkgTexCache[key]) return pkgTexCache[key];
  pkgTexCache[key] = canvasTexture(256, 256, (ctx, w, h) => {
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
      ctx.fillStyle = '#c9ced6';
      for (const [x, y] of [[22, 22], [w - 22, 22], [22, h - 22], [w - 22, h - 22]]) {
        ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = '#ffd23f';
      ctx.fillRect(48, 150, w - 96, 56);
      ctx.fillStyle = '#1b1b1b';
      ctx.font = 'bold 40px system-ui, sans-serif';
      ctx.fillText('500 kg', w / 2, 180);
    } else if (type === FRAGILE) {
      ctx.fillStyle = '#efe9dc';
      ctx.fillRect(0, 0, w, h);
      speckle(ctx, w, h, 500, 0.08);
      ctx.strokeStyle = '#c8bfae';
      ctx.lineWidth = 8;
      ctx.strokeRect(4, 4, w - 8, h - 8);
      ctx.strokeStyle = '#d42f2f';
      ctx.lineWidth = 9;
      ctx.beginPath(); // copa
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
    // etiqueta de zona
    ctx.fillStyle = ZONE_CSS[zone];
    ctx.beginPath(); ctx.arc(w - 58, 58, 40, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 50px system-ui, sans-serif';
    ctx.fillText(ZONE_NAMES[zone], w - 58, 61);
  });
  return pkgTexCache[key];
}

// ---------------------------------------------------------------- bodega

const shadowed = (m) => { m.castShadow = true; m.receiveShadow = true; return m; };

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(120, 120),
  new THREE.MeshStandardMaterial({ color: 0x1b2028, roughness: 1 }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.01;
ground.receiveShadow = true;
scene.add(ground);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(W - 2, H - 2),
  new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.85, metalness: 0.05 }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

// paredes: cada casilla '#' es un bloque, con franja de seguridad en la base
const wallCells = [];
layout.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === '#') wallCells.push([r, c]); }));
const wallMesh = new THREE.InstancedMesh(
  new THREE.BoxGeometry(1, WALL_H, 1),
  new THREE.MeshStandardMaterial({ map: wallTex, roughness: 0.9 }),
  wallCells.length,
);
const hazardMesh = new THREE.InstancedMesh(
  new THREE.BoxGeometry(1.012, 0.16, 1.012),
  new THREE.MeshStandardMaterial({ map: hazardTex, roughness: 0.6 }),
  wallCells.length,
);
const capMesh = new THREE.InstancedMesh(
  new THREE.BoxGeometry(1.02, 0.06, 1.02),
  new THREE.MeshStandardMaterial({ color: 0x3d424b, roughness: 0.5, metalness: 0.6 }),
  wallCells.length,
);
const m4 = new THREE.Matrix4();
wallCells.forEach(([r, c], i) => {
  const p = cellPos(r, c);
  wallMesh.setMatrixAt(i, m4.makeTranslation(p.x, WALL_H / 2, p.z));
  hazardMesh.setMatrixAt(i, m4.makeTranslation(p.x, 0.12, p.z));
  capMesh.setMatrixAt(i, m4.makeTranslation(p.x, WALL_H + 0.03, p.z));
});
[wallMesh, hazardMesh, capMesh].forEach((m) => scene.add(shadowed(m)));

// zonas de entrega: una por color, cada una con su portico y su luz
const zones = config.zone_cells.map((cells, z) => {
  const cols = cells.map(([, c]) => c);
  const minC = Math.min(...cols);
  const width = Math.max(...cols) - minC + 1;
  const row = cells[0][0];
  const center = cellPos(row, minC + (width - 1) / 2);
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
    ctx.fillText(`ZONA ${ZONE_NAMES[z]}`, w / 2, h / 2);
  });
  const mat = new THREE.MeshStandardMaterial({ map: tex, emissive: ZONE_COLORS[z], emissiveIntensity: 0.1, roughness: 0.6 });
  const pad = new THREE.Mesh(new THREE.PlaneGeometry(width, 1), mat);
  pad.rotation.x = -Math.PI / 2;
  pad.position.set(center.x, 0.012, center.z);
  pad.receiveShadow = true;
  scene.add(pad);

  const frameMat = new THREE.MeshStandardMaterial({ color: ZONE_COLORS[z], roughness: 0.4, metalness: 0.5 });
  for (const s of [-1, 1]) {
    const post = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.5, 0.08), frameMat));
    post.position.set(center.x + s * (width / 2 + 0.02), 0.75, center.z + 0.45);
    scene.add(post);
  }
  const beam = shadowed(new THREE.Mesh(new THREE.BoxGeometry(width + 0.12, 0.12, 0.12), frameMat));
  beam.position.set(center.x, 1.5, center.z + 0.45);
  scene.add(beam);
  const lampMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: ZONE_COLORS[z], emissiveIntensity: 0.6 });
  const lamp = new THREE.Mesh(new THREE.BoxGeometry(width - 0.4, 0.04, 0.08), lampMat);
  lamp.position.set(center.x, 1.43, center.z + 0.45);
  scene.add(lamp);
  const light = new THREE.PointLight(ZONE_COLORS[z], 0, 4, 2);
  light.position.set(center.x, 1.2, center.z);
  scene.add(light);
  return { cells, center, width, minC, row, mat, lampMat, light, stack: 0 };
});

// estacion de carga
const chargerPos = cellPos(...config.charger);
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
const chargerPadMat = new THREE.MeshStandardMaterial({ map: chargerTex, emissive: 0xffc400, emissiveIntensity: 0.05, roughness: 0.6 });
const chargerPad = new THREE.Mesh(new THREE.PlaneGeometry(0.96, 0.96), chargerPadMat);
chargerPad.rotation.x = -Math.PI / 2;
chargerPad.position.set(chargerPos.x, 0.012, chargerPos.z);
chargerPad.receiveShadow = true;
scene.add(chargerPad);

const pillarMat = new THREE.MeshStandardMaterial({ color: 0x2f353f, roughness: 0.4, metalness: 0.6 });
const pillar = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.56, 1.0, 0.2, 3, 0.05), pillarMat));
pillar.position.set(chargerPos.x, 0.5, chargerPos.z - 0.52);
scene.add(pillar);
const screenTex = canvasTexture(128, 128, (ctx, w, h) => {
  ctx.fillStyle = '#062a12';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#3dff8b';
  bolt(ctx, w / 2, h / 2, 90);
});
const screenMat = new THREE.MeshStandardMaterial({ map: screenTex, emissive: 0x3dff8b, emissiveMap: screenTex, emissiveIntensity: 0.6 });
const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.34, 0.34), screenMat);
screen.position.set(chargerPos.x, 0.72, chargerPos.z - 0.415);
scene.add(screen);
const chargerLight = new THREE.PointLight(0xffd23f, 0, 3, 2);
chargerLight.position.set(chargerPos.x, 0.9, chargerPos.z);
scene.add(chargerLight);

// estanterias industriales
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const postGeo = new THREE.BoxGeometry(0.06, RACK_H, 0.06);
const postMat = new THREE.MeshStandardMaterial({ color: 0xe8681a, roughness: 0.45, metalness: 0.4 });
const plankGeo = new THREE.BoxGeometry(0.94, 0.04, 0.94);
const plankMat = new THREE.MeshStandardMaterial({ color: 0x2b5ea8, roughness: 0.4, metalness: 0.55 });
const palletGeo = new THREE.BoxGeometry(0.8, 0.1, 0.8);
const palletMat = new THREE.MeshStandardMaterial({ color: 0x9a7448, roughness: 0.9 });
const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const boxMat = new THREE.MeshStandardMaterial({ map: cardboardTex, roughness: 0.85 });
const LEVELS = [0.55, 1.03];
const beamGeo = new THREE.BoxGeometry(0.94, 0.06, 0.05);

function addBoxes(group, y, count) {
  for (let i = 0; i < count; i++) {
    const s = 0.22 + rand() * 0.18;
    const b = shadowed(new THREE.Mesh(boxGeo, boxMat));
    b.scale.set(s * (0.9 + rand() * 0.4), s, s);
    b.position.set((rand() - 0.5) * 0.45, y + s / 2, (rand() - 0.5) * 0.45);
    b.rotation.y = (rand() - 0.5) * 0.4;
    group.add(b);
  }
}

const racks = config.shelf_cells.map(([r, c]) => {
  const rack = new THREE.Group();
  rack.position.copy(cellPos(r, c));
  for (const [x, z] of [[-0.45, -0.45], [0.45, -0.45], [-0.45, 0.45], [0.45, 0.45]]) {
    const post = shadowed(new THREE.Mesh(postGeo, postMat));
    post.position.set(x, RACK_H / 2, z);
    rack.add(post);
  }
  for (const y of LEVELS) {
    const plank = shadowed(new THREE.Mesh(plankGeo, plankMat));
    plank.position.y = y;
    rack.add(plank);
    addBoxes(rack, y + 0.02, 1 + Math.floor(rand() * 3));
  }
  // nivel superior abierto (solo travesanos) para que se vea el interior
  for (const [z, rot] of [[-0.45, 0], [0.45, 0], [-0.45, 1], [0.45, 1]]) {
    const b = shadowed(new THREE.Mesh(beamGeo, plankMat));
    b.position.set(rot ? z : 0, RACK_H - 0.03, rot ? 0 : z);
    b.rotation.y = rot * Math.PI / 2;
    rack.add(b);
  }
  const pallet = shadowed(new THREE.Mesh(palletGeo, palletMat));
  pallet.position.y = 0.05;
  rack.add(pallet);
  const bottom = new THREE.Group(); // cajas del nivel inferior: se ocultan donde hay un paquete del pedido
  addBoxes(bottom, 0.1, 1 + Math.floor(rand() * 2));
  rack.add(bottom);
  scene.add(rack);
  return { rack, bottom, cell: [r, c] };
});

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
  const add = (parent, geo, mat, x = 0, y = 0, z = 0) => {
    const m = shadowed(new THREE.Mesh(geo, mat));
    m.position.set(x, y, z);
    parent.add(m);
    return m;
  };

  const root = new THREE.Group();
  const body = new THREE.Group(); // sube y baja levemente al moverse
  root.add(body);

  // chasis con ruedas
  add(body, new RoundedBoxGeometry(0.6, 0.15, 0.62, 3, 0.05), dark, 0, 0.15, 0);
  add(body, new RoundedBoxGeometry(0.62, 0.05, 0.08, 2, 0.02), accent, 0, 0.17, 0.3);
  const wheels = [];
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

  // torso
  add(body, new RoundedBoxGeometry(0.46, 0.36, 0.38, 4, 0.07), white, 0, 0.42, 0);
  add(body, new RoundedBoxGeometry(0.47, 0.06, 0.39, 2, 0.02), accent, 0, 0.33, 0);
  const chestMat = eyeMat.clone();
  add(body, new RoundedBoxGeometry(0.16, 0.08, 0.02, 2, 0.01), chestMat, 0, 0.47, 0.19);

  // cuello y cabeza
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

  // brazos (hombro -> codo -> pinza)
  const arms = [-1, 1].map((s) => {
    const shoulder = new THREE.Group();
    shoulder.position.set(s * 0.275, 0.55, 0);
    body.add(shoulder);
    add(shoulder, new THREE.SphereGeometry(0.06, 16, 12), metal);
    add(shoulder, new RoundedBoxGeometry(0.075, 0.2, 0.075, 2, 0.03), white, 0, -0.11, 0);
    const elbow = new THREE.Group();
    elbow.position.y = -0.21;
    shoulder.add(elbow);
    add(elbow, new THREE.SphereGeometry(0.045, 12, 10), metal);
    add(elbow, new RoundedBoxGeometry(0.065, 0.17, 0.065, 2, 0.025), accent, 0, -0.09, 0);
    add(elbow, new THREE.BoxGeometry(0.05, 0.05, 0.09), dark, 0, -0.19, 0);
    return { shoulder, elbow, side: s };
  });

  // punto donde va el paquete al cargarlo
  const slot = new THREE.Object3D();
  slot.position.set(0, 0.45, 0.4);
  body.add(slot);

  const light = new THREE.PointLight(0x00e5ff, 1.5, 1.6, 2);
  light.position.set(0, 0.8, 0.45);
  body.add(light);

  return { root, body, head, wheels, eyes, eyeMat, chestMat, tipMat, arms, slot, light };
}

const robot = buildRobot();
scene.add(robot.root);

// cable de la grua que rescata al robot cuando se queda sin bateria
const rescueCable = new THREE.Group();
const cableLine = new THREE.Mesh(
  new THREE.CylinderGeometry(0.015, 0.015, 8, 6),
  new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.8, roughness: 0.3 }),
);
cableLine.position.y = 5;
rescueCable.add(cableLine);
const hook = new THREE.Mesh(new THREE.TorusGeometry(0.08, 0.02, 8, 16), new THREE.MeshStandardMaterial({ color: 0xf2c018, metalness: 0.6 }));
hook.position.y = 1.0;
rescueCable.add(hook);
rescueCable.visible = false;
scene.add(rescueCable);

// barra de bateria sobre el robot (siempre mirando a la camara)
const batteryBar = new THREE.Group();
const barBg = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.11), new THREE.MeshBasicMaterial({ color: 0x0b0e13, transparent: true, opacity: 0.8 }));
const barFillMat = new THREE.MeshBasicMaterial({ color: 0x4fd18b });
const barFill = new THREE.Mesh(new THREE.PlaneGeometry(0.56, 0.06), barFillMat);
barFill.position.z = 0.001;
batteryBar.add(barBg, barFill);
batteryBar.visible = config.use_battery;
scene.add(batteryBar);

// marcador flotante del objetivo actual
const marker = new THREE.Group();
const markerMat = new THREE.MeshStandardMaterial({ color: 0xffc531, emissive: 0xffb300, emissiveIntensity: 1.2 });
const cone = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.32, 4), markerMat);
cone.rotation.x = Math.PI;
marker.add(cone);
const beamMesh = new THREE.Mesh(
  new THREE.CylinderGeometry(0.22, 0.22, 2.4, 24, 1, true),
  new THREE.MeshBasicMaterial({
    color: 0xffc531, transparent: true, opacity: 0.08, depthWrite: false, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  }),
);
beamMesh.position.y = -1.2;
marker.add(beamMesh);
scene.add(marker);

// ---------------------------------------------------------------- particulas y textos

const particles = [];
const particleGeo = new THREE.BoxGeometry(0.05, 0.05, 0.02);

function burst(pos, colors, count, speed, life, gravity = 6) {
  for (let i = 0; i < count; i++) {
    const mat = new THREE.MeshBasicMaterial({ color: colors[i % colors.length], transparent: true });
    const m = new THREE.Mesh(particleGeo, mat);
    m.position.copy(pos);
    const v = new THREE.Vector3(rand() - 0.5, rand() * 0.8 + 0.4, rand() - 0.5).normalize().multiplyScalar(speed * (0.5 + rand()));
    particles.push({ m, v, gravity, spin: new THREE.Vector3(rand(), rand(), rand()).multiplyScalar(10), age: 0, life });
    scene.add(m);
  }
}

function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.age += dt;
    p.v.y -= p.gravity * dt;
    p.m.position.addScaledVector(p.v, dt);
    if (p.m.position.y < 0.02) { p.m.position.y = 0.02; p.v.multiplyScalar(0.4); }
    p.m.rotation.x += p.spin.x * dt;
    p.m.rotation.y += p.spin.y * dt;
    p.m.material.opacity = 1 - p.age / p.life;
    if (p.age >= p.life) {
      scene.remove(p.m);
      p.m.material.dispose();
      particles.splice(i, 1);
    }
  }
}

const popups = [];
function popup(text, cls, worldPos) {
  const el = document.createElement('div');
  el.className = `popup ${cls}`;
  el.textContent = text;
  $('popups').appendChild(el);
  popups.push({ el, pos: worldPos.clone(), age: 0 });
}

function updatePopups(dt) {
  const v = new THREE.Vector3();
  for (let i = popups.length - 1; i >= 0; i--) {
    const p = popups[i];
    p.age += dt;
    v.copy(p.pos).add(new THREE.Vector3(0, p.age * 0.8, 0)).project(camera);
    p.el.style.left = `${(v.x * 0.5 + 0.5) * window.innerWidth}px`;
    p.el.style.top = `${(-v.y * 0.5 + 0.5) * window.innerHeight}px`;
    p.el.style.opacity = String(1 - p.age / 1.4);
    if (p.age > 1.4) { p.el.remove(); popups.splice(i, 1); }
  }
}

// ---------------------------------------------------------------- paquetes del pedido

let pkgs = []; // {mesh, data, mode: shelf|flying|carried|dropping|delivered, rest}

const shelfPkgPos = (shelf, size) => {
  const [r, c] = config.shelf_cells[shelf];
  return cellPos(r, c, 0.1 + size / 2);
};

function zoneSlot(z, size) {
  const zone = zones[z];
  const k = zone.stack++;
  const col = zone.minC + (k % zone.width);
  return cellPos(zone.row, col, size / 2 + Math.floor(k / zone.width) * 0.42);
}

function buildOrder(order) {
  for (const p of pkgs) {
    scene.remove(p.mesh);
    p.mesh.geometry.dispose();
    p.mesh.material.dispose();
  }
  zones.forEach((z) => { z.stack = 0; });
  pkgs = order.map((data) => {
    const size = PKG_SIZES[data.type];
    const mesh = shadowed(new THREE.Mesh(
      new RoundedBoxGeometry(size, size, size, 2, 0.03),
      new THREE.MeshStandardMaterial({
        map: packageTexture(data.type, data.zone),
        roughness: data.type === HEAVY ? 0.35 : 0.55,
        metalness: data.type === HEAVY ? 0.6 : 0,
        emissive: 0xffb300,
        emissiveIntensity: 0,
      }),
    ));
    scene.add(mesh);
    const p = { mesh, data, size, mode: 'shelf', rest: new THREE.Vector3() };
    if (data.status === 'entregado') {
      p.mode = 'delivered';
      p.rest.copy(zoneSlot(data.zone, size));
      mesh.position.copy(p.rest);
    } else if (data.status === 'cargando') {
      p.mode = 'carried';
    }
    return p;
  });
  const used = new Set(order.map((d) => d.shelf));
  racks.forEach((rk, i) => { rk.bottom.visible = !used.has(i); });
}

// ---------------------------------------------------------------- estado visual y animacion

const EYE_COLORS = { normal: 0x00e5ff, carry: 0x3dff8b, bad: 0xff3b30, low: 0xffa000 };

const vis = {
  pos: config.start.slice(),
  heading: Math.PI,
  carrying: false,
  current: 0,
  success: false,
  battery: BAT_MAX,
  armLift: 0,
  alertUntil: 0,
  wheelAngle: 0,
  chargingUntil: 0,
};

const slotWorld = () => robot.slot.getWorldPosition(new THREE.Vector3());

function lerpAngle(a, b, t) {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

const queue = [];
let anim = null;
let stepMs = 200;
let onFrameShown = () => {};

const DURATIONS = {
  reset: 0, move: 1, bump: 1.3, break: 1.8, pick: 2.2, drop: 2.2, pick_fail: 1.3, drop_fail: 1.3,
  wrong_zone: 1.5, charge: 2.5, charge_fail: 1.3, rescue: 4,
};

function durationFor(f) {
  let k = DURATIONS[f.event] ?? 1;
  if (f.event === 'move' && vis.carrying && pkgs[vis.current]?.data.type === HEAVY) k = 1.5;
  const backlog = config.mode === 'manual' && queue.length > 1 ? 0.5 : 1;
  return Math.max(1, stepMs * k * backlog);
}

function beginAnim(f) {
  const a = {
    f, t: 0, dur: durationFor(f),
    from: cellPos(...vis.pos), to: cellPos(...f.pos),
    h0: vis.heading, h1: vis.heading,
    b0: vis.battery, b1: f.battery,
  };
  if (f.action in MOVES) {
    const [dr, dc] = MOVES[f.action];
    a.h1 = headingOf(dr, dc);
    a.dir = new THREE.Vector3(dc, 0, dr);
  }
  switch (f.event) {
    case 'reset':
      buildOrder(f.order);
      Object.assign(vis, { carrying: f.carrying, current: f.current, success: f.success, battery: f.battery });
      a.from = a.to;
      a.h0 = a.h1 = Math.PI;
      a.b0 = a.b1 = f.battery;
      hideBanner();
      break;
    case 'pick': {
      const p = pkgs[f.current];
      const [r, c] = config.shelf_cells[p.data.shelf];
      a.h1 = headingOf(r - f.pos[0], c - f.pos[1]);
      a.pkg = p;
      p.mode = 'flying';
      p.rest.copy(shelfPkgPos(p.data.shelf, p.size));
      break;
    }
    case 'drop': {
      const p = pkgs[f.current - 1];
      a.pkg = p;
      p.mode = 'dropping';
      a.pkgFrom = slotWorld();
      p.rest.copy(zoneSlot(p.data.zone, p.size));
      break;
    }
    case 'break':
      a.pkg = pkgs[f.current];
      break;
    case 'charge':
      vis.chargingUntil = performance.now() + a.dur;
      break;
    case 'rescue':
      rescueCable.visible = true;
      break;
    default:
  }
  if (['bump', 'break', 'pick_fail', 'drop_fail', 'wrong_zone', 'charge_fail', 'rescue'].includes(f.event)) {
    vis.alertUntil = performance.now() + a.dur + 250;
  }
  updateHud(f);
  return a;
}

function applyAnim(a, t) {
  const { f } = a;
  const e = ease(t);
  const p = a.from.clone().lerp(a.to, e);
  if (f.event === 'bump' || f.event === 'break') {
    p.copy(a.from).addScaledVector(a.dir, 0.22 * Math.sin(Math.PI * Math.min(1, t * 1.6)));
  }
  if (f.event === 'rescue') {
    // la grua lo levanta, lo lleva a la estacion de carga y lo baja
    const lift = t < 0.3 ? ease(t / 0.3) : t > 0.75 ? ease((1 - t) / 0.25) : 1;
    p.copy(a.from).lerp(a.to, ease(clamp01((t - 0.3) / 0.45)));
    p.y = lift * 2.2;
    robot.root.rotation.z = Math.sin(t * Math.PI * 6) * 0.08 * lift;
    rescueCable.position.set(p.x, p.y, p.z);
  }
  robot.root.position.copy(p);
  vis.heading = lerpAngle(a.h0, a.h1, clamp01(t / 0.35));
  robot.root.rotation.y = vis.heading;

  if (f.event === 'move' || f.event === 'bump') {
    robot.body.position.y = Math.abs(Math.sin(t * Math.PI * 2)) * 0.02;
  }
  if (['pick_fail', 'drop_fail', 'wrong_zone', 'charge_fail'].includes(f.event)) {
    robot.head.rotation.y = Math.sin(t * Math.PI * 4) * 0.45 * (1 - t);
  }
  if (f.event === 'pick') {
    const u = ease(clamp01((t - 0.3) / 0.7));
    a.pkg.mesh.position.copy(a.pkg.rest).lerp(slotWorld(), u);
    a.pkg.mesh.position.y += Math.sin(Math.PI * u) * 0.35;
    a.pkg.mesh.rotation.y = lerpAngle(0, vis.heading, u);
  }
  if (f.event === 'drop') {
    const u = ease(clamp01((t - 0.15) / 0.85));
    a.pkg.mesh.position.copy(a.pkgFrom).lerp(a.pkg.rest, u);
    a.pkg.mesh.rotation.y = lerpAngle(vis.heading, 0, u);
  }
  // la bateria: al cargar sube poco a poco; en el rescate se llena al aterrizar
  if (f.event === 'charge') vis.battery = a.b0 + (a.b1 - a.b0) * ease(t);
  else if (f.event === 'rescue') vis.battery = t > 0.9 ? a.b1 : Math.max(0, a.b0 - 1);
  else vis.battery = a.b0 + (a.b1 - a.b0) * t;
}

function finishAnim(a) {
  const { f } = a;
  vis.pos = f.pos.slice();
  robot.head.rotation.y = 0;
  robot.body.position.y = 0;
  robot.root.rotation.z = 0;
  robot.root.position.y = 0;
  rescueCable.visible = false;
  Object.assign(vis, { carrying: f.carrying, current: f.current, success: f.success, battery: f.battery });
  const head = cellPos(...f.pos, 1.25);

  switch (f.event) {
    case 'pick':
      a.pkg.mode = 'carried';
      popup(`+5 ${TYPE_LABELS[a.pkg.data.type].toLowerCase()} recogido`, 'gold', head);
      break;
    case 'drop':
      a.pkg.mode = 'delivered';
      a.pkg.mesh.position.copy(a.pkg.rest);
      if (f.success) {
        popup('+35 ¡pedido completo!', 'pos', head);
        burst(cellPos(...f.pos, 0.6), [0x3dff8b, 0xffc531, 0xffffff, 0x2fa8ff, 0xff5a8a], 90, 3.4, 1.8);
      } else {
        popup('+15 entregado', 'pos', head);
        burst(cellPos(...f.pos, 0.6), [ZONE_COLORS[a.pkg.data.zone], 0xffffff], 30, 2.4, 1.2);
      }
      break;
    case 'break': {
      const contact = cellPos(...f.pos, 0.45).addScaledVector(a.dir, 0.5);
      burst(contact, [0xffffff, 0xe8e2d4, 0xd42f2f, 0xc8c8c8], 40, 2.2, 1.2);
      a.pkg.mode = 'shelf'; // hay que volver a buscarlo a la estanteria
      popup(`${f.reward.toFixed(1)} ¡se rompió!`, 'neg', head);
      break;
    }
    case 'bump':
      burst(cellPos(...f.pos, 0.35).addScaledVector(a.dir, 0.5), [0xffb02e, 0xff6a00, 0xfff1a0], 14, 1.6, 0.5);
      popup(f.reward.toFixed(1), 'neg', head);
      break;
    case 'charge':
      popup('⚡ batería llena', 'gold', head);
      break;
    case 'rescue':
      popup(`${f.reward.toFixed(1)} rescate: sin batería`, 'neg', head);
      break;
    case 'wrong_zone':
      popup(`${f.reward.toFixed(1)} zona equivocada`, 'neg', head);
      break;
    default:
      if (f.reward <= -1) popup(f.reward.toFixed(1), 'neg', head);
  }
  // por si en un mismo paso ocurrieron dos cosas (p. ej. se rompio un fragil y se agoto la bateria)
  pkgs.forEach((p, i) => {
    p.data = f.order[i];
    if (p.data.status === 'pendiente') p.mode = 'shelf';
  });
  onFrameShown(f);
}

function batteryColor(frac) {
  return frac > 0.5 ? 0x4fd18b : frac > 0.25 ? 0xffc531 : 0xff5a4e;
}

function updateRobot(dt, now) {
  const target = vis.carrying || anim?.f.event === 'pick' ? 1 : 0;
  vis.armLift += (target - vis.armLift) * Math.min(1, dt * 8);
  for (const arm of robot.arms) {
    arm.shoulder.rotation.x = -0.15 - vis.armLift * 1.25 + Math.sin(now / 600 + arm.side) * 0.03 * (1 - vis.armLift);
    arm.shoulder.rotation.z = arm.side * (0.12 - vis.armLift * 0.16);
    arm.elbow.rotation.x = -0.25 - vis.armLift * 0.35;
  }
  const moving = anim && ['move', 'bump', 'break'].includes(anim.f.event);
  if (moving) vis.wheelAngle += dt * (1000 / anim.dur) * 10;
  robot.wheels.forEach((w) => { w.rotation.x = vis.wheelAngle; });
  if (!anim) robot.head.position.y = 0.79 + Math.sin(now / 500) * 0.008;

  const frac = vis.battery / BAT_MAX;
  const low = config.use_battery && frac <= 0.25;
  const mood = now < vis.alertUntil ? 'bad' : vis.carrying ? 'carry' : low ? 'low' : 'normal';
  const color = EYE_COLORS[mood];
  robot.eyeMat.color.setHex(color);
  robot.eyeMat.emissive.setHex(color);
  robot.chestMat.emissive.setHex(color);
  robot.light.color.setHex(color);
  const blink = (now % 3200) < 120 ? 0.15 : 1;
  robot.eyes.forEach((e) => { e.scale.y = 1.15 * blink; });
  const period = low ? 300 : 1000;
  robot.tipMat.emissiveIntensity = (now % period) < period / 2 ? 2.5 : 0.3;

  // barra de bateria
  batteryBar.position.copy(robot.root.position).add(new THREE.Vector3(0, 1.28, 0));
  batteryBar.quaternion.copy(camera.quaternion);
  barFill.scale.x = Math.max(0.001, frac);
  barFill.position.x = -0.28 * (1 - frac);
  barFillMat.color.setHex(batteryColor(frac));
}

function updateWorld(now) {
  // paquetes que no se estan animando
  pkgs.forEach((p, i) => {
    if (p.mode === 'shelf') {
      p.mesh.position.copy(shelfPkgPos(p.data.shelf, p.size));
      p.mesh.rotation.y = 0;
    } else if (p.mode === 'carried') {
      p.mesh.position.copy(slotWorld());
      p.mesh.rotation.y = robot.root.rotation.y;
    }
    const isTarget = i === vis.current && p.mode === 'shelf';
    p.mesh.material.emissiveIntensity = isTarget ? 0.15 + 0.15 * Math.sin(now / 250) : 0;
  });

  // el marcador indica el siguiente objetivo: el paquete o su zona de entrega
  const cur = pkgs[vis.current];
  let goal = null;
  let color = 0xffc531;
  if (cur && !vis.success) {
    if (vis.carrying) {
      goal = zones[cur.data.zone].center.clone().setY(1.95);
      color = ZONE_COLORS[cur.data.zone];
    } else {
      goal = shelfPkgPos(cur.data.shelf, cur.size).setY(RACK_H + 0.55);
    }
  }
  marker.visible = goal !== null;
  if (goal) {
    if (marker.position.lengthSq() === 0) marker.position.copy(goal);
    marker.position.lerp(goal, 0.08);
    cone.position.y = Math.sin(now / 300) * 0.08;
    marker.rotation.y = now / 700;
    markerMat.color.setHex(color);
    markerMat.emissive.setHex(color);
    beamMesh.material.color.setHex(color);
  }

  zones.forEach((z, i) => {
    const active = vis.carrying && cur && cur.data.zone === i;
    z.light.intensity = active ? 4 + Math.sin(now / 200) * 1.5 : 0;
    z.mat.emissiveIntensity = active ? 0.35 + Math.sin(now / 200) * 0.15 : 0.08;
    z.lampMat.emissiveIntensity = active ? 2 : 0.5;
  });

  // la estacion de carga brilla al cargar y parpadea cuando la bateria esta baja
  const charging = now < vis.chargingUntil;
  const low = config.use_battery && vis.battery / BAT_MAX <= 0.25;
  chargerLight.intensity = charging ? 5 + Math.sin(now / 60) * 2 : low ? 2 + Math.sin(now / 150) * 1.5 : 0;
  chargerPadMat.emissiveIntensity = charging ? 0.6 : low ? 0.2 + Math.sin(now / 150) * 0.15 : 0.05;
  screenMat.emissiveIntensity = charging ? 2 : 0.6;
  if (charging && Math.random() < 0.35) {
    burst(robot.root.position.clone().add(new THREE.Vector3(0, 0.6, 0)), [0xffd23f, 0xfff6b0, 0x3dff8b], 2, 1.2, 0.6, 1);
  }
}

// ---------------------------------------------------------------- camara

let followRobot = false;
const camPresets = {
  iso: { pos: [0, 14, 10.5], target: [0, 0, 0.6] },
  top: { pos: [0, 19, 0.01], target: [0, 0, 0] },
  side: { pos: [10, 9, 10], target: [0, 0, 0.5] },
};
let camTween = null;

function setCamera(name) {
  followRobot = name === 'follow';
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('active', b.dataset.cam === name));
  const preset = followRobot
    ? { pos: robot.root.position.clone().add(new THREE.Vector3(0, 4.5, 5)), target: robot.root.position.clone() }
    : { pos: new THREE.Vector3(...camPresets[name].pos), target: new THREE.Vector3(...camPresets[name].target) };
  camTween = { t: 0, p0: camera.position.clone(), t0: controls.target.clone(), p1: preset.pos, t1: preset.target };
}

document.querySelectorAll('[data-cam]').forEach((b) => b.addEventListener('click', () => setCamera(b.dataset.cam)));
controls.addEventListener('start', () => { camTween = null; });

function updateCamera(dt) {
  if (camTween) {
    camTween.t = Math.min(1, camTween.t + dt * 1.6);
    const e = ease(camTween.t);
    camera.position.lerpVectors(camTween.p0, camTween.p1, e);
    controls.target.lerpVectors(camTween.t0, camTween.t1, e);
    if (camTween.t >= 1) camTween = null;
  } else if (followRobot) {
    const delta = robot.root.position.clone().sub(controls.target).multiplyScalar(Math.min(1, dt * 4));
    controls.target.add(delta);
    camera.position.add(delta);
  }
  controls.update();
}

// direccion de la tecla segun hacia donde mira la camara
function cameraRelativeAction(dir) {
  const fwd = new THREE.Vector3();
  camera.getWorldDirection(fwd);
  fwd.y = 0;
  if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
  fwd.normalize();
  const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
  const d = { up: fwd, down: fwd.clone().negate(), right, left: right.clone().negate() }[dir];
  if (Math.abs(d.x) > Math.abs(d.z)) return d.x > 0 ? RIGHT : LEFT;
  return d.z > 0 ? DOWN : UP;
}

// ---------------------------------------------------------------- HUD

function fmt(x) { return (x >= 0 ? '+' : '') + x.toFixed(1); }

const STATUS_LABELS = { pendiente: 'pendiente', cargando: 'en camino', entregado: '✔ entregado' };

function renderOrder(f) {
  $('order').innerHTML = f.order.map((p, i) => {
    const cls = p.status === 'entregado' ? 'done' : i === f.current ? 'current' : '';
    const broken = p.broken ? ` <span class="broken">roto ×${p.broken}</span>` : '';
    return `<li class="${cls}"><span class="num">${i + 1}.</span>`
      + `<span class="type">${TYPE_LABELS[p.type]}${broken}</span>`
      + `<span class="zone" style="background:${ZONE_CSS[p.zone]}">${ZONE_NAMES[p.zone]}</span>`
      + `<span class="st">${STATUS_LABELS[p.status]}</span></li>`;
  }).join('');
}

function renderBattery(value) {
  const frac = value / BAT_MAX;
  $('hud-battery').textContent = `${Math.round(frac * 100)}%`;
  $('bat-fill').style.width = `${frac * 100}%`;
  $('bat-fill').style.background = `#${batteryColor(frac).toString(16).padStart(6, '0')}`;
}

function updateHud(f) {
  $('hud-step').textContent = `${f.steps} / ${config.max_steps}`;
  $('hud-action').textContent = f.action === null ? '-' : ACTION_NAMES[f.action];
  const rw = $('hud-reward');
  rw.textContent = fmt(f.reward);
  rw.className = f.reward > 0 ? 'pos' : f.reward <= -1 ? 'neg' : '';
  $('hud-total').textContent = fmt(f.total);
  $('hud-rescues').textContent = f.rescues;
  $('hud-rescues').className = f.rescues ? 'neg' : '';
  renderOrder(f);
  renderBattery(f.battery);
  if (config.mode === 'manual') {
    const cur = f.order[Math.min(f.current, f.order.length - 1)];
    const zone = `<b style="color:${ZONE_CSS[cur.zone]}">Zona ${ZONE_NAMES[cur.zone]}</b>`;
    let goal = f.carrying
      ? `Lleva el paquete a la ${zone} y pulsa <kbd>E</kbd>.`
      : `Ve junto a la estantería marcada y pulsa <kbd>R</kbd> para recoger el paquete ${f.current + 1}.`;
    if (f.carrying && cur.type === HEAVY) goal += '<br>Pesado: cada paso gasta el doble de batería.';
    if (f.carrying && cur.type === FRAGILE) goal += '<br><b class="neg">¡Frágil!</b> Si chocas, se rompe.';
    if (config.use_battery && f.battery / BAT_MAX <= 0.3) {
      goal += '<br><b class="gold">Batería baja:</b> ve a la estación ⚡ y pulsa <kbd>C</kbd>.';
    }
    if (f.success) goal = '¡Pedido completo! Pulsa <kbd>N</kbd> para otro.';
    $('goal').innerHTML = goal;
  }
}

let bannerTimer = null;
function showBanner(title, sub, cls, ms) {
  clearTimeout(bannerTimer);
  $('banner').hidden = false;
  $('banner-title').textContent = title;
  $('banner-title').className = cls;
  $('banner-sub').textContent = sub;
  if (ms) bannerTimer = setTimeout(hideBanner, ms);
}
function hideBanner() { $('banner').hidden = true; }

if (!config.use_battery) $('battery-block').hidden = true;

// ---------------------------------------------------------------- bucle principal

const clock = new THREE.Clock();
function loop() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const now = performance.now();
  if (anim) {
    anim.t = Math.min(1, anim.t + (dt * 1000) / anim.dur);
    applyAnim(anim, anim.t);
    if (anim.t >= 1) { finishAnim(anim); anim = null; }
  }
  if (!anim && queue.length) {
    anim = beginAnim(queue.shift());
    applyAnim(anim, 0);
  }
  if (anim) renderBattery(vis.battery);
  updateRobot(dt, now);
  updateWorld(now);
  updateParticles(dt);
  updatePopups(dt);
  updateCamera(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

async function waitIdle() {
  while (anim || queue.length) await sleep(40);
}

robot.root.position.copy(cellPos(...vis.pos));
robot.root.rotation.y = vis.heading;
setCamera(new URLSearchParams(location.search).get('cam') || 'iso');
loop();

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '1') setCamera('iso');
  if (e.key === '2') setCamera('top');
  if (e.key === '3') setCamera('follow');
  if (e.key === '4') setCamera('side');
});

function resultText(f) {
  const delivered = f.order.filter((p) => p.status === 'entregado').length;
  return `${delivered}/${f.order.length} paquetes · ${f.steps} pasos · recompensa ${fmt(f.total)}`
    + (f.rescues ? ` · ${f.rescues} rescate${f.rescues > 1 ? 's' : ''}` : '');
}

// ---------------------------------------------------------------- modo manual

async function startManual() {
  $('mode-tag').textContent = 'MANUAL';
  $('phase').textContent = 'Control manual';
  $('controls').hidden = false;
  stepMs = 190;

  let ended = false;
  let lastKey = '';
  let chain = Promise.resolve();
  const send = (path, body) => {
    chain = chain.then(() => api.post(path, body)).then((f) => {
      const key = `${f.episode}:${f.steps}`;
      if (key === lastKey) return; // el episodio ya termino: el servidor repite el ultimo fotograma
      lastKey = key;
      if (f.event === 'reset') { queue.length = 0; ended = false; }
      if (f.terminated || f.truncated) ended = true;
      $('episode-title').textContent = `Pedido ${f.episode}`;
      queue.push(f);
    }).catch((err) => showError(`Error del servidor: ${err.message}`));
  };
  onFrameShown = (f) => {
    if (f.terminated) showBanner('¡PEDIDO COMPLETO!', `${resultText(f)} · pulsa N para otro`, 'pos');
    else if (f.truncated) showBanner('TIEMPO AGOTADO', `${resultText(f)} · pulsa N para otro`, 'neg');
  };

  const f = await api.get('/api/state');
  lastKey = `${f.episode}:${f.steps}`;
  ended = f.terminated || f.truncated;
  $('episode-title').textContent = `Pedido ${f.episode}`;
  queue.push({ ...f, event: 'reset' });

  const keyMap = {
    arrowup: 'up', w: 'up', arrowdown: 'down', s: 'down',
    arrowleft: 'left', a: 'left', arrowright: 'right', d: 'right',
  };
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    let action = null;
    if (k in keyMap) action = cameraRelativeAction(keyMap[k]);
    else if (k === 'r' || k === ' ') action = PICK;
    else if (k === 'e' || k === 'enter') action = DROP;
    else if (k === 'c') action = CHARGE;
    else if (k === 'n') { e.preventDefault(); send('/api/reset'); return; }
    if (action === null) return;
    e.preventDefault();
    if (ended || queue.length >= 3) return;
    send('/api/step', { action });
  });
}

// ---------------------------------------------------------------- modo automatico (Q-learning)

function drawChart(history) {
  const cv = $('chart');
  const ctx = cv.getContext('2d');
  const w = cv.width;
  const h = cv.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (const y of [0.25, 0.5, 0.75]) {
    ctx.beginPath(); ctx.moveTo(0, h * y); ctx.lineTo(w, h * y); ctx.stroke();
  }
  if (!history.length) return;
  const pad = 12;
  const xs = (i) => pad + (history.length === 1 ? 0 : (i * (w - 2 * pad)) / (history.length - 1));
  const ys = (v) => h - pad - v * (h - 2 * pad);
  const maxRescues = Math.max(1, ...history.map((p) => p.rescues));
  const series = [
    { color: '#ff8a2a', value: (p) => p.rescues / maxRescues },
    { color: '#4fd18b', value: (p) => p.success_rate },
  ];
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 4;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    history.forEach((p, i) => (i ? ctx.lineTo(xs(i), ys(s.value(p))) : ctx.moveTo(xs(i), ys(s.value(p)))));
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(xs(history.length - 1), ys(s.value(history[history.length - 1])), 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

const PHASES = {
  esperando: 'Preparando...',
  entrenamiento: 'Entrenando con Q-learning',
  demo: 'Demostración: agente entrenado (sin exploración)',
  terminado: 'Terminado',
};

function updateStatus(s) {
  $('phase').textContent = PHASES[s.phase] || s.phase;
  const pct = s.episodes ? (100 * s.episode) / s.episodes : 0;
  $('progress-bar').style.width = `${s.phase === 'entrenamiento' ? pct : s.episodes ? 100 : 0}%`;
  $('train-progress').textContent = s.episodes ? `${s.episode.toLocaleString()} / ${s.episodes.toLocaleString()}` : '-';
  $('train-eps').textContent = s.epsilon.toFixed(3);
  drawChart(s.history);
  $('history').innerHTML = s.history.slice(-5).reverse().map((p) =>
    `<li>ep ${p.episode.toLocaleString()} · completos ${(p.success_rate * 100).toFixed(1)}% · `
    + `rescates ${p.rescues.toFixed(2)} · R̄ ${p.mean_reward.toFixed(1)}</li>`).join('');
}

const speedToMs = (v) => Math.round(600 * Math.pow(0.04, v / 100)); // 600 ms .. 24 ms por paso

async function startAuto() {
  $('mode-tag').textContent = 'Q-LEARNING';
  $('auto-block').hidden = false;
  $('speed-block').hidden = false;
  $('goal').textContent = 'El robot aprende por prueba y error a completar pedidos: recoger cada paquete, '
    + 'llevarlo a la zona de su color, no chocar con los frágiles y cargar la batería a tiempo.';
  const speed = $('speed');
  const applySpeed = () => { stepMs = speedToMs(Number(speed.value)); };
  speed.addEventListener('input', applySpeed);
  applySpeed();
  let skip = false;
  $('skip').addEventListener('click', () => { skip = true; });

  // latido: mantiene vivo el entrenamiento mientras la pagina esta abierta
  setInterval(() => api.get('/api/status').then(updateStatus).catch(() => {}), 700);

  let after = -1;
  for (;;) {
    let res;
    try {
      res = await api.get(`/api/next?after=${after}`);
      $('error').hidden = true;
    } catch (err) {
      showError('Se perdió la conexión con Python (¿terminó train3d.py?).');
      await sleep(1500);
      continue;
    }
    updateStatus(res.status);
    const ep = res.episode;
    if (!ep) { await sleep(250); continue; }

    const train = ep.kind === 'train';
    $('episode-title').textContent = train
      ? `Episodio ${ep.number.toLocaleString()} / ${ep.of.toLocaleString()} · ε = ${ep.epsilon.toFixed(3)}`
      : `Demo ${ep.number} / ${ep.of}`;
    hideBanner();
    skip = false;
    queue.push(...ep.frames);
    while ((anim || queue.length) && !skip) await sleep(40);
    if (skip) { queue.length = 0; await waitIdle(); }
    const last = ep.frames[ep.frames.length - 1];
    showBanner(ep.success ? '¡PEDIDO COMPLETO!' : 'PEDIDO INCOMPLETO', resultText(last), ep.success ? 'pos' : 'neg', 0);
    await sleep(Math.max(700, stepMs * 6));
    after = ep.id;
  }
}

if (config.mode === 'manual') startManual();
else startAuto();
