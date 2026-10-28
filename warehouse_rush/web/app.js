// Visualizacion 3D de Warehouse Rush con three.js.
// Python ejecuta el entorno; esta pagina recibe fotogramas por la API y los anima.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const ACTION_NAMES = ['arriba', 'abajo', 'izquierda', 'derecha', 'recoger', 'entregar'];
const [UP, DOWN, LEFT, RIGHT, PICK, DROP] = [0, 1, 2, 3, 4, 5];
const MOVES = { [UP]: [-1, 0], [DOWN]: [1, 0], [LEFT]: [0, -1], [RIGHT]: [0, 1] };

const WALL_H = 0.7;
const RACK_H = 1.5;
const PKG_SIZE = 0.38;

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
  ctx.fillStyle = '#1b1b1b';
  for (let x = -h; x < w + h; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, h); ctx.lineTo(x + 16, h); ctx.lineTo(x + 16 + h, 0); ctx.lineTo(x + h, 0);
    ctx.fill();
  }
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

const packageTex = canvasTexture(128, 128, (ctx, w, h) => {
  ctx.fillStyle = '#ffc21a';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#d63a2f';
  ctx.fillRect(w * 0.4, 0, w * 0.2, h);
  ctx.fillRect(0, h * 0.4, w, h * 0.2);
  ctx.strokeStyle = '#8a5b00';
  ctx.lineWidth = 6;
  ctx.strokeRect(0, 0, w, h);
});

// ---------------------------------------------------------------- bodega

const shadowed = (m) => { m.castShadow = true; m.receiveShadow = true; return m; };

// suelo exterior
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(120, 120),
  new THREE.MeshStandardMaterial({ color: 0x1b2028, roughness: 1 }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.01;
ground.receiveShadow = true;
scene.add(ground);

// suelo interior de concreto
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

// zona de entregas
const dCells = config.delivery_cells;
const dCols = dCells.map(([, c]) => c);
const dMin = Math.min(...dCols);
const dMax = Math.max(...dCols);
const dRow = dCells[0][0];
const deliveryCenter = cellPos(dRow, (dMin + dMax) / 2);
const dWidth = dMax - dMin + 1;
const deliveryTex = canvasTexture(256 * dWidth, 256, (ctx, w, h) => {
  ctx.fillStyle = '#17844a';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(255,255,255,0.10)';
  for (let x = -h; x < w; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, h); ctx.lineTo(x + 22, h); ctx.lineTo(x + 22 + h, 0); ctx.lineTo(x + h, 0);
    ctx.fill();
  }
  ctx.setLineDash([22, 14]);
  ctx.strokeStyle = '#f4f7f2';
  ctx.lineWidth = 10;
  ctx.strokeRect(10, 10, w - 20, h - 20);
  ctx.fillStyle = '#f4f7f2';
  ctx.font = 'bold 72px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('ENTREGAS', w / 2, h / 2);
});
const deliveryMat = new THREE.MeshStandardMaterial({
  map: deliveryTex, emissive: 0x2bd97a, emissiveIntensity: 0.15, roughness: 0.6,
});
const deliveryPad = new THREE.Mesh(new THREE.PlaneGeometry(dWidth, 1), deliveryMat);
deliveryPad.rotation.x = -Math.PI / 2;
deliveryPad.position.set(deliveryCenter.x, 0.012, deliveryCenter.z);
deliveryPad.receiveShadow = true;
scene.add(deliveryPad);

// portico sobre la zona de entregas
const gantryMat = new THREE.MeshStandardMaterial({ color: 0x2c8f57, roughness: 0.4, metalness: 0.6 });
for (const s of [-1, 1]) {
  const post = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.5, 0.08), gantryMat));
  post.position.set(deliveryCenter.x + s * (dWidth / 2 + 0.02), 0.75, deliveryCenter.z + 0.45);
  scene.add(post);
}
const beam = shadowed(new THREE.Mesh(new THREE.BoxGeometry(dWidth + 0.12, 0.12, 0.12), gantryMat));
beam.position.set(deliveryCenter.x, 1.5, deliveryCenter.z + 0.45);
scene.add(beam);
const lampMat = new THREE.MeshStandardMaterial({ color: 0x9dffc7, emissive: 0x3dff95, emissiveIntensity: 1 });
const lamp = new THREE.Mesh(new THREE.BoxGeometry(dWidth - 0.4, 0.04, 0.08), lampMat);
lamp.position.set(deliveryCenter.x, 1.43, deliveryCenter.z + 0.45);
scene.add(lamp);
const deliveryLight = new THREE.PointLight(0x47ff9b, 0, 4, 2);
deliveryLight.position.set(deliveryCenter.x, 1.2, deliveryCenter.z);
scene.add(deliveryLight);

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
  const bottom = new THREE.Group(); // cajas del nivel inferior: se ocultan donde esta el paquete
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
  const tip = add(head, new THREE.SphereGeometry(0.03, 12, 10), tipMat, 0.1, 0.31, -0.05);

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

// paquete objetivo y marcador flotante
const pkg = shadowed(new THREE.Mesh(
  new RoundedBoxGeometry(PKG_SIZE, PKG_SIZE, PKG_SIZE, 2, 0.03),
  new THREE.MeshStandardMaterial({ map: packageTex, roughness: 0.5, emissive: 0xffb300, emissiveIntensity: 0.12 }),
));
scene.add(pkg);

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

// ---------------------------------------------------------------- particulas

const particles = [];
const particleGeo = new THREE.BoxGeometry(0.05, 0.05, 0.02);

function burst(pos, colors, count, speed, life) {
  for (let i = 0; i < count; i++) {
    const mat = new THREE.MeshBasicMaterial({ color: colors[i % colors.length], transparent: true });
    const m = new THREE.Mesh(particleGeo, mat);
    m.position.copy(pos);
    const v = new THREE.Vector3(rand() - 0.5, rand() * 0.8 + 0.4, rand() - 0.5).normalize().multiplyScalar(speed * (0.5 + rand()));
    particles.push({ m, v, spin: new THREE.Vector3(rand(), rand(), rand()).multiplyScalar(10), age: 0, life });
    scene.add(m);
  }
}

function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.age += dt;
    p.v.y -= 6 * dt;
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

// textos flotantes (+10, -1...)
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
    p.el.style.opacity = String(1 - p.age / 1.2);
    if (p.age > 1.2) { p.el.remove(); popups.splice(i, 1); }
  }
}

// ---------------------------------------------------------------- estado visual y animacion

const EYE_COLORS = { normal: 0x00e5ff, carry: 0x3dff8b, bad: 0xff3b30 };

const vis = {
  pos: config.start.slice(),
  heading: Math.PI,
  carrying: false,
  delivered: false,
  itemIdx: 0,
  armLift: 0,
  alertUntil: 0,
  wheelAngle: 0,
};
const pkgState = { mode: 'shelf', from: new THREE.Vector3(), to: new THREE.Vector3() };

const shelfPkgPos = (idx) => {
  const [r, c] = config.shelf_cells[idx];
  return cellPos(r, c, 0.1 + PKG_SIZE / 2);
};
const slotWorld = () => robot.slot.getWorldPosition(new THREE.Vector3());

function setItem(idx) {
  vis.itemIdx = idx;
  racks.forEach((rk, i) => { rk.bottom.visible = i !== idx; });
}

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

function durationFor(event) {
  const k = { reset: 0, move: 1, bump: 1.3, pick: 2.2, drop: 2.2, pick_fail: 1.3, drop_fail: 1.3 }[event];
  const backlog = config.mode === 'manual' && queue.length > 1 ? 0.5 : 1;
  return Math.max(1, stepMs * k * backlog);
}

function beginAnim(f) {
  const a = {
    f, t: 0, dur: durationFor(f.event),
    from: cellPos(...vis.pos), to: cellPos(...f.pos),
    h0: vis.heading, h1: vis.heading,
  };
  if (f.action in MOVES) {
    const [dr, dc] = MOVES[f.action];
    a.h1 = headingOf(dr, dc);
    a.dir = new THREE.Vector3(dc, 0, dr);
  }
  if (f.event === 'reset') {
    vis.carrying = false;
    vis.delivered = false;
    setItem(f.item_idx);
    pkgState.mode = 'shelf';
    a.from = a.to;
    a.h0 = a.h1 = Math.PI;
    hideBanner();
  } else if (f.event === 'pick') {
    const [r, c] = config.shelf_cells[f.item_idx];
    a.h1 = headingOf(r - f.pos[0], c - f.pos[1]);
    pkgState.mode = 'flying';
    pkgState.from.copy(shelfPkgPos(f.item_idx));
  } else if (f.event === 'drop') {
    pkgState.mode = 'dropping';
    pkgState.from.copy(slotWorld());
    pkgState.to.copy(cellPos(...f.pos, PKG_SIZE / 2));
  }
  if (['bump', 'pick_fail', 'drop_fail'].includes(f.event)) vis.alertUntil = performance.now() + a.dur + 250;
  updateHud(f);
  return a;
}

function applyAnim(a, t) {
  const { f } = a;
  const e = ease(t);
  const p = a.from.clone().lerp(a.to, e);
  if (f.event === 'bump') p.addScaledVector(a.dir, 0.22 * Math.sin(Math.PI * Math.min(1, t * 1.6)));
  robot.root.position.copy(p);
  vis.heading = lerpAngle(a.h0, a.h1, clamp01(t / 0.35));
  robot.root.rotation.y = vis.heading;

  if (f.event === 'move' || f.event === 'bump') {
    robot.body.position.y = Math.abs(Math.sin(t * Math.PI * 2)) * 0.02;
  }
  if (f.event === 'pick_fail' || f.event === 'drop_fail') {
    robot.head.rotation.y = Math.sin(t * Math.PI * 4) * 0.45 * (1 - t);
  }
  if (f.event === 'pick') {
    const u = ease(clamp01((t - 0.3) / 0.7));
    pkg.position.copy(pkgState.from).lerp(slotWorld(), u);
    pkg.position.y += Math.sin(Math.PI * u) * 0.35;
    pkg.rotation.y = lerpAngle(0, vis.heading, u);
  }
  if (f.event === 'drop') {
    const u = ease(clamp01((t - 0.15) / 0.85));
    pkg.position.copy(pkgState.from).lerp(pkgState.to, u);
  }
}

function finishAnim(a) {
  const { f } = a;
  vis.pos = f.pos.slice();
  robot.head.rotation.y = 0;
  robot.body.position.y = 0;
  vis.carrying = f.carrying;
  vis.delivered = f.delivered;
  if (f.event === 'pick') pkgState.mode = 'carried';
  if (f.event === 'drop') {
    pkgState.mode = 'delivered';
    burst(cellPos(...f.pos, 0.6), [0x3dff8b, 0xffc531, 0xffffff, 0x2fa8ff, 0xff5a8a], 70, 3.2, 1.6);
  }
  if (f.event === 'bump') {
    const contact = cellPos(...f.pos, 0.35).addScaledVector(a.dir, 0.5);
    burst(contact, [0xffb02e, 0xff6a00, 0xfff1a0], 14, 1.6, 0.5);
  }
  if (f.event !== 'reset') {
    const head = cellPos(...f.pos, 1.2);
    if (f.event === 'pick') popup('+10 ¡recogido!', 'gold', head);
    else if (f.event === 'drop') popup('+20 ¡entregado!', 'pos', head);
    else if (f.reward <= -1) popup(f.reward.toFixed(1), 'neg', head);
  }
  onFrameShown(f);
}

function updateRobot(dt, now) {
  const target = vis.carrying || pkgState.mode === 'flying' ? 1 : 0;
  vis.armLift += (target - vis.armLift) * Math.min(1, dt * 8);
  for (const arm of robot.arms) {
    arm.shoulder.rotation.x = -0.15 - vis.armLift * 1.25 + Math.sin(now / 600 + arm.side) * 0.03 * (1 - vis.armLift);
    arm.shoulder.rotation.z = arm.side * (0.12 - vis.armLift * 0.16);
    arm.elbow.rotation.x = -0.25 - vis.armLift * 0.35;
  }
  // ruedas segun la distancia recorrida
  const moving = anim && (anim.f.event === 'move' || anim.f.event === 'bump');
  if (moving) vis.wheelAngle += dt * (1000 / anim.dur) * 10;
  robot.wheels.forEach((w) => { w.rotation.x = vis.wheelAngle; });
  if (!moving && !anim) robot.head.position.y = 0.79 + Math.sin(now / 500) * 0.008;

  const mood = now < vis.alertUntil ? 'bad' : vis.carrying ? 'carry' : 'normal';
  const color = EYE_COLORS[mood];
  robot.eyeMat.color.setHex(color);
  robot.eyeMat.emissive.setHex(color);
  robot.chestMat.emissive.setHex(color);
  robot.light.color.setHex(color);
  const blink = (now % 3200) < 120 ? 0.15 : 1;
  robot.eyes.forEach((e) => { e.scale.y = 1.15 * blink; });
  robot.tipMat.emissiveIntensity = (now % 1000) < 500 ? 2.5 : 0.3;
}

function updatePackage(now) {
  pkg.visible = true;
  if (pkgState.mode === 'shelf') {
    pkg.position.copy(shelfPkgPos(vis.itemIdx));
    pkg.rotation.y = 0;
  } else if (pkgState.mode === 'carried') {
    pkg.position.copy(slotWorld());
    pkg.rotation.y = robot.root.rotation.y;
  }
  pkg.material.emissiveIntensity = pkgState.mode === 'shelf' ? 0.12 + 0.12 * Math.sin(now / 250) : 0.08;

  // el marcador indica el siguiente objetivo: el paquete o la zona de entregas
  let goal = null;
  if (pkgState.mode === 'shelf') goal = shelfPkgPos(vis.itemIdx).setY(RACK_H + 0.55);
  else if (pkgState.mode === 'carried' || pkgState.mode === 'flying') goal = deliveryCenter.clone().setY(1.95);
  marker.visible = goal !== null;
  if (goal) {
    if (marker.position.lengthSq() === 0) marker.position.copy(goal);
    marker.position.lerp(goal, 0.08);
    cone.position.y = Math.sin(now / 300) * 0.08;
    marker.rotation.y = now / 700;
    const toDelivery = pkgState.mode !== 'shelf';
    markerMat.color.setHex(toDelivery ? 0x3dff8b : 0xffc531);
    markerMat.emissive.setHex(toDelivery ? 0x2bd97a : 0xffb300);
    beamMesh.material.color.setHex(toDelivery ? 0x3dff8b : 0xffc531);
  }
  const active = pkgState.mode === 'carried';
  deliveryLight.intensity = active ? 4 + Math.sin(now / 200) * 1.5 : 0;
  deliveryMat.emissiveIntensity = active ? 0.35 + Math.sin(now / 200) * 0.15 : 0.12;
}

// ---------------------------------------------------------------- camara

let followRobot = false;
const camPresets = {
  iso: { pos: [0, 14, 10.5], target: [0, 0, 0.6] },
  side: { pos: [10, 9, 10], target: [0, 0, 0.5] },
  top: { pos: [0, 19, 0.01], target: [0, 0, 0] },
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

function updateHud(f) {
  $('hud-step').textContent = `${f.steps} / ${config.max_steps}`;
  $('hud-action').textContent = f.action === null ? '-' : ACTION_NAMES[f.action];
  const rw = $('hud-reward');
  rw.textContent = fmt(f.reward);
  rw.className = f.reward > 0 ? 'pos' : f.reward <= -1 ? 'neg' : '';
  $('hud-total').textContent = fmt(f.total);
  const st = $('hud-state');
  st.textContent = f.delivered ? 'entregado' : f.carrying ? 'CARGANDO' : 'sin carga';
  st.className = f.delivered ? 'pos' : f.carrying ? 'gold' : '';
  if (config.mode === 'manual') {
    $('goal').innerHTML = f.carrying
      ? 'Lleva el paquete a la <b class="pos">zona verde</b> y pulsa <kbd>E</kbd>.'
      : 'Ve junto a la estantería con el <b class="gold">paquete dorado</b> y pulsa <kbd>R</kbd>.';
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
  updateRobot(dt, now);
  updatePackage(now);
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
setItem(0);
setCamera(new URLSearchParams(location.search).get('cam') || 'iso');
loop();

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '1') setCamera('iso');
  if (e.key === '2') setCamera('top');
  if (e.key === '3') setCamera('follow');
  if (e.key === '4') setCamera('side');
});

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
      $('episode-title').textContent = `Episodio ${f.episode}`;
      queue.push(f);
    }).catch((err) => showError(`Error del servidor: ${err.message}`));
  };
  onFrameShown = (f) => {
    if (f.terminated) showBanner('¡ENTREGADO!', `${f.steps} pasos · recompensa ${fmt(f.total)} · pulsa N para otro episodio`, 'pos');
    else if (f.truncated) showBanner('TIEMPO AGOTADO', 'Pulsa N para un nuevo episodio', 'neg');
  };

  const f = await api.get('/api/state');
  lastKey = `${f.episode}:${f.steps}`;
  ended = f.terminated || f.truncated;
  $('episode-title').textContent = `Episodio ${f.episode}`;
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
  ctx.strokeStyle = '#4fd18b';
  ctx.lineWidth = 4;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  history.forEach((p, i) => (i ? ctx.lineTo(xs(i), ys(p.delivery_rate)) : ctx.moveTo(xs(i), ys(p.delivery_rate))));
  ctx.stroke();
  ctx.fillStyle = '#4fd18b';
  const last = history[history.length - 1];
  ctx.beginPath();
  ctx.arc(xs(history.length - 1), ys(last.delivery_rate), 7, 0, Math.PI * 2);
  ctx.fill();
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
    `<li>ep ${p.episode.toLocaleString()} · entregas ${(p.delivery_rate * 100).toFixed(1)}% · R̄ ${p.mean_reward.toFixed(1)}</li>`).join('');
}

const speedToMs = (v) => Math.round(600 * Math.pow(0.04, v / 100)); // 600 ms .. 24 ms por paso

async function startAuto() {
  $('mode-tag').textContent = 'Q-LEARNING';
  $('auto-block').hidden = false;
  $('speed-block').hidden = false;
  $('goal').textContent = 'El robot aprende por prueba y error. Al principio explora al azar (ε alto); '
    + 'con el tiempo aprende a recoger el paquete dorado y llevarlo a la zona verde.';
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
    showBanner(ep.delivered ? '¡ENTREGADO!' : 'NO ENTREGADO',
      `${ep.steps} pasos · recompensa ${fmt(ep.total)}`, ep.delivered ? 'pos' : 'neg', 0);
    await sleep(Math.max(700, stepMs * 6));
    after = ep.id;
  }
}

if (config.mode === 'manual') startManual();
else startAuto();
