// Aplicación web: catálogo de experimentos y modelos, simulación 3D y evaluación contra la API.
import './style.css';
import { api, ApiError } from './api';
import { Escena, TYPE_LABELS, ZONE_CSS, batteryColor } from './escena';
import type { Ambiente, Catalogo, Evaluacion, Fotograma, ModeloCatalogo, ModeloInfo, Nivel, Niveles, Run } from './tipos';

const [PICK, DROP, CHARGE] = [4, 5, 6];
const GITHUB = 'https://github.com/c851458/pro_final_visu2';
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const pct = (x: number | undefined) => (x === undefined || x === null ? '-' : `${(x * 100).toFixed(1)}%`);
const num = (x: number | undefined, d = 1) => (x === undefined || x === null ? '-' : x.toFixed(d));
const fmt = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(1);
const short = (s: string | null | undefined, n = 7) => (s ? s.slice(0, n) : '-');
const esc = (s: unknown) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

type Politica = 'manual' | 'aleatoria' | 'regla' | 'modelo';
const MODE_LABELS: Record<string, string> = {
  manual: 'MANUAL', aleatoria: 'ALEATORIA · referencia', regla: 'REGLA · referencia', evaluacion: 'EVALUACIÓN · modelo',
};
const CAUSES: Record<string, string> = {
  tiempo_agotado: 'Tiempo agotado', bateria_agotada: 'Batería agotada (rescates)', fragil_roto: 'Frágil roto',
  choques_repetidos: 'Choques repetidos',
};

let niveles: Niveles;
let catalogo: Catalogo = { catalogo: { recomendado: null, modelos: [] }, runs: { runs: [] } };
let escena: Escena;
const state = { nivel: 'n2', politica: 'modelo' as Politica, modelId: null as string | null, caso: 0 };
const manual = { acciones: [] as number[], mostrados: 0, terminado: false, chain: Promise.resolve() };
let lastModelo: ModeloInfo | null = null;

// ---------------------------------------------------------------- utilidades de interfaz

function toast(msg: string) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout((toast as unknown as { t?: number }).t);
  (toast as unknown as { t?: number }).t = window.setTimeout(() => { el.hidden = true; }, 7000);
}

function fail(error: unknown) {
  const msg = error instanceof ApiError ? `Error de la API (${error.status || 'sin conexión'}): ${error.message}` : String(error);
  setBadge('ERROR', 'error');
  toast(msg);
}

function setBadge(text: string, cls: string) {
  const el = $('mode-badge');
  el.textContent = text;
  el.className = `badge ${cls}`;
}

function showBanner(title: string, sub: string, cls: string) {
  $('banner').hidden = false;
  $('banner-title').textContent = title;
  $('banner-title').className = cls;
  $('banner-sub').textContent = sub;
}
const hideBanner = () => { $('banner').hidden = true; };

const level = (id = state.nivel): Nivel => niveles.niveles.find((n) => n.nivel === id)!;
const models = () => catalogo.catalogo.modelos;

function busy(on: boolean) {
  for (const id of ['btn-episodio', 'btn-evaluar']) $<HTMLButtonElement>(id).disabled = on;
}

// ---------------------------------------------------------------- panel de configuracion

function renderLevelSelect() {
  const sel = $<HTMLSelectElement>('sel-nivel');
  sel.innerHTML = niveles.niveles.map((n) => `<option value="${n.nivel}">${esc(n.nombre)}</option>`).join('');
  sel.value = state.nivel;
  const fn = $<HTMLSelectElement>('f-nivel');
  fn.innerHTML = '<option value="">Todos los niveles</option>'
    + niveles.niveles.map((n) => `<option value="${n.nivel}">${esc(n.nombre)}</option>`).join('');
}

function renderModelSelect() {
  const lv = level();
  const sel = $<HTMLSelectElement>('sel-modelo');
  const recommended = catalogo.catalogo.recomendado;
  const compatible = models().filter((m) => m.familia_modelo === lv.familia_modelo);
  const others = models().filter((m) => m.familia_modelo !== lv.familia_modelo);
  if (!compatible.some((m) => m.model_id === state.modelId)) {
    state.modelId = compatible.find((m) => m.model_id === recommended)?.model_id ?? compatible[0]?.model_id ?? null;
  }
  const label = (m: ModeloCatalogo) => `${m.model_id === recommended ? '★ ' : ''}${m.model_id} · ${pct(m.tasa_exito)}`;
  sel.innerHTML = (compatible.length ? '' : '<option value="">(sin modelos compatibles con este nivel)</option>')
    + compatible.map((m) => `<option value="${m.model_id}">${esc(label(m))}</option>`).join('')
    + others.map((m) => `<option disabled>${esc(m.model_id)} · incompatible (familia ${m.level_id})</option>`).join('');
  if (state.modelId) sel.value = state.modelId;
}

function renderPolicy() {
  document.querySelectorAll<HTMLButtonElement>('#sel-politica button').forEach((b) => {
    b.classList.toggle('active', b.dataset.p === state.politica);
  });
  $('field-modelo').hidden = state.politica !== 'modelo';
  $('btn-evaluar').toggleAttribute('disabled', state.politica === 'manual');
  $('controls').hidden = state.politica !== 'manual';
}

function renderActivo(amb?: Ambiente, modelo?: ModeloInfo | null) {
  const lv = level();
  const a = amb ?? lv;
  const rows: [string, string][] = [
    ['Nivel', a.nombre],
    ['Ambiente', `v${a.environment_version}`],
    ['Familia', `<code>${esc(a.familia_modelo.replace('warehouse-rush-v1/', ''))}</code>`],
    ['Suite', `<code>${esc(a.evaluation_suite_id)}</code> (${lv.casos_evaluacion} casos)`],
  ];
  if (state.politica === 'modelo') {
    const m = modelo ?? lastModelo;
    const meta = models().find((x) => x.model_id === (m?.model_id ?? state.modelId));
    if (meta) {
      rows.push(['Modelo', `<code>${esc(meta.model_id)}</code>`], ['SHA-256', `<code>${short(meta.model_sha256, 12)}</code>`],
        ['Run', esc(meta.run_name)], ['Entrenado en', meta.level_id.toUpperCase()],
        ['Commit', `<a href="${GITHUB}/commit/${meta.git_commit}" target="_blank" rel="noopener">${short(meta.git_commit)}</a>`],
        ['Estado', `<span class="pill ${meta.estado}">${meta.estado}</span>`]);
    }
  } else {
    rows.push(['Política', state.politica === 'manual' ? 'control manual' : `referencia ${state.politica}`]);
  }
  $('activo').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function selectLevel(id: string) {
  state.nivel = id;
  const lv = level();
  $('nivel-decision').textContent = lv.decision;
  $<HTMLInputElement>('inp-caso').max = String(lv.casos_evaluacion - 1);
  state.caso = Math.min(state.caso, lv.casos_evaluacion - 1);
  $<HTMLInputElement>('inp-caso').value = String(state.caso);
  $('battery-block').hidden = !lv.entorno.bateria;
  escena.setNivel(lv.mapa, lv.entorno.bateria, lv.entorno.bateria_max);
  renderModelSelect();
  renderActivo();
  resetHud();
  $('eval-block').hidden = true;
  hideBanner();
}

// ---------------------------------------------------------------- HUD del episodio

const STATUS_LABELS: Record<string, string> = { pendiente: 'pendiente', cargando: 'en camino', entregado: '✔ entregado' };

function resetHud() {
  for (const id of ['hud-step', 'hud-action', 'hud-reward', 'hud-total', 'hud-rescues']) $(id).textContent = '-';
  $('order').innerHTML = '';
  $('goal').textContent = '';
  $('episode-title').textContent = 'Sin episodio';
}

function renderBattery(value: number) {
  const max = level().entorno.bateria_max;
  const frac = value / max;
  $('hud-battery').textContent = `${Math.round(frac * 100)}%`;
  $('bat-fill').style.width = `${frac * 100}%`;
  $('bat-fill').style.background = `#${batteryColor(frac).toString(16).padStart(6, '0')}`;
}

function updateHud(f: Fotograma) {
  const lv = level();
  $('hud-step').textContent = `${f.steps} / ${lv.entorno.max_pasos}`;
  $('hud-action').textContent = f.action === null ? '-' : niveles.acciones[f.action];
  const rw = $('hud-reward');
  rw.textContent = fmt(f.reward);
  rw.className = f.reward > 0 ? 'pos' : f.reward <= -1 ? 'neg' : '';
  $('hud-total').textContent = fmt(f.total);
  $('hud-rescues').textContent = String(f.rescues);
  renderBattery(f.battery);
  $('order').innerHTML = f.order.map((p, i) => {
    const cls = p.status === 'entregado' ? 'done' : i === f.current ? 'current' : '';
    const broken = p.broken ? ` <span class="broken">roto ×${p.broken}</span>` : '';
    return `<li class="${cls}"><span>${i + 1}.</span><span class="type">${TYPE_LABELS[p.type]}${broken}</span>`
      + `<span class="zone" style="background:${ZONE_CSS[p.zone]}">${niveles.zonas[p.zone]}</span>`
      + `<span class="st">${STATUS_LABELS[p.status]}</span></li>`;
  }).join('');
  if (state.politica === 'manual') {
    const cur = f.order[Math.min(f.current, f.order.length - 1)];
    let goal = f.carrying
      ? `Lleva el paquete a la <b style="color:${ZONE_CSS[cur.zone]}">Zona ${niveles.zonas[cur.zone]}</b> y pulsa <kbd>E</kbd>.`
      : 'Ve junto a la estantería marcada y pulsa <kbd>R</kbd>.';
    if (lv.entorno.bateria && f.battery / lv.entorno.bateria_max <= 0.3) goal += '<br><b class="gold">Batería baja:</b> ve a la estación ⚡ y pulsa <kbd>C</kbd>.';
    if (f.success) goal = '¡Pedido completo! Pulsa <kbd>N</kbd> o “Ejecutar episodio” para otro.';
    $('goal').innerHTML = goal;
  } else {
    $('goal').textContent = '';
  }
}

function onEpisodeEnd(f: Fotograma) {
  if (!(f.terminated || f.truncated)) return;
  const delivered = f.order.filter((p) => p.status === 'entregado').length;
  const sub = `${delivered}/${f.order.length} paquetes · ${f.steps} pasos · recompensa ${fmt(f.total)}`
    + (f.rescues ? ` · ${f.rescues} rescate(s)` : '');
  if (f.success) showBanner('¡PEDIDO COMPLETO!', sub, 'pos');
  else showBanner('TIEMPO AGOTADO', sub, 'neg');
}

// ---------------------------------------------------------------- acciones

function modeBadge(modo: string, modelo: ModeloInfo | null) {
  const label = MODE_LABELS[modo] ?? modo.toUpperCase();
  setBadge(modo === 'evaluacion' && modelo ? `${label} ${modelo.model_id}` : label, modo === 'evaluacion' ? 'evaluacion' : modo);
}

async function runEpisode() {
  hideBanner();
  state.caso = Math.max(0, Math.min(Number($<HTMLInputElement>('inp-caso').value) || 0, level().casos_evaluacion - 1));
  syncUrl();
  if (state.politica === 'manual') return startManual();
  busy(true);
  try {
    const ep = await api.episodio({ nivel: state.nivel, politica: state.politica, model_id: state.modelId, caso: state.caso });
    lastModelo = ep.modelo;
    modeBadge(ep.modo, ep.modelo);
    renderActivo(ep.ambiente, ep.modelo);
    escena.limpiar();
    $('episode-title').textContent = `${level().nombre.split('·')[0].trim()} · caso ${ep.caso}`;
    escena.encolar(ep.fotogramas);
  } catch (error) {
    fail(error);
  } finally {
    busy(false);
  }
}

async function runEvaluation() {
  busy(true);
  const casos = Math.max(1, Math.min(Number($<HTMLInputElement>('inp-casos').value) || 50, 500));
  try {
    const ev = await api.evaluar({ nivel: state.nivel, politica: state.politica, model_id: state.modelId, casos });
    lastModelo = ev.modelo;
    modeBadge(ev.politica === 'modelo' ? 'evaluacion' : ev.politica, ev.modelo);
    renderActivo(ev.ambiente, ev.modelo);
    renderEvaluation(ev);
  } catch (error) {
    fail(error);
  } finally {
    busy(false);
  }
}

function renderEvaluation(ev: Evaluacion) {
  const m = ev.metricas;
  $('eval-block').hidden = false;
  $('eval-scope').textContent = `· ${m.pedidos_evaluados} casos ${ev.suite_completa ? '(suite completa)' : '(primeros de la suite)'} · ${ev.tiempo_servidor_s.toFixed(1)} s`;
  const tiles: [string, string][] = [
    [pct(m.tasa_exito), 'pedidos completos'], [num(m.pasos_medios_exitosos), 'pasos (exitosos)'],
    [num(m.rescates_medios, 2), 'rescates / pedido'], [num(m.choques_medios, 2), 'choques / pedido'],
    [num(m.fragiles_rotos_medios, 2), 'frágiles rotos'], [`${num(m.latencia_media_ms, 3)} ms`, 'latencia / decisión'],
  ];
  $('eval-metrics').innerHTML = tiles.map(([v, l]) => `<div class="metric"><b>${v}</b><span>${l}</span></div>`).join('');
  const causes = Object.entries(m.fallos_por_causa);
  $('eval-causes').innerHTML = causes.length
    ? `<h3>Fallos por causa</h3>${causes.map(([k, v]) => `<div class="cause"><span>${CAUSES[k] ?? k}</span><b>${v}</b></div>`).join('')}`
    : '<p class="hint">Sin fallos en estos casos.</p>';
  $('eval-cases').innerHTML = '<tr><th>Caso</th><th>Resultado</th><th>Pasos</th><th>Recomp.</th><th>Resc.</th></tr>'
    + ev.por_caso.map((c) => `<tr class="clickable" data-caso="${c.caso}"><td>${c.caso}</td>`
      + `<td class="${c.exito ? 'pos' : 'neg'}">${c.exito ? 'éxito' : CAUSES[c.causa] ?? c.causa}</td>`
      + `<td class="num">${c.pasos}</td><td class="num">${fmt(c.recompensa)}</td><td class="num">${c.rescates}</td></tr>`).join('');
  $('eval-cases').querySelectorAll<HTMLTableRowElement>('tr[data-caso]').forEach((tr) => {
    tr.addEventListener('click', () => {
      $<HTMLInputElement>('inp-caso').value = tr.dataset.caso!;
      runEpisode();
    });
  });
}

// ---------------------------------------------------------------- control manual (la API reproduce las acciones)

function startManual() {
  Object.assign(manual, { acciones: [], mostrados: 0, terminado: false });
  setBadge(MODE_LABELS.manual, 'manual');
  escena.limpiar();
  $('episode-title').textContent = `${level().nombre.split('·')[0].trim()} · caso ${state.caso} · manual`;
  sendManual();
}

function sendManual() {
  const acciones = manual.acciones.slice();
  manual.chain = manual.chain.then(async () => {
    try {
      const ep = await api.episodio({ nivel: state.nivel, politica: 'manual', caso: state.caso, acciones });
      renderActivo(ep.ambiente, null);
      const nuevos = ep.fotogramas.slice(manual.mostrados);
      manual.mostrados = ep.fotogramas.length;
      const last = ep.fotogramas[ep.fotogramas.length - 1];
      manual.terminado = last.terminated || last.truncated;
      escena.encolar(nuevos);
    } catch (error) {
      manual.acciones.pop();
      fail(error);
    }
  });
}

const KEYS: Record<string, 'up' | 'down' | 'left' | 'right'> = {
  arrowup: 'up', w: 'up', arrowdown: 'down', s: 'down', arrowleft: 'left', a: 'left', arrowright: 'right', d: 'right',
};

window.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return;
  if (['1', '2', '3', '4'].includes(e.key)) {
    escena.setCamara((['iso', 'top', 'follow', 'side'] as const)[Number(e.key) - 1]);
    return;
  }
  if (state.politica !== 'manual' || !$('exp-view').hidden || !$('mod-view').hidden) return;
  const k = e.key.toLowerCase();
  let action: number | null = null;
  if (k in KEYS) action = escena.accionRelativa(KEYS[k]);
  else if (k === 'r' || k === ' ') action = PICK;
  else if (k === 'e' || k === 'enter') action = DROP;
  else if (k === 'c') action = CHARGE;
  else if (k === 'n') { e.preventDefault(); startManual(); return; }
  if (action === null) return;
  e.preventDefault();
  if (manual.terminado || manual.mostrados === 0 || escena.pendientes > 2) return;
  manual.acciones.push(action);
  sendManual();
});

// ---------------------------------------------------------------- experimentos

const selectedRuns = new Set<string>();

function renderExperiments() {
  const runs = catalogo.runs.runs;
  const fNivel = $<HTMLSelectElement>('f-nivel').value;
  const fTipo = $<HTMLSelectElement>('f-tipo').value;
  $('exp-source').textContent = `${runs.length} runs · registro ${catalogo.runs.tracking === 'dagshub' ? 'MLflow en DagsHub' : 'MLflow local exportado'}`;
  const rows = runs.filter((r) => (!fNivel || r.nivel === fNivel) && (!fTipo || r.tipo === fTipo)).slice().reverse();
  const head = '<tr><th></th><th>Run</th><th>Nivel</th><th>Tipo</th><th>Estado</th><th>Hipótesis / cambios</th><th>Semilla</th>'
    + '<th>Episodios</th><th>Éxito eval.</th><th>Pasos</th><th>Rescates</th><th>Choques</th><th>Tiempo</th><th>Modelo</th>'
    + '<th>Commit</th><th>MLflow</th><th></th></tr>';
  $('exp-table').innerHTML = head + rows.map((r) => {
    const m = r.metricas;
    const model = r.model_id ? `<code>${esc(r.model_id)}</code>` : `<span class="muted">${esc(r.estado_modelo || '-')}</span>`;
    const load = r.model_id ? `<button class="small-btn" data-load="${esc(r.model_id)}" data-nivel="${r.nivel}">Cargar</button>`
      : r.tipo === 'referencia' ? `<button class="small-btn" data-ref="${esc(r.nombre)}" data-nivel="${r.nivel}">Evaluar</button>` : '';
    return `<tr><td><input type="checkbox" data-run="${r.run_id}" ${selectedRuns.has(r.run_id) ? 'checked' : ''}></td>`
      + `<td><b>${esc(r.nombre)}</b>${r.parent_run_id ? '<br><span class="muted">continúa otro run</span>' : ''}</td>`
      + `<td>${r.nivel.toUpperCase()}</td><td><span class="pill ${r.tipo}">${r.tipo}</span></td>`
      + `<td><span class="pill ${r.estado}">${r.estado}</span>${r.error ? `<br><span class="neg">${esc(r.error)}</span>` : ''}</td>`
      + `<td class="wrap">${esc(r.hipotesis) || '<span class="muted">sin hipótesis registrada</span>'}<br><code>${esc(r.cambios)}</code></td>`
      + `<td>${esc(r.semilla ?? '-')}</td><td class="num">${r.tipo === 'entrenamiento' ? esc(r.episodios ?? '-') : '-'}</td>`
      + `<td class="num">${pct(m.tasa_exito)}</td><td class="num">${num(m.pasos_medios)}</td>`
      + `<td class="num">${num(m.rescates_medios, 2)}</td><td class="num">${num(m.choques_medios, 2)}</td>`
      + `<td class="num">${r.tiempo_entrenamiento_s ? `${r.tiempo_entrenamiento_s.toFixed(0)} s` : '-'}</td><td>${model}</td>`
      + `<td>${r.git_commit ? `<a href="${GITHUB}/commit/${r.git_commit}" target="_blank" rel="noopener">${short(r.git_commit)}</a>` : '-'}</td>`
      + `<td><a href="${esc(r.mlflow_url)}" target="_blank" rel="noopener">abrir</a></td><td>${load}</td></tr>`;
  }).join('');
  $('exp-table').querySelectorAll<HTMLInputElement>('input[data-run]').forEach((cb) => {
    cb.addEventListener('change', () => {
      if (cb.checked) selectedRuns.add(cb.dataset.run!); else selectedRuns.delete(cb.dataset.run!);
      renderCompare();
    });
  });
  $('exp-table').querySelectorAll<HTMLButtonElement>('button[data-load]').forEach((b) => {
    b.addEventListener('click', () => loadModel(b.dataset.load!, b.dataset.nivel!));
  });
  $('exp-table').querySelectorAll<HTMLButtonElement>('button[data-ref]').forEach((b) => {
    b.addEventListener('click', () => {
      state.politica = b.dataset.ref!.includes('aleatoria') ? 'aleatoria' : 'regla';
      switchTab('sim');
      $<HTMLSelectElement>('sel-nivel').value = b.dataset.nivel!;
      selectLevel(b.dataset.nivel!);
      renderPolicy();
      runEvaluation();
    });
  });
  renderCompare();
}

const PALETTE = ['#ff8a2a', '#5aa9ff', '#4fd18b', '#c084fc', '#ffc531', '#f472b6'];

function drawChart(canvas: HTMLCanvasElement, series: { label: string; color: string; points: [number, number][] }[]) {
  const ctx = canvas.getContext('2d')!;
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 44, r: 12, t: 12, b: 28 };
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const xMax = Math.max(1, ...xs);
  const X = (x: number) => pad.l + (x / xMax) * (w - pad.l - pad.r);
  const Y = (y: number) => h - pad.b - y * (h - pad.t - pad.b);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.fillStyle = '#94a0b2';
  ctx.font = '12px system-ui';
  for (const y of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(y)); ctx.lineTo(w - pad.r, Y(y)); ctx.stroke();
    ctx.fillText(`${y * 100}%`, 6, Y(y) + 4);
  }
  ctx.fillText('episodios', w - 70, h - 8);
  ctx.fillText(xMax.toLocaleString(), X(xMax) - 40, h - 8);
  series.forEach((s, i) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    s.points.forEach(([x, y], j) => (j ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, pad.l + 8, pad.t + 14 + i * 16);
  });
}

function renderCompare() {
  const runs = catalogo.runs.runs.filter((r) => selectedRuns.has(r.run_id));
  $('compare').hidden = runs.length < 2;
  if (runs.length < 2) return;
  const series = (key: string) => runs.map((r, i) => ({ label: r.nombre, color: PALETTE[i % PALETTE.length], points: r.curvas[key] ?? [] }));
  drawChart($<HTMLCanvasElement>('chart-train'), series('entrenamiento.tasa_exito'));
  drawChart($<HTMLCanvasElement>('chart-eval'), series('curva_eval.tasa_exito'));
  const metrics: [string, (r: Run) => string][] = [
    ['Nivel', (r) => r.nivel.toUpperCase()], ['Cambios', (r) => esc(r.cambios)], ['Semilla', (r) => esc(r.semilla ?? '-')],
    ['Episodios (presupuesto)', (r) => (r.tipo === 'entrenamiento' ? esc(r.episodios ?? '-') : 'sin entrenamiento')], ['Suite', (r) => `<code>${esc(r.evaluation_suite_id || '-')}</code>`],
    ['Pedidos completos', (r) => pct(r.metricas.tasa_exito)], ['Pasos medios', (r) => num(r.metricas.pasos_medios)],
    ['Pasos (exitosos)', (r) => num(r.metricas.pasos_medios_exitosos)], ['Recompensa media', (r) => num(r.metricas.recompensa_media)],
    ['Rescates / pedido', (r) => num(r.metricas.rescates_medios, 3)], ['Choques / pedido', (r) => num(r.metricas.choques_medios, 3)],
    ['Frágiles rotos', (r) => num(r.metricas.fragiles_rotos_medios, 3)], ['Latencia (ms)', (r) => num(r.metricas.latencia_media_ms, 4)],
  ];
  $('compare-table').innerHTML = `<tr><th>Métrica</th>${runs.map((r) => `<th>${esc(r.nombre)}</th>`).join('')}</tr>`
    + metrics.map(([label, f]) => `<tr><td>${label}</td>${runs.map((r) => `<td class="num">${f(r)}</td>`).join('')}</tr>`).join('');
}

function loadModel(modelId: string, nivel: string) {
  const meta = models().find((m) => m.model_id === modelId);
  if (!meta) return toast(`El modelo ${modelId} no está en el catálogo publicado.`);
  switchTab('sim');
  const compatible = niveles.niveles.find((n) => n.nivel === nivel && n.familia_modelo === meta.familia_modelo)
    ?? niveles.niveles.find((n) => n.familia_modelo === meta.familia_modelo)!;
  $<HTMLSelectElement>('sel-nivel').value = compatible.nivel;
  selectLevel(compatible.nivel);
  state.politica = 'modelo';
  state.modelId = modelId;
  renderPolicy();
  renderModelSelect();
  renderActivo();
  runEpisode();
}

// ---------------------------------------------------------------- catalogo de modelos

function renderModels() {
  const cat = catalogo.catalogo;
  $('mod-updated').textContent = cat.actualizado ? `actualizado ${new Date(cat.actualizado).toLocaleString()}` : '';
  $('mod-table').innerHTML = '<tr><th>Modelo</th><th>Estado</th><th>Nivel</th><th>Ambiente</th><th>Éxito eval.</th><th>SHA-256</th>'
    + '<th>Hugging Face</th><th>Commit</th><th>Publicado</th><th>Hipótesis</th><th></th></tr>'
    + (cat.modelos.length ? cat.modelos.slice().reverse().map((m) => `<tr>`
      + `<td><b>${esc(m.model_id)}</b>${m.model_id === cat.recomendado ? ' ★' : ''}</td>`
      + `<td><span class="pill ${m.estado}">${m.estado}</span>${m.motivo ? `<br><span class="muted">${esc(m.motivo)}</span>` : ''}</td><td>${m.level_id.toUpperCase()}</td>`
      + `<td>v${esc(m.environment_version)}</td><td class="num">${pct(m.tasa_exito)}</td>`
      + `<td><code>${short(m.model_sha256, 12)}</code></td>`
      + `<td><code>${short(m.hf_commit, 8)}</code>${m.hf_tag ? ` · ${esc(m.hf_tag)}` : ''}</td>`
      + `<td><a href="${GITHUB}/commit/${m.git_commit}" target="_blank" rel="noopener">${short(m.git_commit)}</a></td>`
      + `<td>${new Date(m.publicado).toLocaleDateString()}</td><td class="wrap">${esc(m.hipotesis)}</td>`
      + `<td><button class="small-btn" data-manifest="${esc(m.model_id)}">Manifiesto</button> `
      + `<button class="small-btn" data-load="${esc(m.model_id)}" data-nivel="${m.level_id}">Cargar</button></td></tr>`).join('')
      : '<tr><td colspan="11" class="muted">Todavía no hay modelos publicados.</td></tr>');
  $('mod-table').querySelectorAll<HTMLButtonElement>('button[data-load]').forEach((b) => {
    b.addEventListener('click', () => loadModel(b.dataset.load!, b.dataset.nivel!));
  });
  $('mod-table').querySelectorAll<HTMLButtonElement>('button[data-manifest]').forEach((b) => {
    b.addEventListener('click', async () => {
      try {
        const res = await fetch(`${(import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')}/api/modelos/${b.dataset.manifest}`);
        const pre = $('manifest-view');
        pre.hidden = false;
        pre.textContent = JSON.stringify(await res.json(), null, 2);
        pre.scrollIntoView({ behavior: 'smooth' });
      } catch (error) {
        fail(error);
      }
    });
  });
}

// ---------------------------------------------------------------- pestanas e inicio

// Enlaces profundos: ?tab=exp&nivel=n3&politica=modelo&modelo=<id>&caso=4
function syncUrl(tab?: string) {
  const params = new URLSearchParams(location.search);
  if (tab) params.set('tab', tab);
  params.set('nivel', state.nivel);
  params.set('politica', state.politica);
  params.set('caso', String(state.caso));
  if (state.politica === 'modelo' && state.modelId) params.set('modelo', state.modelId); else params.delete('modelo');
  history.replaceState(null, '', `${location.pathname}?${params}`);
}

function switchTab(tab: string) {
  syncUrl(tab);
  document.querySelectorAll<HTMLButtonElement>('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $('exp-view').hidden = tab !== 'exp';
  $('mod-view').hidden = tab !== 'mod';
  if (tab === 'exp') renderExperiments();
  if (tab === 'mod') renderModels();
}

async function init() {
  document.querySelectorAll<HTMLButtonElement>('#tabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab!)));
  try {
    const [salud, nv, cat] = await Promise.all([api.salud(), api.niveles(), api.catalogo()]);
    niveles = nv;
    catalogo = cat;
    const st = $('api-status');
    st.className = 'api-status ok';
    st.textContent = `API v${salud.api_version} · ambiente v${salud.environment_version} · commit ${short(salud.git_commit)}`;
  } catch (error) {
    $('api-status').className = 'api-status bad';
    $('api-status').textContent = 'API no disponible';
    fail(error);
    return;
  }

  escena = new Escena($('scene'), $('popups'), niveles.zonas);
  escena.onFotograma = updateHud;
  escena.onFinFotograma = onEpisodeEnd;
  const speed = $<HTMLInputElement>('speed');
  const applySpeed = () => { escena.stepMs = Math.round(500 * Math.pow(0.05, Number(speed.value) / 100)); };
  speed.addEventListener('input', applySpeed);
  applySpeed();

  const params = new URLSearchParams(location.search);
  if (niveles.niveles.some((n) => n.nivel === params.get('nivel'))) state.nivel = params.get('nivel')!;
  if (['manual', 'aleatoria', 'regla', 'modelo'].includes(params.get('politica') ?? '')) state.politica = params.get('politica') as Politica;
  if (params.get('modelo')) state.modelId = params.get('modelo');
  state.caso = Number(params.get('caso')) || 0;
  renderLevelSelect();
  selectLevel(state.nivel);
  renderPolicy();
  $<HTMLSelectElement>('sel-nivel').addEventListener('change', (e) => selectLevel((e.target as HTMLSelectElement).value));
  $<HTMLSelectElement>('sel-modelo').addEventListener('change', (e) => {
    state.modelId = (e.target as HTMLSelectElement).value || null;
    renderActivo();
  });
  document.querySelectorAll<HTMLButtonElement>('#sel-politica button').forEach((b) => b.addEventListener('click', () => {
    state.politica = b.dataset.p as Politica;
    renderPolicy();
    renderActivo();
    setBadge(state.politica === 'modelo' ? 'EVALUACIÓN' : MODE_LABELS[state.politica], state.politica === 'modelo' ? 'evaluacion' : state.politica);
  }));
  $('btn-episodio').addEventListener('click', runEpisode);
  $('btn-evaluar').addEventListener('click', runEvaluation);
  $('btn-azar').addEventListener('click', () => {
    $<HTMLInputElement>('inp-caso').value = String(Math.floor(Math.random() * level().casos_evaluacion));
  });
  $<HTMLSelectElement>('f-nivel').addEventListener('change', renderExperiments);
  $<HTMLSelectElement>('f-tipo').addEventListener('change', renderExperiments);
  document.querySelectorAll<HTMLButtonElement>('[data-cam]').forEach((b) => b.addEventListener('click', () => {
    escena.setCamara(b.dataset.cam as 'iso' | 'top' | 'follow' | 'side');
  }));
  if (state.politica === 'modelo' && !models().some((m) => m.familia_modelo === level().familia_modelo)) {
    state.politica = 'regla';
    renderPolicy();
    renderActivo();
  }
  if (params.get('tab') && params.get('tab') !== 'sim') switchTab(params.get('tab')!);
  if (state.politica !== 'manual') runEpisode();
  else startManual();
}

init();
