"""API de simulacion e inferencia de Warehouse Rush (FastAPI, desplegada como Vercel Function).

Python es la autoridad: la web solo dibuja los fotogramas que devuelve esta API. La API no guarda estado
entre peticiones; el control manual reproduce desde el inicio la lista de acciones que envia el navegador.

Endpoints:
    GET  /api/salud               version de la API, commit desplegado y version del ambiente
    GET  /api/niveles             mapa y configuracion de cada nivel (N1, N2, N3)
    GET  /api/catalogo            modelos publicados (manifiestos), recomendado y runs de MLflow
    GET  /api/modelos/{model_id}  manifiesto completo de un modelo
    POST /api/episodio            ejecuta un pedido (manual, aleatoria, regla o modelo) y devuelve los fotogramas
    POST /api/evaluar             evalua una politica en la suite del nivel (resultados por caso y fallos por causa)
    GET  /api/monitoreo           sonda del modelo servido en cada nivel (contra lo registrado y los criterios) y
                                  peticiones, latencias y errores de esta instancia

Variables de entorno: HF_TOKEN (solo si el repositorio de modelos es privado), CORS_ORIGINS,
MANIFESTS_DIR (solo para pruebas locales).
"""

import collections
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.request
from functools import lru_cache
from typing import Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from warehouse_rush import niveles  # noqa: E402
from warehouse_rush.env_v1 import ACTION_NAMES, LAYOUT, TYPE_NAMES, ZONE_NAMES  # noqa: E402
from warehouse_rush.modelos import FILES, TABULAR, load_policy  # noqa: E402
from warehouse_rush.politicas import RandomPolicy, RulePolicy  # noqa: E402
from warehouse_rush.simulacion import evaluate, replay_manual, run_episode  # noqa: E402

API_VERSION = "1.0.0"
MANIFESTS = os.environ.get("MANIFESTS_DIR", os.path.join(ROOT, "manifests"))
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/warehouse-rush-modelos")
MAX_EVAL_CASES = 500

app = FastAPI(title="Warehouse Rush API", version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


# ---------------------------------------------------------------- monitoreo de la instancia
# La API no tiene estado persistente: estos registros viven en la memoria de la instancia de Vercel que atiende
# la peticion y se reinician en cada arranque en frio. La sonda de /api/monitoreo no depende de ellos.

STARTED = time.time()
REQUESTS = collections.deque(maxlen=500)
ERRORS = collections.deque(maxlen=50)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@app.middleware("http")
async def record_request(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as error:  # error no controlado: se registra y se propaga (respuesta 500)
        ERRORS.append({"t": _now(), "ruta": request.url.path, "estado": 500,
                       "detalle": f"{type(error).__name__}: {error}"[:300]})
        raise
    finally:
        if request.url.path.startswith("/api"):
            REQUESTS.append({"t": _now(), "ruta": request.url.path, "metodo": request.method, "estado": status,
                             "ms": round((time.perf_counter() - start) * 1000, 1)})


@app.exception_handler(HTTPException)
async def record_http_error(request: Request, error: HTTPException):
    ERRORS.append({"t": _now(), "ruta": request.url.path, "estado": error.status_code,
                   "detalle": str(error.detail)[:300]})
    return await http_exception_handler(request, error)


def instance_stats():
    by_route = collections.defaultdict(list)
    for r in REQUESTS:
        by_route[f"{r['metodo']} {r['ruta']}"].append(r)
    routes = {}
    for route, rows in sorted(by_route.items()):
        ms = sorted(r["ms"] for r in rows)
        routes[route] = {"peticiones": len(rows), "errores": sum(r["estado"] >= 400 for r in rows),
                         "p50_ms": ms[len(ms) // 2], "p95_ms": ms[min(len(ms) - 1, int(len(ms) * 0.95))]}
    return {"inicio": datetime.datetime.fromtimestamp(STARTED, datetime.timezone.utc).isoformat(timespec="seconds"),
            "segundos_activa": round(time.time() - STARTED), "peticiones": len(REQUESTS),
            "errores_4xx": sum(400 <= r["estado"] < 500 for r in REQUESTS),
            "errores_5xx": sum(r["estado"] >= 500 for r in REQUESTS), "por_ruta": routes,
            "ultimos_errores": list(ERRORS)[-15:][::-1],
            "nota": "memoria de esta instancia de Vercel: se reinicia en cada arranque en frio"}


# ---------------------------------------------------------------- datos versionados

def read_json(*parts, default=None):
    path = os.path.join(MANIFESTS, *parts)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=8)
def level_bundle(level):
    try:
        cfg = niveles.resolve(level)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    env = niveles.make_env(cfg)
    cases, suite_id = niveles.evaluation_suite(cfg)
    return cfg, niveles.family(env), cases, suite_id


@lru_cache(maxsize=8)
def level_families(level):
    return tuple(niveles.compatible_families(niveles.make_env(niveles.resolve(level))))


def manifest(model_id):
    data = read_json("modelos", f"{model_id}.json")
    if data is None:
        raise HTTPException(404, f"Modelo desconocido: {model_id}")
    return data


@lru_cache(maxsize=16)
def load_model(model_id):
    """Descarga el modelo (con cache en /tmp), verifica su SHA-256 contra el manifiesto y devuelve su politica.

    Tabla Q (.npy) o red PPO exportada a NumPy (.npz), segun `tipo_modelo` del manifiesto."""
    info = manifest(model_id)
    extension = os.path.splitext(FILES[info.get("tipo_modelo", TABULAR)])[1]
    path = os.path.join(CACHE_DIR, f"{info['model_sha256']}{extension}")
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        request = urllib.request.Request(info["model_download_url"])
        if os.environ.get("HF_TOKEN"):
            request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        except Exception as error:  # noqa: BLE001
            raise HTTPException(502, f"No se pudo descargar el modelo {model_id}: {error}") from error
        if hashlib.sha256(data).hexdigest() != info["model_sha256"]:
            raise HTTPException(502, f"El SHA-256 del modelo {model_id} no coincide con su manifiesto")
        with open(path, "wb") as f:
            f.write(data)
    return load_policy(path)


def resolve_policy(policy, model_id, level, family, seed):
    """Devuelve (politica, descripcion del modelo). Valida la compatibilidad modelo-ambiente."""
    if policy == "aleatoria":
        return RandomPolicy(seed=seed), None
    if policy == "regla":
        return RulePolicy(), None
    if policy != "modelo":
        raise HTTPException(400, f"Politica desconocida: {policy}")
    model_id = model_id or recommended().get(level)
    if not model_id:
        raise HTTPException(404, f"No hay un modelo recomendado para el nivel {level}; elige uno del catalogo")
    info = manifest(model_id)
    if info["familia_modelo"] not in level_families(level):
        raise HTTPException(409, f"El modelo {model_id} ({info['familia_modelo']}) no es compatible con el nivel "
                                 f"{level} ({family}): sus observaciones tienen otras dimensiones")
    return load_model(model_id), {
        "model_id": model_id, "tipo_modelo": info.get("tipo_modelo", TABULAR),
        "model_sha256": info["model_sha256"], "nivel_entrenamiento": info["level_id"],
        "environment_version": info["environment_version"], "run_name": info["run_name"],
        "hf_commit": info["hf_commit"], "git_commit": info["git_commit"]}


def recommended():
    """Modelo recomendado de cada nivel segun el catalogo publicado."""
    catalog = read_json("catalogo.json", default={}) or {}
    return catalog.get("recomendados", {})


def environment_info(cfg, family, suite_id):
    return {"nivel": cfg["nivel"], "nombre": cfg["nombre"], "environment_version": cfg["version_ambiente"],
            "familia_modelo": family, "evaluation_suite_id": suite_id}


# ---------------------------------------------------------------- endpoints

@app.get("/api/salud")
def health():
    return {"estado": "ok", "api_version": API_VERSION, "environment_versions": niveles.env_versions(),
            "git_commit": os.environ.get("VERCEL_GIT_COMMIT_SHA", "local"),
            "despliegue": os.environ.get("VERCEL_URL", "local"), "modelos_recomendados": recommended()}


@app.get("/api/niveles")
def levels():
    result = []
    for level in niveles.level_ids():
        cfg, family, cases, suite_id = level_bundle(level)
        env = niveles.make_env(cfg).unwrapped
        result.append({
            **environment_info(cfg, family, suite_id), "familias_compatibles": list(level_families(level)),
            "decision": cfg["decision"], "entorno": cfg["entorno"],
            "datos": cfg["datos"], "casos_evaluacion": len(cases),
            "mapa": {"layout": LAYOUT, "shelf_cells": [list(p) for p in env.shelf_cells],
                     "zone_cells": [sorted(list(p) for p in z) for z in env.zone_cells], "charger": list(env.charger),
                     "start": list(env.start_pos), "bloqueos": [list(c) for c in env.blocked_cells]},
        })
    return {"niveles": result, "acciones": ACTION_NAMES, "tipos": TYPE_NAMES, "zonas": ZONE_NAMES}


@app.get("/api/catalogo")
def catalog():
    return {"catalogo": read_json("catalogo.json", default={"recomendados": {}, "modelos": []}),
            "runs": read_json("runs.json", default={"runs": []})}


@app.get("/api/modelos/{model_id}")
def model_manifest(model_id: str):
    return manifest(model_id)


class EpisodeRequest(BaseModel):
    nivel: str = "n2"
    politica: Literal["manual", "aleatoria", "regla", "modelo"] = "modelo"
    model_id: Optional[str] = None
    caso: int = Field(0, ge=0, description="indice del pedido en la suite de evaluacion del nivel")
    acciones: list[int] = Field(default_factory=list, description="acciones del control manual")


@app.post("/api/episodio")
def episode(req: EpisodeRequest):
    cfg, family, cases, suite_id = level_bundle(req.nivel)
    if req.caso >= len(cases):
        raise HTTPException(400, f"El nivel {req.nivel} tiene {len(cases)} casos (0-{len(cases) - 1})")
    env = niveles.make_env(cfg)
    info = environment_info(cfg, family, suite_id)
    if req.politica == "manual":
        if any(a < 0 or a >= len(ACTION_NAMES) for a in req.acciones):
            raise HTTPException(400, "Accion invalida")
        frames = replay_manual(env, cases[req.caso], req.acciones, seed=req.caso)
        return {"modo": "manual", "ambiente": info, "modelo": None, "caso": req.caso, "fotogramas": frames}
    policy, model = resolve_policy(req.politica, req.model_id, req.nivel, family, seed=req.caso)
    start = time.perf_counter()
    result, frames = run_episode(env, policy, cases[req.caso], seed=req.caso)
    return {"modo": "evaluacion" if req.politica == "modelo" else req.politica, "ambiente": info, "modelo": model,
            "caso": req.caso, "resultado": result, "fotogramas": frames,
            "tiempo_servidor_ms": round((time.perf_counter() - start) * 1000, 2)}


# ---------------------------------------------------------------- monitoreo del modelo servido

PROBE_KEYS = ("tasa_exito", "otif", "cumplimiento_express", "rescates_medios", "fragiles_rotos_medios",
              "utilidad_media_cop", "pasos_medios", "latencia_media_ms")
DRIFT_TOLERANCE = 0.10  # caida de exito frente a lo registrado que se marca como deriva


@lru_cache(maxsize=1)
def requirements():
    """Requisitos de configs/criterios.yaml (los mismos que se exigen para promover un modelo)."""
    defaults = {"tasa_exito_min": 0.95, "fragiles_rotos_medios_max": 0.05, "rescates_medios_max": 0.30}
    try:
        with open(os.path.join(ROOT, "configs", "criterios.yaml")) as f:
            return {**defaults, **yaml.safe_load(f)["requisitos"]}
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return defaults


def check(metrics, registered, req):
    alerts = []
    if metrics["tasa_exito"] < req["tasa_exito_min"]:
        alerts.append(f"exito {metrics['tasa_exito']:.1%} por debajo del minimo {req['tasa_exito_min']:.0%}")
    if metrics.get("rescates_medios", 0) > req["rescates_medios_max"]:
        alerts.append(f"rescates {metrics['rescates_medios']:.2f} por pedido, maximo {req['rescates_medios_max']}")
    if metrics.get("fragiles_rotos_medios", 0) > req["fragiles_rotos_medios_max"]:
        alerts.append(f"fragiles rotos {metrics['fragiles_rotos_medios']:.2f} por pedido, "
                      f"maximo {req['fragiles_rotos_medios_max']}")
    if registered.get("tasa_exito") is not None and registered["tasa_exito"] - metrics["tasa_exito"] > DRIFT_TOLERANCE:
        alerts.append(f"deriva: exito {metrics['tasa_exito']:.1%} frente a {registered['tasa_exito']:.1%} registrado")
    return alerts


def probe_level(level, catalog, n_cases, req):
    cfg, family, cases, suite_id = level_bundle(level)
    item = {"nivel": level, "nombre": cfg["nombre"], "environment_version": cfg["version_ambiente"],
            "evaluation_suite_id": suite_id, "historial": catalog.get("historial_recomendados", {}).get(level, [])}
    model_id = catalog.get("recomendados", {}).get(level)
    if not model_id:
        return {**item, "estado": "sin_modelo", "alertas": ["no hay un modelo recomendado; se sirve la politica de reglas"]}
    try:
        info = manifest(model_id)
        item["modelo"] = {"model_id": model_id, "tipo_modelo": info.get("tipo_modelo", TABULAR),
                          "model_sha256": info["model_sha256"], "run_name": info["run_name"],
                          "publicado": info.get("publicado"), "hf_commit": info["hf_commit"],
                          "git_commit": info["git_commit"]}
        registered = info.get("metricas_evaluacion", {})
        item["registrado"] = {k: registered.get(k) for k in PROBE_KEYS if registered.get(k) is not None}
        item["registrado"]["casos"] = int(registered.get("pedidos_evaluados", len(cases)))
        start = time.perf_counter()
        policy, _ = resolve_policy("modelo", model_id, level, family, seed=0)
        metrics, per_case = evaluate(niveles.make_env(cfg), policy, cases[:n_cases])
    except HTTPException as error:  # modelo inaccesible, SHA distinto, familia incompatible...
        return {**item, "estado": "error", "alertas": [f"error de inferencia ({error.status_code}): {error.detail}"]}
    except Exception as error:  # noqa: BLE001
        ERRORS.append({"t": _now(), "ruta": f"/api/monitoreo ({level})", "estado": 500,
                       "detalle": f"{type(error).__name__}: {error}"[:300]})
        return {**item, "estado": "error", "alertas": [f"error de inferencia: {type(error).__name__}: {error}"]}
    item["sonda"] = {k: metrics.get(k) for k in PROBE_KEYS if metrics.get(k) is not None}
    elapsed = time.perf_counter() - start
    by_order = sorted(c["latencia_media_ms"] for c in per_case)  # latencia media por decision en cada pedido
    item["sonda"].update(casos=len(per_case), fallos_por_causa=metrics["fallos_por_causa"], tiempo_s=round(elapsed, 3),
                         latencia_p95_ms=round(by_order[min(len(by_order) - 1, int(len(by_order) * 0.95))], 4),
                         ms_por_pedido=round(elapsed * 1000 / len(per_case), 2))
    alerts = check(metrics, registered, req)
    return {**item, "estado": "alerta" if alerts else "ok", "alertas": alerts}


@app.get("/api/monitoreo")
def monitoring(casos: int = Query(50, ge=5, le=200, description="pedidos de la suite que ejecuta la sonda por nivel")):
    req = requirements()
    catalog = read_json("catalogo.json", default={}) or {}
    result = [probe_level(level, catalog, casos, req) for level in niveles.level_ids()]
    return {"generado": _now(), "git_commit": os.environ.get("VERCEL_GIT_COMMIT_SHA", "local"),
            "api_version": API_VERSION, "environment_versions": niveles.env_versions(), "requisitos": req,
            "tolerancia_deriva": DRIFT_TOLERANCE, "casos_sonda": casos, "catalogo_actualizado": catalog.get("actualizado"),
            "resumen": {e: sum(r["estado"] == e for r in result) for e in ("ok", "alerta", "error", "sin_modelo")},
            "niveles": result, "instancia": instance_stats()}


class EvaluationRequest(BaseModel):
    nivel: str = "n2"
    politica: Literal["aleatoria", "regla", "modelo"] = "modelo"
    model_id: Optional[str] = None
    casos: int = Field(100, ge=1, le=MAX_EVAL_CASES)


@app.post("/api/evaluar")
def evaluation(req: EvaluationRequest):
    cfg, family, cases, suite_id = level_bundle(req.nivel)
    policy, model = resolve_policy(req.politica, req.model_id, req.nivel, family, seed=0)
    start = time.perf_counter()
    metrics, per_case = evaluate(niveles.make_env(cfg), policy, cases[:req.casos])
    elapsed = time.perf_counter() - start
    return {"modo": "evaluacion", "ambiente": environment_info(cfg, family, suite_id), "modelo": model,
            "politica": req.politica, "metricas": metrics, "por_caso": per_case,
            "suite_completa": req.casos >= len(cases), "tiempo_servidor_s": round(elapsed, 3)}
