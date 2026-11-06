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

Variables de entorno: HF_TOKEN (solo si el repositorio de modelos es privado), CORS_ORIGINS,
MANIFESTS_DIR (solo para pruebas locales).
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from warehouse_rush import niveles  # noqa: E402
from warehouse_rush.env_v1 import ACTION_NAMES, LAYOUT, TYPE_NAMES, ZONE_NAMES  # noqa: E402
from warehouse_rush.politicas import QTablePolicy, RandomPolicy, RulePolicy  # noqa: E402
from warehouse_rush.simulacion import evaluate, replay_manual, run_episode  # noqa: E402

API_VERSION = "1.0.0"
MANIFESTS = os.environ.get("MANIFESTS_DIR", os.path.join(ROOT, "manifests"))
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/warehouse-rush-modelos")
MAX_EVAL_CASES = 500

app = FastAPI(title="Warehouse Rush API", version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


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


def manifest(model_id):
    data = read_json("modelos", f"{model_id}.json")
    if data is None:
        raise HTTPException(404, f"Modelo desconocido: {model_id}")
    return data


@lru_cache(maxsize=16)
def load_weights(model_id):
    """Descarga los pesos (con cache en /tmp) y verifica su SHA-256 contra el manifiesto."""
    info = manifest(model_id)
    path = os.path.join(CACHE_DIR, f"{info['model_sha256']}.npy")
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
    return np.load(path, allow_pickle=False)


def resolve_policy(policy, model_id, level, family, seed):
    """Devuelve (politica, descripcion del modelo). Valida la compatibilidad modelo-ambiente."""
    if policy == "aleatoria":
        return RandomPolicy(seed=seed), None
    if policy == "regla":
        return RulePolicy(), None
    if policy != "modelo":
        raise HTTPException(400, f"Politica desconocida: {policy}")
    model_id = model_id or (read_json("catalogo.json", default={}) or {}).get("recomendado")
    if not model_id:
        raise HTTPException(404, "No hay un modelo recomendado; elige uno del catalogo")
    info = manifest(model_id)
    if info["familia_modelo"] != family:
        raise HTTPException(409, f"El modelo {model_id} ({info['familia_modelo']}) no es compatible con el nivel "
                                 f"{level} ({family}): sus observaciones tienen otras dimensiones")
    weights = load_weights(model_id)
    return QTablePolicy(weights), {
        "model_id": model_id, "model_sha256": info["model_sha256"], "nivel_entrenamiento": info["level_id"],
        "environment_version": info["environment_version"], "run_name": info["run_name"],
        "hf_commit": info["hf_commit"], "git_commit": info["git_commit"]}


def environment_info(cfg, family, suite_id):
    return {"nivel": cfg["nivel"], "nombre": cfg["nombre"], "environment_version": niveles.env_version(),
            "familia_modelo": family, "evaluation_suite_id": suite_id}


# ---------------------------------------------------------------- endpoints

@app.get("/api/salud")
def health():
    catalog = read_json("catalogo.json", default={}) or {}
    return {"estado": "ok", "api_version": API_VERSION, "environment_version": niveles.env_version(),
            "git_commit": os.environ.get("VERCEL_GIT_COMMIT_SHA", "local"),
            "despliegue": os.environ.get("VERCEL_URL", "local"), "modelo_recomendado": catalog.get("recomendado")}


@app.get("/api/niveles")
def levels():
    result = []
    for level in niveles.level_ids():
        cfg, family, cases, suite_id = level_bundle(level)
        env = niveles.make_env(cfg).unwrapped
        result.append({
            **environment_info(cfg, family, suite_id), "decision": cfg["decision"], "entorno": cfg["entorno"],
            "datos": cfg["datos"], "casos_evaluacion": len(cases),
            "mapa": {"layout": LAYOUT, "shelf_cells": [list(p) for p in env.shelf_cells],
                     "zone_cells": [sorted(list(p) for p in z) for z in env.zone_cells], "charger": list(env.charger),
                     "start": list(env.start_pos), "bloqueos": [list(c) for c in env.blocked_cells]},
        })
    return {"niveles": result, "acciones": ACTION_NAMES, "tipos": TYPE_NAMES, "zonas": ZONE_NAMES}


@app.get("/api/catalogo")
def catalog():
    return {"catalogo": read_json("catalogo.json", default={"recomendado": None, "modelos": []}),
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
