"""Niveles de dificultad (configs/niveles.yaml): construccion del ambiente, dataset y compatibilidad de modelos.

Lo usan el pipeline, los experimentos, las pruebas y la API, para que todos generen exactamente
los mismos ambientes y casos de evaluacion.
"""

import copy
import hashlib
import json
import os

import gymnasium as gym
import numpy as np
import yaml

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v1 y -v2)
from warehouse_rush.env_v1 import random_order
from warehouse_rush.env_v2 import random_order_v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "niveles.yaml")
TYPE_ORDER = ("normal", "pesado", "fragil")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def env_version(level_id=None):
    """Version del ambiente de un nivel (por defecto, la de los niveles v1)."""
    if level_id is None:
        return load_config()["version_ambiente"]
    return resolve(level_id)["version_ambiente"]


def env_versions():
    """Version de ambiente de cada nivel: {nivel: version}."""
    return {lv: resolve(lv)["version_ambiente"] for lv in level_ids()}


def level_ids():
    return list(load_config()["niveles"])


def resolve(level_id):
    """Configuracion completa de un nivel: {"nivel", "nombre", "decision", "entorno", "datos"}."""
    levels = load_config()["niveles"]
    if level_id not in levels:
        raise KeyError(f"Nivel desconocido: {level_id} (disponibles: {', '.join(levels)})")
    cfg = dict(copy.deepcopy(levels[level_id]), nivel=level_id)
    cfg.setdefault("ambiente", "WarehouseRush-v1")
    cfg.setdefault("version_ambiente", load_config()["version_ambiente"])
    return cfg


def make_env(cfg):
    """Ambiente de un nivel resuelto (o de cualquier config con las secciones entorno y datos)."""
    e, d = cfg["entorno"], cfg["datos"]
    env_id = cfg.get("ambiente", "WarehouseRush-v1")
    extra = {"express_prob": d.get("prob_express", 0.0)} if env_id == "WarehouseRush-v2" else {}
    return gym.make(env_id, n_packages=d["paquetes_por_pedido"], use_battery=e["bateria"],
                    battery_max=e["bateria_max"], max_steps=e["max_pasos"],
                    blocked_cells=[tuple(c) for c in e.get("bloqueos", [])], **extra)


def family(env):
    """Identificador de la familia de modelos compatibles: dimensiones de observacion y acciones."""
    base = env.unwrapped
    dims = "x".join(str(int(n)) for n in base.observation_space.nvec)
    return f"{base.FAMILY_PREFIX}/obs-{dims}/acciones-{base.action_space.n}"


def sha256_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _order_key(order):
    return tuple((p["shelf"], p["zone"], p["type"], p.get("service")) for p in order["paquetes"]) + (order["bateria_inicial"],)


def _order_fn(datos):
    """Generador de pedidos: v2 agrega el servicio (Estandar/Express) si el nivel define prob_express."""
    if "prob_express" in datos:
        return lambda rng, n, shelves, probs: random_order_v2(rng, n, shelves, probs, datos["prob_express"])
    return random_order


def generate_orders(datos, shelves, evaluation_only=False):
    """Genera (pedidos_entrenamiento, pedidos_evaluacion) sin repetidos entre ambos.

    Los de evaluacion se generan primero, asi la suite de evaluacion no depende del tamano del entrenamiento.
    """
    rng = np.random.default_rng(datos["semilla"])
    probs = [datos["prob_tipos"][name] for name in TYPE_ORDER]
    if abs(sum(probs) - 1) > 1e-9:
        raise ValueError("prob_tipos debe sumar 1")

    make_order = _order_fn(datos)

    def generate(n, seen):
        orders = []
        while len(orders) < n:
            order = make_order(rng, datos["paquetes_por_pedido"], shelves, probs)
            if _order_key(order) not in seen:
                seen.add(_order_key(order))
                orders.append(order)
        return orders

    seen = set()
    evaluation = generate(datos["pedidos_evaluacion"], seen)
    training = [] if evaluation_only else generate(datos["pedidos_entrenamiento"], seen)
    return training, evaluation


def evaluation_suite(cfg):
    """Suite de evaluacion del nivel: (casos, suite_id). El id es el hash de los casos."""
    env = make_env(cfg)
    _, cases = generate_orders(cfg["datos"], env.unwrapped.reachable_shelves, evaluation_only=True)
    return cases, f"{cfg['nivel']}-{sha256_json(cases)[:12]}"


def reserved_suite(cfg, seed, n_cases):
    """Test final reservado: pedidos del nivel generados con una semilla secreta, sin repetir ningun pedido
    de entrenamiento ni de la suite de evaluacion publica. Devuelve (casos, suite_id)."""
    env = make_env(cfg)
    shelves = env.unwrapped.reachable_shelves
    training, evaluation = generate_orders(cfg["datos"], shelves)
    seen = {_order_key(o) for o in training + evaluation}
    rng = np.random.default_rng(seed)
    probs = [cfg["datos"]["prob_tipos"][name] for name in TYPE_ORDER]
    make_order = _order_fn(cfg["datos"])
    cases = []
    while len(cases) < n_cases:
        order = make_order(rng, cfg["datos"]["paquetes_por_pedido"], shelves, probs)
        if _order_key(order) not in seen:
            seen.add(_order_key(order))
            cases.append(order)
    return cases, f"{cfg['nivel']}-reservado-{sha256_json(cases)[:12]}"
