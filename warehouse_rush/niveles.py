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

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v1)
from warehouse_rush.env_v1 import random_order

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "niveles.yaml")
TYPE_ORDER = ("normal", "pesado", "fragil")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def env_version():
    return load_config()["version_ambiente"]


def level_ids():
    return list(load_config()["niveles"])


def resolve(level_id):
    """Configuracion completa de un nivel: {"nivel", "nombre", "decision", "entorno", "datos"}."""
    levels = load_config()["niveles"]
    if level_id not in levels:
        raise KeyError(f"Nivel desconocido: {level_id} (disponibles: {', '.join(levels)})")
    return dict(copy.deepcopy(levels[level_id]), nivel=level_id)


def make_env(cfg):
    """Ambiente de un nivel resuelto (o de cualquier config con las secciones entorno y datos)."""
    e, d = cfg["entorno"], cfg["datos"]
    return gym.make("WarehouseRush-v1", n_packages=d["paquetes_por_pedido"], use_battery=e["bateria"],
                    battery_max=e["bateria_max"], max_steps=e["max_pasos"],
                    blocked_cells=[tuple(c) for c in e.get("bloqueos", [])])


def family(env):
    """Identificador de la familia de modelos compatibles: dimensiones de observacion y acciones."""
    base = env.unwrapped
    dims = "x".join(str(int(n)) for n in base.observation_space.nvec)
    return f"warehouse-rush-v1/obs-{dims}/acciones-{base.action_space.n}"


def sha256_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _order_key(order):
    return tuple((p["shelf"], p["zone"], p["type"]) for p in order["paquetes"]) + (order["bateria_inicial"],)


def generate_orders(datos, shelves, evaluation_only=False):
    """Genera (pedidos_entrenamiento, pedidos_evaluacion) sin repetidos entre ambos.

    Los de evaluacion se generan primero, asi la suite de evaluacion no depende del tamano del entrenamiento.
    """
    rng = np.random.default_rng(datos["semilla"])
    probs = [datos["prob_tipos"][name] for name in TYPE_ORDER]
    if abs(sum(probs) - 1) > 1e-9:
        raise ValueError("prob_tipos debe sumar 1")

    def generate(n, seen):
        orders = []
        while len(orders) < n:
            order = random_order(rng, datos["paquetes_por_pedido"], shelves, probs)
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
    cases = []
    while len(cases) < n_cases:
        order = random_order(rng, cfg["datos"]["paquetes_por_pedido"], shelves, probs)
        if _order_key(order) not in seen:
            seen.add(_order_key(order))
            cases.append(order)
    return cases, f"{cfg['nivel']}-reservado-{sha256_json(cases)[:12]}"
