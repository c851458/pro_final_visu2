"""Utilidades compartidas por las etapas del pipeline."""

import json
import os
import sys

import gymnasium as gym
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warehouse_rush  # noqa: E402,F401  (registra WarehouseRush-v1)


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(params):
    e = params["entorno"]
    return gym.make("WarehouseRush-v1", n_packages=params["datos"]["paquetes_por_pedido"],
                    use_battery=e["bateria"], battery_max=e["bateria_max"], max_steps=e["max_pasos"])


def load_orders(path):
    with open(path) as f:
        return json.load(f)["pedidos"]


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
