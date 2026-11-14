"""Utilidades compartidas por las etapas del pipeline."""

import copy
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warehouse_rush import niveles  # noqa: E402


def parse_value(text):
    """Valor de un parametro clave=valor de la linea de comandos. Prueba int y float antes que YAML, porque
    YAML 1.1 lee la notacion cientifica sin punto ("5e-05") como texto."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_params(params):
    """Configuracion completa de un entrenamiento: nivel (ambiente y datos) + agente + entrenamiento."""
    level = niveles.resolve(params["nivel"])
    return {
        "nivel": params["nivel"],
        "ambiente": level["ambiente"],
        "version_ambiente": level["version_ambiente"],
        "entorno": level["entorno"],
        "datos": level["datos"],
        "agente": copy.deepcopy(params["agente"]),
        "entrenamiento": copy.deepcopy(params["entrenamiento"]),
    }


def make_env(cfg):
    return niveles.make_env(cfg)


def data_path(level, name):
    return f"data/{level}/pedidos_{name}.json"


def load_orders(path):
    with open(path) as f:
        return json.load(f)["pedidos"]


def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
