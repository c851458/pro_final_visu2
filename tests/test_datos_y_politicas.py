"""Pruebas del dataset de pedidos, la compatibilidad de modelos y las politicas de referencia."""

import json
import os

import numpy as np
import pytest

from warehouse_rush import niveles
from warehouse_rush.politicas import QTablePolicy, RandomPolicy, RulePolicy
from warehouse_rush.simulacion import evaluate, run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("level", ["n1", "n2", "n3"])
def test_suite_determinista_y_sin_repetidos(level):
    cfg = niveles.resolve(level)
    cases_a, id_a = niveles.evaluation_suite(cfg)
    cases_b, id_b = niveles.evaluation_suite(cfg)
    assert cases_a == cases_b and id_a == id_b
    env = niveles.make_env(cfg)
    training, evaluation = niveles.generate_orders(cfg["datos"], env.unwrapped.reachable_shelves)
    key = lambda o: json.dumps(o, sort_keys=True)
    assert not {key(o) for o in training} & {key(o) for o in evaluation}
    assert evaluation == cases_a
    reachable = set(env.unwrapped.reachable_shelves)
    assert all(p["shelf"] in reachable for o in evaluation for p in o["paquetes"])


@pytest.mark.parametrize("level", ["n1", "n2", "n3"])
def test_suite_igual_al_dataset_versionado_con_dvc(level):
    path = os.path.join(ROOT, "data", level, "pedidos_evaluacion.json")
    if not os.path.exists(path):
        pytest.skip("dataset no descargado (dvc pull)")
    with open(path) as f:
        stored = json.load(f)["pedidos"]
    cases, _ = niveles.evaluation_suite(niveles.resolve(level))
    assert cases == stored


def test_familias_de_modelos():
    families = {lv: niveles.family(niveles.make_env(niveles.resolve(lv))) for lv in ("n1", "n2", "n3")}
    assert families["n2"] == families["n3"]  # el incidente no cambia el contrato de observaciones
    assert families["n1"] != families["n2"]  # sin bateria: otra familia de modelos


@pytest.mark.parametrize("level", ["n1", "n2", "n3"])
def test_politica_de_regla_completa_los_pedidos(level):
    cfg = niveles.resolve(level)
    cases, _ = niveles.evaluation_suite(cfg)
    metrics, _ = evaluate(niveles.make_env(cfg), RulePolicy(), cases[:20])
    assert metrics["tasa_exito"] == 1.0


def test_politica_aleatoria_reproducible():
    cfg = niveles.resolve("n1")
    cases, _ = niveles.evaluation_suite(cfg)
    a, _ = run_episode(niveles.make_env(cfg), RandomPolicy(seed=3), cases[0], record=False)
    b, _ = run_episode(niveles.make_env(cfg), RandomPolicy(seed=3), cases[0], record=False)
    assert a["pasos"] == b["pasos"] and a["recompensa"] == b["recompensa"]


def test_politica_q_usa_la_accion_de_mayor_valor():
    env = niveles.make_env(niveles.resolve("n1"))
    q = np.zeros(tuple(env.observation_space.nvec) + (env.action_space.n,), dtype=np.float32)
    obs, _ = env.reset(seed=0)
    q[tuple(obs)][4] = 1.0
    assert QTablePolicy(q).act(env, obs) == 4
