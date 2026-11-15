"""Pruebas del ambiente v3: dos pisos con montacargas y hora de corte de los camiones."""

import numpy as np
import pytest

from warehouse_rush import niveles
from warehouse_rush.env_v1 import DROP, NORMAL, PICK, RIGHT, UP
from warehouse_rush.env_v2 import EXPRESS, STANDARD
from warehouse_rush.env_v3 import (
    COSTOS, ELEVATOR, SLA_STEPS, TIEMPO_MONTACARGAS, WarehouseRushV3Env, random_order_v3,
)
from warehouse_rush.politicas import RulePolicy
from warehouse_rush.ppo import FEATURE_NAMES_V3, features
from warehouse_rush.simulacion import evaluate

C = COSTOS
UPPER = 40  # la estanteria 40 es la primera del piso 2, en (2, 2) como la 0 del piso 1


def order(*packages, battery=1.0, cutoff=(500, 500, 500)):
    return {"paquetes": [{"shelf": s, "zone": z, "type": t, "service": sv} for s, z, t, sv in packages],
            "bateria_inicial": battery, "corte": list(cutoff)}


def make(order_, **kwargs):
    env = WarehouseRushV3Env(**kwargs)
    obs, _ = env.reset(seed=0, options={"pedido": order_})
    return env, obs


def go_to(env, goals):
    """Lleva el robot a cualquiera de las metas (posiciones (piso, fila, columna)) por el camino mas corto."""
    rule = RulePolicy()
    goals = frozenset(goals)
    for _ in range(200):
        if env.position in goals:
            return
        env.step(rule._step_floors(env, env.position, goals))
    raise AssertionError(f"no llego a {sorted(goals)}")


def test_observacion_familia_y_mapa():
    env, obs = make(order((UPPER, 0, NORMAL, EXPRESS)))
    assert env.action_space.n == 8 and len(obs) == 8 and env.observation_space.contains(obs)
    assert obs[0] == 0 and obs[6] == UPPER and obs[7] == EXPRESS
    assert niveles.family(env) == "warehouse-rush-v3/obs-2x11x15x5x2x3x80x2/acciones-8"
    assert len(env.shelf_cells) == 80 and env.shelf_floor.count(1) == 40
    assert env.shelf_cells[UPPER] == env.shelf_cells[0] and env.elevator == (4, 14)
    assert len(features(env)) == len(FEATURE_NAMES_V3) == 30


def test_montacargas_cambia_de_piso_y_cuesta_tiempo():
    env, _ = make(order((0, 0, NORMAL, STANDARD)))
    _, reward, *_ = env.step(ELEVATOR)  # lejos del montacargas: accion invalida
    assert env.event == "elevator_fail" and env.floor == 0 and reward == -C["costo_paso"] - C["costo_accion_invalida"]
    go_to(env, {(0, 4, 14)})
    steps = env.steps
    _, reward, *_ = env.step(ELEVATOR)
    assert env.event == "elevator" and env.floor == 1 and env.steps == steps + TIEMPO_MONTACARGAS
    assert reward == -C["costo_paso"] * TIEMPO_MONTACARGAS - C["costo_montacargas"]
    env.step(RIGHT)  # el montacargas esta en el muro: a la derecha no hay nada (sin salirse del mapa)
    assert env.event == "bump" and env.position == (1, 4, 14)
    env.step(ELEVATOR)
    assert env.floor == 0


def test_paquete_del_segundo_piso_solo_se_recoge_arriba():
    env, _ = make(order((UPPER, 0, NORMAL, STANDARD)))
    go_to(env, {(0, 1, 2)})  # misma casilla, pero en el piso 1
    env.step(PICK)
    assert env.event == "pick_fail" and not env.carrying
    go_to(env, {(1, 1, 2)})
    env.step(PICK)
    assert env.event == "pick" and env.carrying
    go_to(env, {(1, 9, 2)})  # el piso 2 no tiene muelles
    env.step(DROP)
    assert env.event == "drop_fail"
    go_to(env, {(0, *cell) for cell in env.zone_cells[0]})
    _, _, terminated, *_ = env.step(DROP)
    assert terminated and env.success and env.order[0]["camion_perdido"] is False


def test_perder_el_camion_cobra_compensacion_una_vez_y_reproceso():
    env, _ = make(order((0, 0, NORMAL, EXPRESS), cutoff=(5, 500, 500)))
    go_to(env, {(0, 1, 2)})
    env.step(PICK)
    go_to(env, {(0, *cell) for cell in env.zone_cells[0]})
    assert env.steps < SLA_STEPS[EXPRESS]  # a tiempo para el SLA, pero el camion ya salio
    _, reward, terminated, *_ = env.step(DROP)
    pkg = env.order[0]
    assert terminated and pkg["camion_perdido"] and pkg["late"]
    assert reward == (C["tarifa_express"] - C["hito_recogida"] - C["compensacion_express_tarde"]
                      - C["costo_reproceso_camion"] + C["bono_pedido_completo"] - C["costo_paso"])
    summary = env.sla_summary()
    assert summary["camiones_perdidos"] == 1 and summary["pedido_a_tiempo"] is False
    assert env.state_dict()["camiones"] == ["salio", "esperando", "esperando"]


def test_rescate_vuelve_al_cargador_del_primer_piso():
    env, _ = make(order((UPPER, 0, NORMAL, STANDARD), battery=0.2))
    go_to(env, {(0, 4, 13)})
    env.step(RIGHT)
    env.step(ELEVATOR)
    while env.event != "rescue":
        env.step(UP)
    assert env.position == (0, *env.charger) and env.battery == env.battery_max and env.rescues == 1


def test_bloqueos_por_piso_y_validacion():
    env = WarehouseRushV3Env(blocked_cells=[(0, 7, 4), (1, 7, 7)])
    assert env.grids[0][7, 4] == "X" and env.grids[1][7, 4] == "." and env.grids[1][7, 7] == "X"
    with pytest.raises(ValueError):
        WarehouseRushV3Env(blocked_cells=[(1, 2, 2)])  # una estanteria no se puede bloquear


def test_pedidos_secuenciados_por_la_salida_del_camion():
    rng = np.random.default_rng(0)
    shelves = WarehouseRushV3Env().reachable_shelves
    for _ in range(50):
        o = random_order_v3(rng, 3, shelves, cutoff_range=(120, 360))
        times = [o["corte"][p["zone"]] for p in o["paquetes"]]
        assert times == sorted(times) and all(120 <= t <= 360 for t in o["corte"])


@pytest.mark.parametrize("level", ["v3-n1", "v3-n2", "v3-n3"])
def test_planificador_de_reglas_resuelve_los_niveles_de_dos_pisos(level):
    cfg = niveles.resolve(level)
    cases, suite_id = niveles.evaluation_suite(cfg)
    metrics, per_case = evaluate(niveles.make_env(cfg), RulePolicy(), cases[:40])
    assert suite_id.startswith(level) and metrics["tasa_exito"] == 1.0 and metrics["rescates_medios"] == 0
    assert "camiones_perdidos_medios" in metrics and metrics["viajes_montacargas_medios"] > 0
    assert {"camiones_perdidos", "montacargas"} <= set(per_case[0])
