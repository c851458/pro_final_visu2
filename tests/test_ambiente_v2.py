"""Pruebas del ambiente v2 (negocio): recompensa en COP y SLA Express/Estandar."""

import pytest

from warehouse_rush import niveles
from warehouse_rush.env_v1 import CHARGE, FRAGILE, NORMAL, PICK, DROP, UP
from warehouse_rush.env_v2 import COSTOS, EXPRESS, SLA_STEPS, STANDARD, WarehouseRushV2Env
from warehouse_rush.politicas import RulePolicy

C = COSTOS


def order(*packages, battery=1.0):
    return {"paquetes": [{"shelf": s, "zone": z, "type": t, "service": sv} for s, z, t, sv in packages],
            "bateria_inicial": battery}


def make(order_, **kwargs):
    env = WarehouseRushV2Env(**kwargs)
    obs, _ = env.reset(seed=0, options={"pedido": order_})
    return env, obs


def go_to(env, goals):
    rule = RulePolicy()
    while env.agent_pos not in goals:
        env.step(rule._step_towards(env, goals))


def test_observacion_incluye_el_servicio():
    env, obs = make(order((0, 0, NORMAL, EXPRESS)))
    assert len(obs) == 7 and obs[6] == EXPRESS and env.observation_space.contains(obs)
    assert niveles.family(env).startswith("warehouse-rush-v2/")


def test_entrega_estandar_a_tiempo_cobra_la_tarifa():
    env, _ = make(order((0, 0, NORMAL, STANDARD), (1, 1, NORMAL, STANDARD)))
    go_to(env, {(1, 2)})
    _, r_pick, *_ = env.step(PICK)
    assert r_pick == C["hito_recogida"] - C["costo_paso"]
    go_to(env, env.zone_cells[0])
    _, r_drop, terminated, *_ = env.step(DROP)
    assert not terminated and r_drop == C["tarifa_estandar"] - C["hito_recogida"] - C["costo_paso"]
    assert env.order[0]["late"] is False and env.ledger["ingresos"] == C["tarifa_estandar"]


def test_express_tarde_paga_compensacion():
    env, _ = make(order((0, 0, NORMAL, EXPRESS)))
    for _ in range(SLA_STEPS[EXPRESS]):  # deja vencer el plazo moviendose sin avanzar
        env.step(UP if env.agent_pos[0] > 1 else 1)
    go_to(env, {(1, 2)})
    env.step(PICK)
    go_to(env, env.zone_cells[0])
    _, reward, terminated, *_ = env.step(DROP)
    assert terminated and env.order[0]["late"]
    assert reward == (C["tarifa_express"] - C["hito_recogida"] - C["compensacion_express_tarde"]
                      + C["bono_pedido_completo"] - C["costo_paso"])
    assert env.sla_summary()["pedido_a_tiempo"] is False


def test_hito_de_recogida_una_sola_vez_y_fragil_roto():
    env, _ = make(order((0, 0, FRAGILE, STANDARD)))
    go_to(env, {(1, 2)})
    env.step(PICK)
    _, reward, *_ = env.step(UP)  # pared: se rompe
    assert env.event == "break" and reward == -(C["costo_paso"] + C["costo_choque"] + C["indemnizacion_fragil"])
    _, reward, *_ = env.step(PICK)
    assert env.carrying and reward == -C["costo_paso"]  # el hito ya se habia premiado


def test_cargar_cuesta_la_energia_recargada():
    env, _ = make(order((0, 0, NORMAL, STANDARD), battery=0.5))
    go_to(env, {env.charger})
    battery = env.battery
    _, reward, *_ = env.step(CHARGE)
    expected = round(C["costo_carga_completa"] * (env.battery_max - battery) / env.battery_max)
    assert reward == -(C["costo_paso"] + expected) and env.ledger["energia"] == expected


@pytest.mark.parametrize("level", ["v2-n1", "v2-n2", "v2-n3"])
def test_niveles_v2_generan_servicios(level):
    cases, suite_id = niveles.evaluation_suite(niveles.resolve(level))
    services = [p["service"] for o in cases for p in o["paquetes"]]
    assert set(services) == {STANDARD, EXPRESS} and suite_id.startswith(level)
