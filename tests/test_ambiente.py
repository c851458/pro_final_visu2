"""Pruebas de las reglas del ambiente: observaciones, acciones, recompensas y terminaciones."""

import pytest

from warehouse_rush import niveles
from warehouse_rush.env_v1 import (
    CHARGE, DOWN, DROP, FRAGILE, HEAVY, LEFT, NORMAL, PICK, RIGHT, UP,
    COLLISION_PENALTY, DELIVERY_REWARD, INVALID_ACTION_PENALTY, ORDER_BONUS, PICK_REWARD, RESCUE_PENALTY,
    STEP_PENALTY, WarehouseRushV1Env,
)
from warehouse_rush.politicas import RulePolicy

SHELF_2_2 = 0  # estanteria en (2, 2): se recoge desde (1, 2) o (2, 1)


def order(*packages, battery=1.0):
    return {"paquetes": [{"shelf": s, "zone": z, "type": t} for s, z, t in packages], "bateria_inicial": battery}


def make(order_, **kwargs):
    env = WarehouseRushV1Env(**kwargs)
    obs, _ = env.reset(seed=0, options={"pedido": order_})
    return env, obs


def walk(env, *actions):
    for action in actions:
        result = env.step(action)
    return result


def go_to(env, goals):
    """Lleva el robot a una de las casillas meta usando el planificador BFS de la politica de regla."""
    rule = RulePolicy()
    while env.agent_pos not in goals:
        env.step(rule._step_towards(env, goals))


@pytest.mark.parametrize("level", ["n1", "n2", "n3"])
def test_observacion_dentro_del_espacio(level):
    env = niveles.make_env(niveles.resolve(level))
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for action in range(env.action_space.n):
        obs, *_ = env.step(action)
        assert env.observation_space.contains(obs)


def test_mover_y_chocar_con_pared():
    env, _ = make(order((SHELF_2_2, 0, NORMAL)))
    _, reward, *_ = env.step(UP)
    assert env.agent_pos == (7, 7) and reward == pytest.approx(STEP_PENALTY)
    env.agent_pos = (1, 1)
    _, reward, *_ = env.step(LEFT)
    assert env.agent_pos == (1, 1) and env.event == "bump"
    assert reward == pytest.approx(STEP_PENALTY + COLLISION_PENALTY)


def test_recoger_solo_junto_a_la_estanteria_correcta():
    env, _ = make(order((SHELF_2_2, 0, NORMAL)))
    _, reward, *_ = env.step(PICK)
    assert not env.carrying and env.event == "pick_fail"
    assert reward == pytest.approx(STEP_PENALTY + INVALID_ACTION_PENALTY)
    go_to(env, {(1, 2)})
    obs, reward, *_ = env.step(PICK)
    assert env.carrying and reward == pytest.approx(STEP_PENALTY + PICK_REWARD)
    assert obs[3] == 1 and obs[5] == 0  # lleva paquete; objetivo = zona A


def test_entrega_en_zona_correcta_y_bono_de_pedido():
    env, _ = make(order((SHELF_2_2, 0, NORMAL), (1, 2, NORMAL)))
    go_to(env, {(1, 2)})
    env.step(PICK)
    go_to(env, env.zone_cells[1])
    _, reward, *_ = env.step(DROP)
    assert env.event == "wrong_zone" and env.carrying
    go_to(env, env.zone_cells[0])
    _, reward, terminated, *_ = env.step(DROP)
    assert reward == pytest.approx(STEP_PENALTY + DELIVERY_REWARD) and not terminated and env.current == 1
    go_to(env, {(1, 3)})
    env.step(PICK)
    go_to(env, env.zone_cells[2])
    _, reward, terminated, truncated, _ = env.step(DROP)
    assert terminated and not truncated and env.success
    assert reward == pytest.approx(STEP_PENALTY + DELIVERY_REWARD + ORDER_BONUS)


def test_fragil_se_rompe_al_chocar_y_vuelve_a_la_estanteria():
    env, _ = make(order((SHELF_2_2, 0, FRAGILE)))
    go_to(env, {(1, 2)})
    env.step(PICK)
    _, reward, *_ = env.step(UP)  # pared
    assert env.event == "break" and not env.carrying and env.order[0]["broken"] == 1
    assert env.order[0]["status"] == "pendiente"


def test_pesado_gasta_doble_bateria():
    env, _ = make(order((SHELF_2_2, 0, HEAVY)))
    go_to(env, {(1, 2)})
    env.step(PICK)
    before = env.battery
    env.step(RIGHT)  # de (1, 2) a (1, 3): piso libre
    assert env.event == "move" and before - env.battery == 2


def test_cargar_solo_en_la_estacion():
    env, _ = make(order((SHELF_2_2, 0, NORMAL), battery=0.5))
    env.step(CHARGE)
    assert env.event == "charge_fail"
    go_to(env, {env.charger})
    env.step(CHARGE)
    assert env.event == "charge" and env.battery == env.battery_max


def test_bateria_agotada_provoca_rescate():
    env, _ = make(order((SHELF_2_2, 0, NORMAL), battery=0.01))
    _, reward, terminated, *_ = env.step(UP)
    assert env.event == "rescue" and env.agent_pos == env.charger and env.battery == env.battery_max
    assert reward == pytest.approx(STEP_PENALTY + RESCUE_PENALTY) and not terminated


def test_truncado_al_agotar_los_pasos():
    env, _ = make(order((SHELF_2_2, 0, NORMAL)), max_steps=5)
    *_, terminated, truncated, _ = walk(env, LEFT, RIGHT, LEFT, RIGHT, LEFT)
    assert truncated and not terminated


def test_bloqueos_del_nivel_n3_son_obstaculos():
    env = niveles.make_env(niveles.resolve("n3")).unwrapped
    env.reset(seed=0)
    env.agent_pos = (8, 7)
    env.step(UP)  # (7, 7) esta bloqueada
    assert env.agent_pos == (8, 7) and env.event == "bump"
    with pytest.raises(ValueError):
        WarehouseRushV1Env(blocked_cells=[(2, 2)])  # una estanteria no se puede bloquear
