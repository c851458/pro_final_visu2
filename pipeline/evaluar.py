"""Etapa 3: evalua el modelo (sin exploracion) en los pedidos de evaluacion.

Salidas: metrics/evaluacion.json (metricas), data/trayectorias_evaluacion.csv (cada paso de cada pedido)
"""

import csv
import os

import numpy as np

from common import load_orders, load_params, make_env, write_json
from warehouse_rush import QLearningAgent
from warehouse_rush.env_v1 import ACTION_NAMES, TYPE_NAMES


def main():
    params = load_params()
    orders = load_orders("data/pedidos_evaluacion.json")
    env = make_env(params)
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n)
    agent.load("models/q_table_v1.pkl")

    episodes, steps_log = [], []
    for i, order in enumerate(orders):
        obs, _ = env.reset(seed=i, options={"pedido": order})
        base = env.unwrapped
        events, done = [], False
        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            events.append(base.event)
            steps_log.append({"pedido": i, "paso": base.steps, "fila": base.agent_pos[0], "columna": base.agent_pos[1],
                              "bateria": base.battery, "lleva_paquete": int(base.carrying),
                              "paquete_actual": min(base.current, base.n_packages - 1),
                              "tipo": TYPE_NAMES[base.package["type"]], "accion": ACTION_NAMES[action],
                              "evento": base.event, "recompensa": round(reward, 2)})
        episodes.append({"exito": base.success, "pasos": base.steps, "recompensa": base.total_reward,
                         "rescates": base.rescues, "entregados": base.current,
                         "rotos": sum(p["broken"] for p in base.order), "cargas": events.count("charge"),
                         "choques": events.count("bump") + events.count("break")})

    mean = lambda key: float(np.mean([e[key] for e in episodes]))
    successes = [e for e in episodes if e["exito"]]
    metrics = {
        "tasa_exito": mean("exito"),
        "paquetes_entregados_medios": mean("entregados"),
        "pasos_medios": mean("pasos"),
        "pasos_medios_exitosos": float(np.mean([e["pasos"] for e in successes])) if successes else None,
        "recompensa_media": mean("recompensa"),
        "rescates_medios": mean("rescates"),
        "fragiles_rotos_medios": mean("rotos"),
        "cargas_medias": mean("cargas"),
        "choques_medios": mean("choques"),
        "pedidos_evaluados": len(episodes),
    }
    write_json("metrics/evaluacion.json", metrics)
    os.makedirs("data", exist_ok=True)
    with open("data/trayectorias_evaluacion.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(steps_log[0]))
        writer.writeheader()
        writer.writerows(steps_log)
    for k, v in metrics.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
