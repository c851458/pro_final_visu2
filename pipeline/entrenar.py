"""Etapa 2: entrena el agente Q-learning recorriendo los pedidos de entrenamiento.

Salidas: models/q_table_v1.pkl (modelo), metrics/entrenamiento.csv (curva de aprendizaje)
"""

import csv
import os

import numpy as np

from common import load_orders, load_params, make_env
from warehouse_rush import QLearningAgent


def main():
    params = load_params()
    a, t = params["agente"], params["entrenamiento"]
    orders = load_orders("data/pedidos_entrenamiento.json")
    env = make_env(params)
    episodes = t["episodios"]
    decay = a["epsilon_min"] ** (1 / max(1, a["fraccion_exploracion"] * episodes))
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n, alpha=a["alpha"], gamma=a["gamma"],
                           epsilon_min=a["epsilon_min"], epsilon_decay=decay, seed=t["semilla"])

    rows, window = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=t["semilla"] + ep, options={"pedido": orders[ep % len(orders)]})
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)
            obs, done = next_obs, terminated or truncated
        agent.decay_epsilon()
        base = env.unwrapped
        window.append((base.success, base.total_reward, base.rescues, base.steps))

        if (ep + 1) % t["registrar_cada"] == 0:
            success, reward, rescues, steps = (float(np.mean(x)) for x in zip(*window))
            rows.append({"episodio": ep + 1, "tasa_exito": success, "recompensa_media": reward,
                         "rescates_medios": rescues, "pasos_medios": steps, "epsilon": agent.epsilon})
            print(f"Episodio {ep + 1:6d} | exito {success * 100:5.1f}% | recompensa {reward:7.2f} | "
                  f"rescates {rescues:.2f} | epsilon {agent.epsilon:.3f}", flush=True)
            window = []

    os.makedirs("models", exist_ok=True)
    agent.save("models/q_table_v1.pkl")
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/entrenamiento.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
