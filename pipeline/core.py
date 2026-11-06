"""Entrenamiento del agente Q-learning, compartido por el pipeline de DVC y experimentar.py.

La generacion de datos esta en warehouse_rush/niveles.py y la evaluacion en warehouse_rush/simulacion.py,
que tambien usa la API desplegada: asi todos evaluan con exactamente el mismo codigo.
"""

import numpy as np

from warehouse_rush import QLearningAgent
from warehouse_rush.politicas import QTablePolicy
from warehouse_rush.simulacion import evaluate


def train(env, cfg, orders, curve_cases=(), on_checkpoint=None, initial_q=None):
    """Entrena con Q-learning recorriendo los pedidos. Devuelve (agente, curva).

    En cada registro la curva incluye las metricas del entrenamiento (con exploracion) y una evaluacion
    sin exploracion sobre `curve_cases` (curva de evaluacion, separada como pide el enunciado).
    `initial_q` continua el entrenamiento desde un modelo anterior en vez de empezar desde cero.
    """
    a, t = cfg["agente"], cfg["entrenamiento"]
    episodes = t["episodios"]
    decay = a["epsilon_min"] ** (1 / max(1, a["fraccion_exploracion"] * episodes))
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n, alpha=a["alpha"], gamma=a["gamma"],
                           epsilon_min=a["epsilon_min"], epsilon_decay=decay, seed=t["semilla"])
    if initial_q is not None:
        if initial_q.shape != agent.q.shape:
            raise ValueError(f"El modelo inicial tiene forma {initial_q.shape} y el nivel espera {agent.q.shape}")
        agent.q = initial_q.copy()
        agent.epsilon = a.get("epsilon_inicial", 1.0)

    eval_env = type(env.unwrapped)(**_env_kwargs(env)) if curve_cases else None
    curve, window = [], []
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
            row = {"episodio": ep + 1, "tasa_exito": success, "recompensa_media": reward,
                   "rescates_medios": rescues, "pasos_medios": steps, "epsilon": agent.epsilon}
            if curve_cases:
                metrics, _ = evaluate(eval_env, QTablePolicy(agent.q), curve_cases)
                row.update({"eval_tasa_exito": metrics["tasa_exito"], "eval_recompensa_media": metrics["recompensa_media"],
                            "eval_pasos_medios": metrics["pasos_medios"]})
            curve.append(row)
            window = []
            if on_checkpoint:
                on_checkpoint(row)
    return agent, curve


def _env_kwargs(env):
    base = env.unwrapped
    return {"n_packages": base.n_packages, "use_battery": base.use_battery, "max_steps": base.max_steps,
            "battery_max": base.battery_max, "blocked_cells": base.blocked_cells}
