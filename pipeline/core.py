"""Logica del pipeline (generar datos, entrenar, evaluar) como funciones reutilizables.

La usan las etapas de DVC (generar_datos.py, entrenar.py, evaluar.py) y experimentar.py.
"""

import numpy as np

from warehouse_rush import QLearningAgent
from warehouse_rush.env_v1 import ACTION_NAMES, LAYOUT, SHELF, TYPE_NAMES, random_order


def _order_key(order):
    return tuple((p["shelf"], p["zone"], p["type"]) for p in order["paquetes"]) + (order["bateria_inicial"],)


def generate_orders(datos):
    """Devuelve (pedidos_entrenamiento, pedidos_evaluacion), sin pedidos repetidos entre ambos."""
    rng = np.random.default_rng(datos["semilla"])
    n_shelves = sum(row.count(SHELF) for row in LAYOUT)
    probs = [datos["prob_tipos"][name] for name in ("normal", "pesado", "fragil")]
    assert abs(sum(probs) - 1) < 1e-9, "prob_tipos debe sumar 1"

    def generate(n, seen):
        orders = []
        while len(orders) < n:
            order = random_order(rng, datos["paquetes_por_pedido"], n_shelves, probs)
            if _order_key(order) not in seen:
                seen.add(_order_key(order))
                orders.append(order)
        return orders

    seen = set()
    evaluation = generate(datos["pedidos_evaluacion"], seen)
    training = generate(datos["pedidos_entrenamiento"], seen)
    return training, evaluation


def train(env, params, orders, on_checkpoint=None):
    """Entrena con Q-learning recorriendo los pedidos. Devuelve (agente, curva de aprendizaje)."""
    a, t = params["agente"], params["entrenamiento"]
    episodes = t["episodios"]
    decay = a["epsilon_min"] ** (1 / max(1, a["fraccion_exploracion"] * episodes))
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n, alpha=a["alpha"], gamma=a["gamma"],
                           epsilon_min=a["epsilon_min"], epsilon_decay=decay, seed=t["semilla"])
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
            curve.append(row)
            window = []
            if on_checkpoint:
                on_checkpoint(row)
    return agent, curve


def evaluate(env, agent, orders):
    """Evalua sin exploracion. Devuelve (metricas, trayectorias paso a paso)."""
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
    successes = [e["pasos"] for e in episodes if e["exito"]]
    metrics = {
        "tasa_exito": mean("exito"),
        "paquetes_entregados_medios": mean("entregados"),
        "pasos_medios": mean("pasos"),
        "pasos_medios_exitosos": float(np.mean(successes)) if successes else 0.0,
        "recompensa_media": mean("recompensa"),
        "rescates_medios": mean("rescates"),
        "fragiles_rotos_medios": mean("rotos"),
        "cargas_medias": mean("cargas"),
        "choques_medios": mean("choques"),
        "pedidos_evaluados": len(episodes),
    }
    return metrics, steps_log
