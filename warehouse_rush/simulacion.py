"""Ejecucion de episodios y evaluaciones: el mismo codigo para el pipeline, las pruebas y la API.

Un fotograma es el estado completo del ambiente tras cada paso (lo que la vista 3D necesita para dibujarlo).
"""

import time

import numpy as np

from warehouse_rush.env_v1 import ACTION_NAMES, TYPE_NAMES

# Umbral para considerar que un episodio fallido se debio a choques repetidos
MANY_BUMPS = 20


def frame(env, action=None, reward=0.0, terminated=False, truncated=False):
    data = env.unwrapped.state_dict()
    data.update(action=None if action is None else int(action), reward=float(reward),
                terminated=bool(terminated), truncated=bool(truncated))
    return data


def failure_cause(result):
    """Causa principal de un episodio sin exito (todos terminan por tiempo agotado: truncated)."""
    if result["exito"]:
        return "exito"
    if result["rotos"]:
        return "fragil_roto"
    if result["rescates"]:
        return "bateria_agotada"
    if result["choques"] >= MANY_BUMPS:
        return "choques_repetidos"
    return "tiempo_agotado"


def run_episode(env, policy, order, seed=0, record=True, trajectory=None):
    """Ejecuta un pedido con una politica. Devuelve (resultado, fotogramas)."""
    obs, _ = env.reset(seed=seed, options={"pedido": order})
    base = env.unwrapped
    frames = [frame(env)] if record else None
    events, latencies, done = [], [], False
    while not done:
        t0 = time.perf_counter()
        action = policy.act(env, obs)
        latencies.append((time.perf_counter() - t0) * 1000)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        events.append(base.event)
        if record:
            frames.append(frame(env, action, reward, terminated, truncated))
        if trajectory is not None:
            trajectory.append({"paso": base.steps, "fila": base.agent_pos[0], "columna": base.agent_pos[1],
                               "bateria": base.battery, "lleva_paquete": int(base.carrying),
                               "paquete_actual": min(base.current, base.n_packages - 1),
                               "tipo": TYPE_NAMES[base.package["type"]], "accion": ACTION_NAMES[action],
                               "evento": base.event, "recompensa": round(reward, 2)})
    result = {
        "exito": bool(base.success), "terminado": bool(terminated), "truncado": bool(truncated),
        "pasos": base.steps, "recompensa": round(base.total_reward, 3), "entregados": base.current,
        "paquetes": base.n_packages, "rescates": base.rescues, "rotos": sum(p["broken"] for p in base.order),
        "choques": events.count("bump") + events.count("break"), "cargas": events.count("charge"),
        "zona_equivocada": events.count("wrong_zone"),
        "acciones_invalidas": sum(events.count(e) for e in ("pick_fail", "drop_fail", "charge_fail", "wrong_zone")),
        "latencia_media_ms": round(float(np.mean(latencies)), 4),
    }
    result["causa"] = failure_cause(result)
    return result, frames


def summarize(per_case, latencies_ms=None):
    """Metricas agregadas de una evaluacion a partir de los resultados por caso."""
    mean = lambda key: float(np.mean([c[key] for c in per_case]))
    successes = [c["pasos"] for c in per_case if c["exito"]]
    causes = {}
    for c in per_case:
        if not c["exito"]:
            causes[c["causa"]] = causes.get(c["causa"], 0) + 1
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
        "acciones_invalidas_medias": mean("acciones_invalidas"),
        "latencia_media_ms": mean("latencia_media_ms"),
        "pedidos_evaluados": len(per_case),
        "fallos_por_causa": causes,
    }
    if latencies_ms:
        metrics["latencia_p95_ms"] = float(np.percentile(latencies_ms, 95))
    return metrics


def evaluate(env, policy, cases, trajectories=None):
    """Evalua una politica (congelada) en una suite de casos. Devuelve (metricas, resultados por caso)."""
    per_case = []
    for i, order in enumerate(cases):
        trajectory = [] if trajectories is not None else None
        result, _ = run_episode(env, policy, order, seed=i, record=False, trajectory=trajectory)
        per_case.append(dict(result, caso=i))
        if trajectories is not None:
            trajectories.extend(dict(step, pedido=i) for step in trajectory)
    return summarize(per_case), per_case


def replay_manual(env, order, actions, seed=0):
    """Control manual sin estado en el servidor: reproduce las acciones del usuario desde el inicio."""
    env.reset(seed=seed, options={"pedido": order})
    frames = [frame(env)]
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(int(action))
        frames.append(frame(env, action, reward, terminated, truncated))
        if terminated or truncated:
            break
    return frames
