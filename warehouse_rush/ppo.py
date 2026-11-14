"""Politica PPO (red neuronal) para Warehouse Rush: observacion numerica, envoltorio de entrenamiento e inferencia
en NumPy.

La red no usa la observacion tabular: recibe un vector de caracteristicas continuas calculado del estado del
ambiente, que incluye lo que la tabla Q no ve (bateria exacta y plazo restante del SLA). La misma funcion
`features` se usa al entrenar y al servir el modelo, asi la API evalua exactamente lo que se entreno.

La inferencia es NumPy puro (sin PyTorch) para que la API quepa en Vercel: los pesos de la red se exportan a .npz.
"""

import gymnasium as gym
import numpy as np

from warehouse_rush.env_v1 import BLOCKING, MOVES, TYPE_NAMES

FEATURE_NAMES = [
    "fila", "columna", "bateria", "lleva_paquete", "tipo_normal", "tipo_pesado", "tipo_fragil", "express",
    "sla_restante", "objetivo_dfila", "objetivo_dcolumna", "objetivo_distancia", "cargador_dfila",
    "cargador_dcolumna", "bloqueado_arriba", "bloqueado_abajo", "bloqueado_izquierda", "bloqueado_derecha",
    "junto_a_estanteria_objetivo", "en_muelle_destino", "en_cargador", "paquetes_restantes",
]
N_FEATURES = len(FEATURE_NAMES)
SLA_SCALE = 100.0  # pasos: el plazo restante se divide por este valor y se recorta a [-1, 1]


def _target(base):
    """Casilla objetivo: estanteria del paquete actual (sin carga) o centro del muelle de destino (con carga)."""
    pkg = base.package
    if base.carrying:
        cells = sorted(base.zone_cells[pkg["zone"]])
        return cells[len(cells) // 2]
    return base.shelf_cells[pkg["shelf"]]


def features(base):
    """Vector de caracteristicas normalizadas (float32) del estado del ambiente (env.unwrapped)."""
    h, w = base.height - 1, base.width - 1
    r, c = base.agent_pos
    pkg = base.package
    tr, tc = _target(base)
    cr, cc = base.charger
    deadline = pkg.get("deadline")
    remaining = 1.0 if deadline is None else float(np.clip((deadline - base.steps) / SLA_SCALE, -1, 1))
    type_onehot = [1.0 if pkg["type"] == i else 0.0 for i in range(len(TYPE_NAMES))]
    blocked = [1.0 if base.grid[r + dr, c + dc] in BLOCKING else 0.0 for dr, dc in MOVES.values()]
    adjacent_shelf = (not base.carrying) and abs(r - tr) + abs(c - tc) == 1
    in_zone = base.carrying and (r, c) in base.zone_cells[pkg["zone"]]
    return np.array([
        r / h, c / w,
        base.battery / base.battery_max if base.use_battery else 1.0,
        float(base.carrying), *type_onehot, float(pkg.get("service", 0)), remaining,
        (tr - r) / h, (tc - c) / w, (abs(tr - r) + abs(tc - c)) / (h + w),
        (cr - r) / h, (cc - c) / w, *blocked,
        float(adjacent_shelf), float(in_zone), float((r, c) == base.charger),
        (base.n_packages - min(base.current, base.n_packages)) / base.n_packages,
    ], dtype=np.float32)


class FeatureObservation(gym.ObservationWrapper):
    """Reemplaza la observacion tabular por el vector de caracteristicas (para entrenar con Stable-Baselines3)."""

    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(N_FEATURES,), dtype=np.float32)

    def observation(self, obs):
        return features(self.env.unwrapped)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward * self.reward_scale, terminated, truncated, info


class MLPPolicy:
    """Red de la politica exportada (capas densas con tanh); accion = argmax de los logits (sin exploracion)."""

    kind = "modelo"

    def __init__(self, weights):
        self.layers = [(weights[f"W{i}"], weights[f"b{i}"]) for i in range(int(weights["n_layers"]))]
        self.out = (weights["W_out"], weights["b_out"])

    def logits(self, x):
        for W, b in self.layers:
            x = np.tanh(x @ W + b)
        W, b = self.out
        return x @ W + b

    def act(self, env, obs):
        return int(np.argmax(self.logits(features(env.unwrapped))))


def export_sb3(model):
    """Pesos de una politica PPO de Stable-Baselines3 (MlpPolicy con tanh) como arreglos NumPy."""
    import torch.nn as nn

    policy = model.policy
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    activations = [m for m in policy.mlp_extractor.policy_net if not isinstance(m, nn.Linear)]
    if not all(isinstance(a, nn.Tanh) for a in activations):
        raise ValueError("La exportacion supone activaciones tanh")
    weights = {"n_layers": np.array(len(linears))}
    for i, layer in enumerate(linears):
        weights[f"W{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        weights[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    weights["W_out"] = policy.action_net.weight.detach().cpu().numpy().T.astype(np.float32)
    weights["b_out"] = policy.action_net.bias.detach().cpu().numpy().astype(np.float32)
    return weights
