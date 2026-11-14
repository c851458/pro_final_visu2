"""Carga de modelos publicados: tabla Q (.npy) o red PPO exportada a NumPy (.npz).

Lo usan la API, el flujo de Prefect y las evaluaciones, para que todos interpreten igual cada artefacto.
"""

import numpy as np

from warehouse_rush.politicas import QTablePolicy

TABULAR, PPO = "tabla-q", "ppo-mlp"
FILES = {TABULAR: "q_table.npy", PPO: "politica.npz"}


def model_type(path):
    return PPO if str(path).endswith(".npz") else TABULAR


def load_policy(path):
    """Politica lista para `act(env, obs)` a partir del archivo del modelo."""
    if model_type(path) == PPO:
        from warehouse_rush.ppo import MLPPolicy

        with np.load(path, allow_pickle=False) as data:
            return MLPPolicy({k: data[k] for k in data.files})
    return QTablePolicy(np.load(path, allow_pickle=False))
