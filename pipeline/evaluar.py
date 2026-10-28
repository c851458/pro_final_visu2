"""Etapa 3: evalua el modelo (sin exploracion) en los pedidos de evaluacion.

Registra las metricas en el mismo run de MLflow del entrenamiento.
Salidas: metrics/evaluacion.json (metricas), data/trayectorias_evaluacion.csv (cada paso de cada pedido)
"""

import csv
import json
import os

import mlflow

import tracking
from common import load_orders, load_params, make_env, write_json
from core import evaluate
from warehouse_rush import QLearningAgent


def main():
    params = load_params()
    env = make_env(params)
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n)
    agent.load("models/q_table_v1.pkl")
    metrics, steps_log = evaluate(env, agent, load_orders("data/pedidos_evaluacion.json"))

    write_json("metrics/evaluacion.json", metrics)
    os.makedirs("data", exist_ok=True)
    with open("data/trayectorias_evaluacion.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(steps_log[0]))
        writer.writeheader()
        writer.writerows(steps_log)

    with open("metrics/mlflow_run.json") as f:
        run_id = json.load(f)["run_id"]
    tracking.setup()
    with mlflow.start_run(run_id=run_id):
        tracking.log_evaluation(metrics)
        mlflow.log_artifact("metrics/evaluacion.json")
    for k, v in metrics.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
