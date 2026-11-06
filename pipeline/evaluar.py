"""Etapa 3: evalua el modelo congelado (sin exploracion) en la suite de evaluacion del nivel.

Registra las metricas y los resultados por caso en el mismo run de MLflow del entrenamiento.
Salidas: metrics/evaluacion.json, metrics/evaluacion_por_caso.json, data/trayectorias_evaluacion.csv
"""

import csv
import json
import os

import mlflow
import numpy as np

import tracking
from common import data_path, load_orders, load_params, make_env, resolve_params, write_json
from warehouse_rush import niveles
from warehouse_rush.politicas import QTablePolicy
from warehouse_rush.simulacion import evaluate


def main():
    cfg = resolve_params(load_params())
    cases = load_orders(data_path(cfg["nivel"], "evaluacion"))
    suite_id = f"{cfg['nivel']}-{niveles.sha256_json(cases)[:12]}"
    trajectories = []
    metrics, per_case = evaluate(make_env(cfg), QTablePolicy(np.load("models/q_table.npy")), cases, trajectories)

    write_json("metrics/evaluacion.json", metrics)
    write_json("metrics/evaluacion_por_caso.json", {"suite_id": suite_id, "casos": per_case})
    os.makedirs("data", exist_ok=True)
    with open("data/trayectorias_evaluacion.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)

    with open("metrics/mlflow_run.json") as f:
        run_id = json.load(f)["run_id"]
    tracking.setup()
    with mlflow.start_run(run_id=run_id):
        tracking.log_evaluation(metrics, per_case, suite_id)
    for k, v in metrics.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
