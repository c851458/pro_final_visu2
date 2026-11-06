"""Etapa 2: entrena el agente Q-learning en el nivel de params.yaml.

Registra el entrenamiento en MLflow (run "pipeline"); evaluar.py completa el mismo run.
Salidas: models/q_table.npy (modelo), metrics/entrenamiento.csv (curvas), metrics/mlflow_run.json
"""

import csv
import os
import time

import mlflow

import tracking
from common import data_path, load_orders, load_params, make_env, resolve_params, write_json
from core import train

MODEL_PATH = "models/q_table.npy"


def report(row):
    tracking.log_curve(row)
    print(f"Episodio {row['episodio']:6d} | exito entrenamiento {row['tasa_exito'] * 100:5.1f}% | "
          f"exito evaluacion {row.get('eval_tasa_exito', 0) * 100:5.1f}% | epsilon {row['epsilon']:.3f}", flush=True)


def main():
    cfg = resolve_params(load_params())
    level = cfg["nivel"]
    orders = load_orders(data_path(level, "entrenamiento"))
    curve_cases = load_orders(data_path(level, "evaluacion"))[:cfg["entrenamiento"]["casos_curva"]]

    with tracking.start_run(f"pipeline-{level}", cfg, origin="pipeline-dvc") as run:
        start = time.time()
        agent, curve = train(make_env(cfg), cfg, orders, curve_cases, on_checkpoint=report)
        mlflow.log_metric("tiempo_entrenamiento_s", time.time() - start)
        os.makedirs("models", exist_ok=True)
        agent.save(MODEL_PATH)
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/entrenamiento.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        mlflow.log_artifact(MODEL_PATH, "modelo")
        mlflow.log_artifact("metrics/entrenamiento.csv")
        write_json("metrics/mlflow_run.json", {"run_id": run.info.run_id, "nivel": level})


if __name__ == "__main__":
    main()
