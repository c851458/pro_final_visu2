"""Etapa 2: entrena el agente Q-learning recorriendo los pedidos de entrenamiento.

Registra el entrenamiento en MLflow (run "pipeline"); evaluar.py completa el mismo run.
Salidas: models/q_table_v1.pkl (modelo), metrics/entrenamiento.csv (curva), metrics/mlflow_run.json
"""

import csv
import os

import mlflow

import tracking
from common import load_orders, load_params, make_env, write_json
from core import train

MODEL_PATH = "models/q_table_v1.pkl"


def main():
    params = load_params()
    orders = load_orders("data/pedidos_entrenamiento.json")

    def report(row):
        tracking.log_curve(row)
        print(f"Episodio {row['episodio']:6d} | exito {row['tasa_exito'] * 100:5.1f}% | "
              f"recompensa {row['recompensa_media']:7.2f} | rescates {row['rescates_medios']:.2f} | "
              f"epsilon {row['epsilon']:.3f}", flush=True)

    with tracking.start_run("pipeline", params, origin="pipeline-dvc") as run:
        agent, curve = train(make_env(params), params, orders, on_checkpoint=report)
        os.makedirs("models", exist_ok=True)
        agent.save(MODEL_PATH)
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/entrenamiento.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        mlflow.log_artifact(MODEL_PATH, "modelo")
        mlflow.log_artifact("metrics/entrenamiento.csv")
        mlflow.log_artifact("params.yaml")
        write_json("metrics/mlflow_run.json", {"run_id": run.info.run_id})


if __name__ == "__main__":
    main()
