#!/usr/bin/env python3
"""Corre experimentos (variantes de params.yaml) y los registra en MLflow.

Cada experimento entrena y evalua con el mismo codigo del pipeline y queda como un run de MLflow
con sus parametros, curva de aprendizaje, metricas de evaluacion, modelo, commit de git y version del dataset.

Uso:
    ./experimentar.py -e alpha-alto agente.alpha=0.2
    ./experimentar.py -e base -e alpha-alto agente.alpha=0.2 -e gamma-bajo agente.gamma=0.95 -j 3
    mlflow ui --backend-store-uri sqlite:///mlflow.db      # ver y comparar en http://localhost:5000
"""

import argparse
import copy
import csv
import json
import os
import sys
import tempfile
from multiprocessing import Pool

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import tracking  # noqa: E402  (antes que mlflow: configura su entorno)
import mlflow  # noqa: E402
from common import load_orders, load_params, make_env  # noqa: E402
from core import evaluate, generate_orders, train  # noqa: E402


def apply_overrides(params, overrides):
    params = copy.deepcopy(params)
    for item in overrides:
        key, _, raw = item.partition("=")
        node = params
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        if last not in node:
            raise SystemExit(f"Parametro desconocido: {key}")
        node[last] = yaml.safe_load(raw)
    return params


def dataset(params, overrides):
    """Usa el dataset versionado con DVC; si se cambian parametros de datos, lo genera aparte."""
    files = ["data/pedidos_entrenamiento.json", "data/pedidos_evaluacion.json"]
    if not any(o.startswith("datos.") for o in overrides) and all(os.path.exists(f) for f in files):
        return load_orders(files[0]), load_orders(files[1]), "dvc"
    training, evaluation = generate_orders(params["datos"])
    return training, evaluation, "generado"


def run_experiment(job):
    """Entrena y evalua en un proceso aparte; el registro en MLflow lo hace el proceso principal."""
    name, overrides, workdir = job
    params = apply_overrides(load_params(), overrides)
    training_orders, eval_orders, source = dataset(params, overrides)
    agent, curve = train(make_env(params), params, training_orders)
    metrics, _ = evaluate(make_env(params), agent, eval_orders)
    model_path = os.path.join(workdir, name, "q_table_v1.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"  terminado: {name} (exito {metrics['tasa_exito'] * 100:.1f}%)", flush=True)
    return {"name": name, "overrides": overrides, "params": params, "curve": curve, "metrics": metrics,
            "model_path": model_path, "dataset": source}


def log_to_mlflow(result, workdir):
    with tracking.start_run(result["name"], result["params"], origin="experimentar.py") as run:
        mlflow.set_tags({"cambios": " ".join(result["overrides"]) or "(base)", "dataset": result["dataset"]})
        for row in result["curve"]:
            tracking.log_curve(row)
        tracking.log_evaluation(result["metrics"])
        folder = os.path.join(workdir, result["name"])
        with open(os.path.join(folder, "entrenamiento.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["curve"][0]))
            writer.writeheader()
            writer.writerows(result["curve"])
        with open(os.path.join(folder, "evaluacion.json"), "w") as f:
            json.dump(result["metrics"], f, indent=2)
        with open(os.path.join(folder, "params.yaml"), "w") as f:
            yaml.safe_dump(result["params"], f, allow_unicode=True, sort_keys=False)
        mlflow.log_artifact(result["model_path"], "modelo")
        for artifact in ("entrenamiento.csv", "evaluacion.json", "params.yaml"):
            mlflow.log_artifact(os.path.join(folder, artifact))
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--experimento", nargs="+", action="append", required=True,
                        metavar=("NOMBRE", "PARAM=VALOR"),
                        help="nombre del experimento y cambios sobre params.yaml (se puede repetir)")
    parser.add_argument("-j", "--procesos", type=int, default=4, help="experimentos en paralelo")
    args = parser.parse_args()
    os.chdir(ROOT)

    names = [e[0] for e in args.experimento]
    if len(set(names)) != len(names):
        raise SystemExit("Los nombres de experimento deben ser distintos")
    tracking.setup()
    with tempfile.TemporaryDirectory() as workdir:
        jobs = [(e[0], e[1:], workdir) for e in args.experimento]
        for name, overrides, _ in jobs:  # valida los parametros antes de lanzar procesos
            apply_overrides(load_params(), overrides)
        print(f"Corriendo {len(jobs)} experimento(s) con {min(args.procesos, len(jobs))} proceso(s)...", flush=True)
        with Pool(min(args.procesos, len(jobs))) as pool:
            results = pool.map(run_experiment, jobs)
        for result in results:
            result["run_id"] = log_to_mlflow(result, workdir)

    print(f"\n{'experimento':18s} {'exito':>7s} {'pasos':>7s} {'rescates':>9s} {'choques':>8s}  run_id")
    for r in sorted(results, key=lambda r: -r["metrics"]["tasa_exito"]):
        m = r["metrics"]
        print(f"{r['name']:18s} {m['tasa_exito'] * 100:6.1f}% {m['pasos_medios']:7.1f} "
              f"{m['rescates_medios']:9.3f} {m['choques_medios']:8.3f}  {r['run_id']}")
    print("\nVer en MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
