"""Copia los runs del MLflow local (mlflow.db) a otro servidor MLflow, por ejemplo el de DagsHub.

Conserva nombre, estado, parametros, tags, historial completo de metricas y artefactos; cada run copiado
lleva el tag migrado_de=<run_id original>. Los runs ya migrados se omiten (se puede repetir sin duplicar).

Uso: python pipeline/migrar_mlflow.py                 # destino: MLFLOW_TRACKING_URI (del .env)
     python pipeline/migrar_mlflow.py --destino <uri>
"""

import argparse
import tempfile

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

import tracking


def migrate(dest_uri):
    if dest_uri.startswith("sqlite:") and dest_uri == tracking.LOCAL_URI:
        raise SystemExit("El destino es el mismo MLflow local")
    src = MlflowClient(tracking.LOCAL_URI)
    src_exp = src.get_experiment_by_name(tracking.EXPERIMENT)
    if src_exp is None:
        raise SystemExit("No hay experimentos locales para migrar")
    runs = src.search_runs([src_exp.experiment_id], order_by=["attributes.start_time ASC"], max_results=1000)

    tracking.setup(dest_uri)
    dst = MlflowClient(dest_uri)
    dst_exp = dst.get_experiment_by_name(tracking.EXPERIMENT)
    done = {r.data.tags.get("migrado_de") for r in dst.search_runs([dst_exp.experiment_id], max_results=1000)}

    for run in runs:
        if run.info.run_id in done:
            print(f"  ya migrado: {run.info.run_name}")
            continue
        new = dst.create_run(dst_exp.experiment_id, start_time=run.info.start_time, run_name=run.info.run_name,
                             tags={"migrado_de": run.info.run_id, "migrado_desde": "mlflow local (mlflow.db)"})
        run_id = new.info.run_id
        metrics = []
        for key in run.data.metrics:
            metrics += [Metric(m.key, m.value, m.timestamp, m.step) for m in src.get_metric_history(run.info.run_id, key)]
        params = [Param(k, v) for k, v in run.data.params.items()]
        tags = [RunTag(k, v) for k, v in run.data.tags.items() if not k.startswith("mlflow.")]
        # limites de MLflow por lote: 1000 metricas, 100 parametros y 100 tags
        for i in range(0, len(metrics), 1000):
            dst.log_batch(run_id, metrics=metrics[i:i + 1000])
        for i in range(0, len(params), 100):
            dst.log_batch(run_id, params=params[i:i + 100])
        for i in range(0, len(tags), 100):
            dst.log_batch(run_id, tags=tags[i:i + 100])
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=tmp,
                                                        tracking_uri=tracking.LOCAL_URI)
            dst.log_artifacts(run_id, local)
        dst.set_terminated(run_id, status=run.info.status, end_time=run.info.end_time)
        print(f"  migrado: {run.info.run_name} -> {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destino", help="tracking URI de destino (por defecto MLFLOW_TRACKING_URI)")
    args = parser.parse_args()
    destination = args.destino or tracking.tracking_uri()
    print(f"Migrando {tracking.LOCAL_URI} -> {destination}")
    migrate(destination)
