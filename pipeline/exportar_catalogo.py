"""Exporta el registro de experimentos de MLflow a manifests/runs.json (lo sirve la API y lo muestra la web).

Incluye todos los runs, tambien los fallidos, interrumpidos y los que no mejoraron, con su hipotesis,
cambios, procedencia, metricas, curvas y enlace a MLflow.

Uso: python pipeline/exportar_catalogo.py
"""

import json
import os

import mlflow

import tracking

OUT = os.path.join(tracking.ROOT, "manifests", "runs.json")
CURVES = ("entrenamiento.tasa_exito", "curva_eval.tasa_exito", "entrenamiento.recompensa_media",
          "curva_eval.recompensa_media")
STATUS = {"FINISHED": "terminado", "FAILED": "fallido", "KILLED": "interrumpido", "RUNNING": "en curso"}


def export():
    tracking.setup()
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], order_by=["start_time ASC"],
                              output_format="list", max_results=1000)
    published = {}
    catalog_path = os.path.join(tracking.ROOT, "manifests", "catalogo.json")
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            published = {m["run_id"]: m["model_id"] for m in json.load(f)["modelos"]}

    rows = []
    for run in runs:
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
        curves = {}
        for key in CURVES:
            if key in metrics:
                history = sorted(client.get_metric_history(run.info.run_id, key), key=lambda m: m.step)
                curves[key] = [[m.step, round(m.value, 4)] for m in history]
        rows.append({
            "run_id": run.info.run_id,
            "nombre": run.info.run_name,
            "estado": STATUS.get(run.info.status, run.info.status.lower()),
            "inicio": run.info.start_time,
            "duracion_s": round((run.info.end_time - run.info.start_time) / 1000, 1) if run.info.end_time else None,
            "tipo": tags.get("tipo", "entrenamiento"),
            "origen": tags.get("origen", ""),
            "nivel": tags.get("nivel", "n2"),
            "environment_version": tags.get("environment_version", "1.0.0"),
            "familia_modelo": tags.get("familia_modelo", ""),
            "hipotesis": tags.get("hipotesis", ""),
            "hipotesis_registro": tags.get("hipotesis_registro", ""),
            "resultado_hipotesis": tags.get("resultado_hipotesis", tags.get("motivo_decision", "")),
            "grupo": tags.get("grupo"),
            "brazo": tags.get("brazo"),
            "cambios": tags.get("cambios", "(base)"),
            "semilla": tags.get("seed", params.get("entrenamiento.semilla")),
            "episodios": params.get("entrenamiento.episodios"),
            "parametros": {k: v for k, v in params.items() if k.startswith(("agente.", "entorno."))},
            "git_commit": tags.get("git.commit", ""),
            "config_sha256": tags.get("config_sha256", ""),
            "evaluation_suite_id": tags.get("evaluation_suite_id", ""),
            "estado_modelo": tags.get("estado_modelo", "checkpoint final disponible" if "eval.tasa_exito" in metrics else ""),
            "parent_run_id": tags.get("parent_run_id"),
            "model_id": published.get(run.info.run_id) or tags.get("model_id"),
            "error": tags.get("error"),
            "metricas": {k[5:]: round(v, 4) for k, v in metrics.items() if k.startswith("eval.")},
            "tiempo_entrenamiento_s": metrics.get("tiempo_entrenamiento_s"),
            "curvas": curves,
            "mlflow_url": tracking.run_url(run),
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"tracking": "dagshub" if tracking.is_remote() else "local", "runs": rows}, f, indent=1,
                  ensure_ascii=False)
    print(f"{len(rows)} runs exportados a {os.path.relpath(OUT, tracking.ROOT)}")


if __name__ == "__main__":
    export()
