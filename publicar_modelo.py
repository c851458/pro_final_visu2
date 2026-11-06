#!/usr/bin/env python3
"""Publica modelos en Hugging Face Hub con su manifiesto de recuperacion, y administra el catalogo.

Cada publicacion:
  - sube los pesos (q_table.npy, sin pickle) a modelos/<model_id>/ en Hugging Face (un commit nuevo e inmutable);
  - escribe manifests/modelos/<model_id>.json: pesos y SHA-256, contrato de observaciones/acciones,
    ambiente compatible, configuracion y su hash, run, semilla, presupuesto, suite y origen;
  - lo agrega a manifests/catalogo.json como candidato y lo enlaza al run de MLflow (tags hf.*).
Publicar NO promueve: el modelo recomendado (el que sirve la web por defecto) se cambia con --recomendar,
y --rollback vuelve al recomendado anterior. Republicar el mismo artefacto no crea otra version.

Uso:
    ./publicar_modelo.py --run pipeline-n2                 # publica el modelo de un run (nombre o id)
    ./publicar_modelo.py --run pipeline-n2 --tag v1.1      # con tag de version en Hugging Face
    ./publicar_modelo.py --recomendar <model_id>           # promueve un modelo publicado
    ./publicar_modelo.py --rollback                        # restaura el recomendado anterior
    ./publicar_modelo.py --rechazar <model_id> --motivo "..."   # conserva el modelo marcado como rechazado
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import tracking  # noqa: E402  (antes que mlflow: configura su entorno)
import mlflow  # noqa: E402
from warehouse_rush import niveles  # noqa: E402
from warehouse_rush.env_v1 import ACTION_NAMES  # noqa: E402

REPO_NAME = "warehouse-rush-qlearning"
GITHUB_REPO = "github.com/c851458/pro_final_visu2"
MANIFESTS = os.path.join(ROOT, "manifests")
CATALOG = os.path.join(MANIFESTS, "catalogo.json")

OBSERVATION_CONTRACT = [
    {"nombre": "fila", "unidad": "casilla", "rango": "0-10"},
    {"nombre": "columna", "unidad": "casilla", "rango": "0-14"},
    {"nombre": "nivel_bateria", "unidad": "cuartos de bateria (0 vacia, 4 llena); 0 fijo sin bateria", "rango": "0-4"},
    {"nombre": "lleva_paquete", "unidad": "booleano", "rango": "0-1"},
    {"nombre": "tipo_paquete_cargado", "unidad": "0 normal, 1 pesado, 2 fragil (0 si no lleva)", "rango": "0-2"},
    {"nombre": "objetivo", "unidad": "indice de estanteria (sin paquete) o de zona A/B/C (con paquete)", "rango": "0-39"},
]


def find_run(spec):
    """Busca un run por id o por nombre (el mas reciente con ese nombre)."""
    tracking.setup()
    if re.fullmatch(r"[0-9a-f]{32}", spec):
        return mlflow.get_run(spec)
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], filter_string=f"tags.mlflow.runName = '{spec}'",
                              order_by=["start_time DESC"], max_results=1, output_format="list")
    if not runs:
        raise SystemExit(f"No hay ningun run de MLflow llamado '{spec}'")
    return runs[0]


def download_model(run, folder):
    """Descarga los pesos de un run (formato .npy, o .pkl de runs antiguos) y devuelve la ruta al .npy."""
    os.makedirs(folder, exist_ok=True)
    src = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="modelo", dst_path=folder)
    npy = os.path.join(src, "q_table.npy")
    if not os.path.exists(npy):
        from warehouse_rush import QLearningAgent

        agent = QLearningAgent([1], 1)
        agent.load(os.path.join(src, "q_table_v1.pkl"))
        np.save(npy, agent.q, allow_pickle=False)
    return npy


def run_config(run, folder):
    """Configuracion resuelta del run (config.json; en runs antiguos se reconstruye desde params.yaml)."""
    client = mlflow.MlflowClient()
    names = {a.path for a in client.list_artifacts(run.info.run_id)}
    if "config.json" in names:
        with open(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="config.json",
                                                      dst_path=folder)) as f:
            return json.load(f)
    with open(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="params.yaml",
                                                  dst_path=folder)) as f:
        old = yaml.safe_load(f)
    level = niveles.resolve("n2")  # los runs anteriores a los niveles usaban exactamente el ambiente N2
    level["entorno"].update(bateria=old["entorno"]["bateria"], bateria_max=old["entorno"]["bateria_max"],
                            max_pasos=old["entorno"]["max_pasos"])
    return {"nivel": "n2", "version_ambiente": niveles.env_version(), "entorno": level["entorno"],
            "datos": old["datos"], "agente": old["agente"], "entrenamiento": old["entrenamiento"]}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_catalog():
    if os.path.exists(CATALOG):
        with open(CATALOG) as f:
            return json.load(f)
    return {"recomendado": None, "historial_recomendados": [], "modelos": []}


def save_catalog(catalog):
    catalog["actualizado"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(MANIFESTS, exist_ok=True)
    with open(CATALOG, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def record_reproduction(model_id, run):
    """Anota en el manifiesto que otro run produjo exactamente los mismos pesos (prueba de reproducibilidad)."""
    path = os.path.join(MANIFESTS, "modelos", f"{model_id}.json")
    with open(path) as f:
        manifest = json.load(f)
    entries = manifest.setdefault("reproducido_por", [])
    if run.info.run_id in {e["run_id"] for e in entries} or run.info.run_id == manifest["run_id"]:
        return
    tags = run.data.tags
    entries.append({"run_id": run.info.run_id, "run_name": run.info.run_name, "git_commit": tags.get("git.commit", ""),
                    "git_cambios_sin_commit": tags.get("git.cambios_sin_commit") == "True",
                    "mlflow_run_url": tracking.run_url(run)})
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tracking.setup()
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tags({"model_id": model_id, "reproduce_modelo": model_id})


def publish(run_spec, tag=None, public=False):
    from huggingface_hub import HfApi

    run = find_run(run_spec)
    metrics = run.data.metrics
    if "eval.tasa_exito" not in metrics:
        raise SystemExit(f"El run {run.info.run_name} no tiene evaluacion; no se publica un modelo sin evaluar")
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{REPO_NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        npy = download_model(run, os.path.join(tmp, "modelo"))
        cfg = run_config(run, tmp)
        weights_sha = sha256_file(npy)
        catalog = load_catalog()
        for entry in catalog["modelos"]:
            if entry["model_sha256"] == weights_sha:
                record_reproduction(entry["model_id"], run)
                print(f"Ese artefacto ya esta publicado como {entry['model_id']}; no se crea otra version "
                      f"(el run {run.info.run_name} lo reprodujo con el mismo SHA-256).")
                return entry["model_id"]

        level = cfg["nivel"]
        model_id = f"{level}-{run.info.run_name}-{weights_sha[:8]}"
        env = niveles.make_env(cfg)
        q = np.load(npy)
        if tuple(q.shape) != tuple(env.observation_space.nvec) + (env.action_space.n,):
            raise SystemExit(f"Los pesos {q.shape} no coinciden con el ambiente del nivel {level}")
        tags = run.data.tags
        eval_metrics = {k[5:]: v for k, v in metrics.items() if k.startswith("eval.") and not k.startswith("eval.fallos.")}
        failures = {k[12:]: v for k, v in metrics.items() if k.startswith("eval.fallos.")}

        api.create_repo(repo_id, private=not public, exist_ok=True)
        folder = f"modelos/{model_id}"
        message = f"Modelo {model_id} (run {run.info.run_name}): {metrics['eval.tasa_exito'] * 100:.1f}% de pedidos completos"
        commit = api.upload_file(path_or_fileobj=npy, path_in_repo=f"{folder}/q_table.npy", repo_id=repo_id,
                                 commit_message=message)
        if tag:
            api.create_tag(repo_id, tag=tag, revision=commit.oid, tag_message=message)

    manifest = {
        "model_id": model_id,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "mlflow_run_url": tracking.run_url(run),
        "level_id": level,
        "environment_version": cfg["version_ambiente"],
        "familia_modelo": niveles.family(env),
        "git_commit": tags.get("git.commit", ""),
        # True: el run se entreno con cambios sin commit, el codigo exacto no es git_commit
        "git_cambios_sin_commit": tags.get("git.cambios_sin_commit") == "True",
        "config_sha256": niveles.sha256_json(cfg),
        "config": cfg,
        "model_uri": f"hf://{repo_id}@{commit.oid}/{folder}/q_table.npy",
        "model_download_url": f"https://huggingface.co/{repo_id}/resolve/{commit.oid}/{folder}/q_table.npy",
        "model_sha256": weights_sha,
        "formato": {"archivo": "q_table.npy", "dtype": str(q.dtype), "forma": list(q.shape)},
        "normalization_uri": None,
        "normalizacion": "ninguna: las observaciones son enteros discretos usados como indices de la tabla Q (sin escalado ni clipping)",
        "observacion": OBSERVATION_CONTRACT,
        "acciones": [{"id": i, "nombre": n} for i, n in enumerate(ACTION_NAMES)],
        "runtime_ref": f"{GITHUB_REPO}@{tags.get('git.commit', '')} (warehouse_rush/env_v1.py, configs/niveles.yaml, requirements.txt)",
        "evaluation_suite_id": tags.get("evaluation_suite_id", ""),
        "seed": cfg["entrenamiento"]["semilla"],
        "parent_run_id": tags.get("parent_run_id"),
        "origen": tags.get("origen_modelo", "desde cero"),
        "optimizador": "Q-learning tabular (sin optimizador ni contador: se continua desde la tabla Q)",
        "presupuesto": {"episodios": cfg["entrenamiento"]["episodios"],
                        "tiempo_entrenamiento_s": metrics.get("tiempo_entrenamiento_s")},
        "hipotesis": tags.get("hipotesis", ""),
        "metricas_evaluacion": eval_metrics,
        "fallos_por_causa": failures,
        "hf_repo": repo_id,
        "hf_commit": commit.oid,
        "hf_tag": tag,
        "publicado": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.join(MANIFESTS, "modelos"), exist_ok=True)
    with open(os.path.join(MANIFESTS, "modelos", f"{model_id}.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    catalog["modelos"].append({k: manifest[k] for k in (
        "model_id", "run_id", "run_name", "level_id", "familia_modelo", "environment_version", "model_sha256",
        "hf_commit", "hf_tag", "git_commit", "publicado", "hipotesis")} | {
        "tasa_exito": eval_metrics.get("tasa_exito"), "estado": "candidato"})
    save_catalog(catalog)

    tracking.setup()
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tags({"hf.repo": repo_id, "hf.commit": commit.oid, "model_id": model_id, "model_sha256": weights_sha,
                         **({"hf.tag": tag} if tag else {})})
    print(f"Publicado {model_id}\n  pesos: {manifest['model_download_url']}\n  manifiesto: manifests/modelos/{model_id}.json")
    return model_id


def recommend(model_id):
    catalog = load_catalog()
    ids = {m["model_id"] for m in catalog["modelos"]}
    if model_id not in ids:
        raise SystemExit(f"{model_id} no esta en el catalogo")
    if catalog["recomendado"] and catalog["recomendado"] != model_id:
        catalog["historial_recomendados"].append(catalog["recomendado"])
    catalog["recomendado"] = model_id
    for m in catalog["modelos"]:
        if m["model_id"] == model_id:
            m["estado"] = "recomendado"
        elif m["estado"] == "recomendado":
            m["estado"] = "anterior"
    save_catalog(catalog)
    print(f"Recomendado: {model_id}")


def reject(model_id, reason):
    catalog = load_catalog()
    for m in catalog["modelos"]:
        if m["model_id"] == model_id:
            if m["model_id"] == catalog["recomendado"]:
                raise SystemExit("No se puede rechazar el modelo recomendado; primero recomienda otro")
            m["estado"] = "rechazado"
            m["motivo"] = reason
            save_catalog(catalog)
            print(f"Rechazado: {model_id} ({reason})")
            return
    raise SystemExit(f"{model_id} no esta en el catalogo")


def rollback():
    catalog = load_catalog()
    if not catalog["historial_recomendados"]:
        raise SystemExit("No hay un recomendado anterior al que volver")
    previous = catalog["historial_recomendados"].pop()
    current = catalog["recomendado"]
    catalog["recomendado"] = previous
    for m in catalog["modelos"]:
        if m["model_id"] == previous:
            m["estado"] = "recomendado"
        elif m["model_id"] == current:
            m["estado"] = "revertido"
    save_catalog(catalog)
    print(f"Rollback: {current} -> {previous}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="nombre o id del run de MLflow cuyo modelo se publica")
    parser.add_argument("--tag", help="etiqueta de version en Hugging Face, por ejemplo v1.1")
    parser.add_argument("--publico", action="store_true", help="crear el repositorio de Hugging Face como publico")
    parser.add_argument("--recomendar", metavar="MODEL_ID", help="promueve un modelo publicado a recomendado")
    parser.add_argument("--rollback", action="store_true", help="vuelve al modelo recomendado anterior")
    parser.add_argument("--rechazar", metavar="MODEL_ID", help="marca un modelo publicado como rechazado")
    parser.add_argument("--motivo", default="", help="motivo del rechazo (se guarda en el catalogo)")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.run:
        publish(args.run, args.tag, args.publico)
    if args.recomendar:
        recommend(args.recomendar)
    if args.rollback:
        rollback()
    if args.rechazar:
        reject(args.rechazar, args.motivo)
    if not (args.run or args.recomendar or args.rollback or args.rechazar):
        parser.print_help()


if __name__ == "__main__":
    main()
