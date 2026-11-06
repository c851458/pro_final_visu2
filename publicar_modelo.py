#!/usr/bin/env python3
"""Publica modelos en Hugging Face Hub con su manifiesto de recuperacion, y administra el catalogo.

Cada publicacion:
  - sube los pesos (q_table.npy, sin pickle) a modelos/<model_id>/ en Hugging Face (un commit nuevo e inmutable);
  - escribe manifests/modelos/<model_id>.json: pesos y SHA-256, contrato de observaciones/acciones,
    ambiente compatible, configuracion y su hash, run, semilla, presupuesto, suite y origen;
  - lo agrega a manifests/catalogo.json como candidato y lo enlaza al run de MLflow (tags hf.*).
Cada modelo publicado es tambien una version del Model Registry de MLflow (warehouse-rush-<nivel>), con el
alias "recomendado" en el que sirve la web y tags con su estado, hash y URI de Hugging Face.
Publicar NO promueve: cada nivel tiene un modelo recomendado (el que sirve la web por defecto), que se cambia
con --recomendar solo si cumple los criterios de configs/criterios.yaml; --rollback NIVEL vuelve al anterior.
Republicar el mismo artefacto no crea otra version.

Uso:
    ./publicar_modelo.py --run pipeline-n2                 # publica el modelo de un run (nombre o id)
    ./publicar_modelo.py --run pipeline-n2 --tag v1.1      # con tag de version en Hugging Face
    ./publicar_modelo.py --recomendar <model_id>           # promueve un modelo publicado
    ./publicar_modelo.py --rollback n2                     # restaura el recomendado anterior del nivel
    ./publicar_modelo.py --rechazar <model_id> --motivo "..."   # conserva el modelo marcado como rechazado
    ./publicar_modelo.py --sincronizar-registro            # crea/actualiza las versiones del Model Registry de MLflow
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


def registry_name(level):
    return f"warehouse-rush-{level}"


def _registry_url(name, version):
    return f"{tracking.tracking_uri().rstrip('/')}/#/models/{name}/versions/{version}"


def register_version(manifest, run=None):
    """Crea (o reutiliza) la version del Model Registry de MLflow de un modelo publicado. Devuelve (nombre, version)."""
    from mlflow.exceptions import MlflowException

    tracking.setup()
    client = mlflow.MlflowClient()
    name = registry_name(manifest["level_id"])
    try:
        client.create_registered_model(name, description=f"Modelos Q-learning del nivel {manifest['level_id'].upper()} "
                                                         "de Warehouse Rush (pesos en Hugging Face; ver tags).")
    except MlflowException as error:
        if "RESOURCE_ALREADY_EXISTS" not in str(error) and "already exists" not in str(error):
            raise
    for version in client.search_model_versions(f"name='{name}'"):
        if version.tags.get("model_id") == manifest["model_id"]:
            return name, version.version
    run = run or find_run(manifest["run_id"])
    version = client.create_model_version(
        name, source=f"{run.info.artifact_uri}/modelo", run_id=run.info.run_id,
        description=manifest.get("hipotesis") or "",
        tags={"model_id": manifest["model_id"], "model_sha256": manifest["model_sha256"],
              "model_uri": manifest["model_uri"], "environment_version": manifest["environment_version"],
              "familia_modelo": manifest["familia_modelo"], "evaluation_suite_id": manifest["evaluation_suite_id"],
              "tasa_exito": str(manifest["metricas_evaluacion"].get("tasa_exito")), "estado": "candidato"})
    return name, version.version


def update_registry(model_id, estado, motivo=None, alias=False):
    """Refleja en el Model Registry el estado del catalogo (y mueve el alias "recomendado")."""
    manifest = _manifest(model_id)
    reg = manifest.get("mlflow_registro")
    if not reg:
        return
    client = mlflow.MlflowClient()
    client.set_model_version_tag(reg["nombre"], reg["version"], "estado", estado)
    if motivo:
        client.set_model_version_tag(reg["nombre"], reg["version"], "motivo", motivo[:500])
    if alias:
        client.set_registered_model_alias(reg["nombre"], "recomendado", reg["version"])


def _attach_registry(model_id, name, version):
    """Guarda la referencia al Model Registry en el manifiesto y en el catalogo."""
    path = os.path.join(MANIFESTS, "modelos", f"{model_id}.json")
    with open(path) as f:
        manifest = json.load(f)
    manifest["mlflow_registro"] = {"nombre": name, "version": str(version), "url": _registry_url(name, version)}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    catalog = load_catalog()
    for m in catalog["modelos"]:
        if m["model_id"] == model_id:
            m["registro_mlflow"] = f"{name} v{version}"
            m["registro_mlflow_url"] = _registry_url(name, version)
    save_catalog(catalog)


def sync_registry():
    """Crea las versiones que falten y alinea estados y aliases con el catalogo."""
    tracking.setup()
    catalog = load_catalog()
    for entry in catalog["modelos"]:
        manifest = _manifest(entry["model_id"])
        name, version = register_version(manifest)
        _attach_registry(entry["model_id"], name, version)
        update_registry(entry["model_id"], entry["estado"], entry.get("motivo"),
                        alias=catalog["recomendados"].get(entry["level_id"]) == entry["model_id"])
        print(f"  {entry['model_id']:40s} -> {name} v{version} ({entry['estado']})")


def load_catalog():
    """Catalogo con un modelo recomendado por nivel (y su historial para rollback)."""
    catalog = {"recomendados": {}, "historial_recomendados": {}, "modelos": []}
    if os.path.exists(CATALOG):
        with open(CATALOG) as f:
            catalog = json.load(f)
    if "recomendados" not in catalog:  # formato anterior: un solo recomendado global
        levels = {m["model_id"]: m["level_id"] for m in catalog["modelos"]}
        old = catalog.pop("recomendado", None)
        catalog["recomendados"] = {levels[old]: old} if old else {}
        catalog["historial_recomendados"] = {}
    return catalog


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
    name, version = register_version(manifest, run)
    _attach_registry(model_id, name, version)
    print(f"  Model Registry de MLflow: {name} v{version}")
    print(f"Publicado {model_id}\n  pesos: {manifest['model_download_url']}\n  manifiesto: manifests/modelos/{model_id}.json")
    return model_id


def _manifest(model_id):
    with open(os.path.join(MANIFESTS, "modelos", f"{model_id}.json")) as f:
        return json.load(f)


def _requirements():
    with open(os.path.join(ROOT, "configs", "criterios.yaml")) as f:
        return yaml.safe_load(f)


def recommend(model_id, force=False):
    """Promueve un modelo a recomendado de su nivel, si cumple los criterios fijados en configs/criterios.yaml."""
    catalog = load_catalog()
    entry = next((m for m in catalog["modelos"] if m["model_id"] == model_id), None)
    if entry is None:
        raise SystemExit(f"{model_id} no esta en el catalogo")
    level = entry["level_id"]
    criteria = _requirements()
    req = criteria["requisitos"]
    candidate = _manifest(model_id)["metricas_evaluacion"]
    problems = []
    if candidate["tasa_exito"] < req["tasa_exito_min"]:
        problems.append(f"tasa_exito {candidate['tasa_exito']:.3f} < {req['tasa_exito_min']}")
    if candidate.get("fragiles_rotos_medios", 0) > req["fragiles_rotos_medios_max"]:
        problems.append("demasiados fragiles rotos")
    if candidate.get("rescates_medios", 0) > req["rescates_medios_max"]:
        problems.append("demasiados rescates")
    current = catalog["recomendados"].get(level)
    if current and current != model_id:
        cur = _manifest(current)
        tolerance = criteria["promocion"]["tolerancia_tasa_exito"]
        if cur["evaluation_suite_id"] == _manifest(model_id)["evaluation_suite_id"] and \
                candidate["tasa_exito"] < cur["metricas_evaluacion"]["tasa_exito"] - tolerance:
            problems.append(f"empeora al recomendado actual {current} en la misma suite")
    if problems and not force:
        raise SystemExit(f"No se promueve {model_id}: {'; '.join(problems)} (usa --forzar solo con justificacion)")
    if current and current != model_id:
        catalog["historial_recomendados"].setdefault(level, []).append(current)
    catalog["recomendados"][level] = model_id
    for m in catalog["modelos"]:
        if m["model_id"] == model_id:
            m["estado"] = "recomendado"
        elif m["model_id"] == current:
            m["estado"] = "anterior"
    save_catalog(catalog)
    tracking.setup()
    if current and current != model_id:
        update_registry(current, "anterior")
    update_registry(model_id, "recomendado", alias=True)
    print(f"Recomendado en {level}: {model_id}" + (f" (antes {current})" if current else ""))


def reject(model_id, reason):
    catalog = load_catalog()
    for m in catalog["modelos"]:
        if m["model_id"] == model_id:
            if model_id in catalog["recomendados"].values():
                raise SystemExit("No se puede rechazar un modelo recomendado; primero recomienda otro")
            m["estado"] = "rechazado"
            m["motivo"] = reason
            save_catalog(catalog)
            tracking.setup()
            update_registry(model_id, "rechazado", reason)
            print(f"Rechazado: {model_id} ({reason})")
            return
    raise SystemExit(f"{model_id} no esta en el catalogo")


def rollback(level):
    """Restaura el recomendado anterior de un nivel (el actual queda como revertido)."""
    catalog = load_catalog()
    history = catalog["historial_recomendados"].get(level, [])
    if not history:
        raise SystemExit(f"No hay un recomendado anterior en {level} al que volver")
    previous = history.pop()
    current = catalog["recomendados"].get(level)
    catalog["recomendados"][level] = previous
    for m in catalog["modelos"]:
        if m["model_id"] == previous:
            m["estado"] = "recomendado"
        elif m["model_id"] == current:
            m["estado"] = "revertido"
    save_catalog(catalog)
    tracking.setup()
    update_registry(current, "revertido")
    update_registry(previous, "recomendado", alias=True)
    print(f"Rollback en {level}: {current} -> {previous}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="nombre o id del run de MLflow cuyo modelo se publica")
    parser.add_argument("--tag", help="etiqueta de version en Hugging Face, por ejemplo v1.1")
    parser.add_argument("--publico", action="store_true", help="crear el repositorio de Hugging Face como publico")
    parser.add_argument("--recomendar", metavar="MODEL_ID", help="promueve un modelo a recomendado de su nivel")
    parser.add_argument("--forzar", action="store_true", help="promover aunque no cumpla los criterios")
    parser.add_argument("--rollback", metavar="NIVEL", help="vuelve al recomendado anterior de un nivel (n1, n2, n3)")
    parser.add_argument("--rechazar", metavar="MODEL_ID", help="marca un modelo publicado como rechazado")
    parser.add_argument("--motivo", default="", help="motivo del rechazo (se guarda en el catalogo)")
    parser.add_argument("--sincronizar-registro", action="store_true", help="alinear el Model Registry de MLflow")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.run:
        publish(args.run, args.tag, args.publico)
    if args.recomendar:
        recommend(args.recomendar, args.forzar)
    if args.rollback:
        rollback(args.rollback)
    if args.rechazar:
        reject(args.rechazar, args.motivo)
    if args.sincronizar_registro:
        sync_registry()
    if not (args.run or args.recomendar or args.rollback or args.rechazar or args.sincronizar_registro):
        parser.print_help()


if __name__ == "__main__":
    main()
