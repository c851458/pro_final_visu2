"""Registro de experimentos en MLflow (base de datos local mlflow.db en la raiz del repo).

Ver los experimentos:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
import subprocess

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = "warehouse-rush"


def setup():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=f"file://{ROOT}/mlartifacts")
    mlflow.set_experiment(EXPERIMENT)


def flatten(params, prefix=""):
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def dataset_hashes():
    """Hash md5 de cada archivo del dataset segun dvc.lock (identifica la version de los datos)."""
    try:
        with open(os.path.join(ROOT, "dvc.lock")) as f:
            lock = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    outs = lock["stages"].get("generar_datos", {}).get("outs", [])
    return {f"datos.md5.{os.path.basename(o['path'])}": o["md5"] for o in outs}


def start_run(name, params, origin):
    """Abre un run de MLflow con los parametros, el commit de git y la version del dataset."""
    setup()
    run = mlflow.start_run(run_name=name)
    mlflow.log_params(flatten(params))
    mlflow.set_tags({
        "origen": origin,
        "git.commit": _git("rev-parse", "HEAD"),
        "git.rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git.cambios_sin_commit": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        **dataset_hashes(),
    })
    return run


def log_curve(row):
    step = row["episodio"]
    mlflow.log_metrics({f"entrenamiento.{k}": v for k, v in row.items() if k != "episodio"}, step=step)


def log_evaluation(metrics):
    mlflow.log_metrics({f"eval.{k}": v for k, v in metrics.items()})
