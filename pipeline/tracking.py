"""Registro de experimentos en MLflow.

Si MLFLOW_TRACKING_URI esta definida (por ejemplo el MLflow de DagsHub, en el archivo .env), se usa ese
servidor; si no, una base local (mlflow.db). Credenciales: MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD.

Ver los experimentos locales:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
import subprocess

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = "warehouse-rush"
LOCAL_URI = f"sqlite:///{ROOT}/mlflow.db"


def load_dotenv(path=os.path.join(ROOT, ".env")):
    """Carga variables KEY=VALUE de .env sin pisar las que ya existen en el entorno."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tracking_uri():
    load_dotenv()
    return os.environ.get("MLFLOW_TRACKING_URI") or LOCAL_URI


def is_remote():
    return not tracking_uri().startswith("sqlite:")


def setup(uri=None):
    uri = uri or tracking_uri()
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        if uri.startswith("sqlite:"):
            mlflow.create_experiment(EXPERIMENT, artifact_location=f"file://{ROOT}/mlartifacts")
        else:
            mlflow.create_experiment(EXPERIMENT)
    mlflow.set_experiment(EXPERIMENT)
    return uri


def run_url(run):
    """Enlace al run en la interfaz de MLflow (DagsHub o local)."""
    uri = tracking_uri()
    if uri.startswith("sqlite:"):
        return f"http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    return f"{uri.rstrip('/')}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"


def flatten(params, prefix=""):
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# Archivos que el propio pipeline reescribe al ejecutarse: no cuentan como cambios de codigo.
PIPELINE_OUTPUTS = ("dvc.lock", "metrics/", "manifests/", "data/", "models/")


def uncommitted_code_changes():
    """Archivos versionados con cambios sin commit, sin contar las salidas del pipeline."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    files = [line[3:] for line in out.splitlines() if len(line) > 3]
    return [f for f in files if not f.startswith(PIPELINE_OUTPUTS)]


def dataset_hashes(level):
    """Hash md5 de cada archivo del dataset del nivel segun dvc.lock (identifica la version de los datos)."""
    try:
        with open(os.path.join(ROOT, "dvc.lock")) as f:
            lock = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    outs = lock["stages"].get(f"generar_datos@{level}", {}).get("outs", [])
    return {f"datos.md5.{os.path.basename(o['path'])}": o["md5"] for o in outs}


def start_run(name, cfg, origin, extra_tags=None):
    """Abre un run con los parametros resueltos, la configuracion completa y su procedencia."""
    from warehouse_rush import niveles

    setup()
    run = mlflow.start_run(run_name=name)
    mlflow.log_params(flatten(cfg))
    env = niveles.make_env(cfg)
    mlflow.set_tags({
        "origen": origin,
        "nivel": cfg["nivel"],
        "environment_version": cfg["version_ambiente"],
        "familia_modelo": niveles.family(env),
        "config_sha256": niveles.sha256_json(cfg),
        "seed": str(cfg["entrenamiento"]["semilla"]),
        "git.commit": git("rev-parse", "HEAD"),
        "git.rama": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git.cambios_sin_commit": str(bool(changes := uncommitted_code_changes())),
        "git.archivos_modificados": ", ".join(changes)[:500],
        **dataset_hashes(cfg["nivel"]),
        **(extra_tags or {}),
    })
    mlflow.log_dict(cfg, "config.json")
    return run


def log_curve(row):
    step = row["episodio"]
    mlflow.log_metrics({f"entrenamiento.{k}": v for k, v in row.items() if k != "episodio" and not k.startswith("eval_")},
                       step=step)
    mlflow.log_metrics({f"curva_eval.{k[5:]}": v for k, v in row.items() if k.startswith("eval_")}, step=step)


def log_evaluation(metrics, per_case, suite_id):
    mlflow.log_metrics({f"eval.{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    mlflow.log_metrics({f"eval.fallos.{k}": v for k, v in metrics.get("fallos_por_causa", {}).items()})
    mlflow.set_tag("evaluation_suite_id", suite_id)
    mlflow.log_dict(metrics, "evaluacion.json")
    mlflow.log_dict({"suite_id": suite_id, "casos": per_case}, "evaluacion_por_caso.json")
