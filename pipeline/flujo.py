"""Flujo de Prefect que orquesta el pipeline completo (secuencia minima del enunciado).

  1. validar      perfil, esquema, rutas y compatibilidad del ambiente
  2. pruebas      reglas, observaciones, acciones y terminaciones (pytest)
  3. candidato    entrenar con DVC (modo "pipeline") o cargar un checkpoint compatible (modo "checkpoint")
  4. evaluar      suite del nivel, resultados por caso; verifica que coincidan con lo registrado en MLflow
  5. registrar    publicar los pesos en Hugging Face con su manifiesto y hashes (sin duplicar versiones)
  6. criterios    promover solo si cumple configs/criterios.yaml; si no, queda como candidato con su motivo
  7. catalogo     exportar los runs de MLflow; con --publicar, commit y push (Vercel despliega solo)
  8. verificar    comprobar la version servida en la URL; si falla tras una promocion, rollback

Cada etapa deja su estado en Prefect y la evidencia en docs/ejecuciones/<fecha>-<flujo>.json.
Historial de ejecuciones: .venv-prefect/bin/prefect server start  (UI en http://127.0.0.1:4200)

Uso:
    python pipeline/flujo.py --modo checkpoint --run pipeline-n2               # sin entrenar, sin publicar
    python pipeline/flujo.py --modo pipeline --promover --publicar             # ciclo completo
    python pipeline/flujo.py --modo checkpoint --run contraste-n3-A-s1 --nivel n3 --promover --publicar
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")
# Servidor de Prefect en .venv-prefect (ver requirements-prefect-server.txt); el flujo es un cliente HTTP.
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402

import tracking  # noqa: E402
from common import data_path, load_orders, load_params, make_env, resolve_params  # noqa: E402

ROOT = tracking.ROOT
sys.path.insert(0, ROOT)
PYTHON = sys.executable
DVC = os.path.join(os.path.dirname(PYTHON), "dvc")
URL = "https://profinalvisu2.vercel.app"


def sh(*cmd, timeout=3600):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} fallo ({result.returncode}):\n{(result.stdout + result.stderr)[-2000:]}")
    return result.stdout


# ---------------------------------------------------------------- 1. validar

@task(name="1-validar")
def validate(level, mode, run_spec):
    from warehouse_rush import niveles

    log = get_run_logger()
    params = load_params(os.path.join(ROOT, "params.yaml"))
    for key in ("nivel", "agente", "entrenamiento"):
        if key not in params:
            raise ValueError(f"params.yaml no tiene la seccion '{key}'")
    for key in ("alpha", "gamma", "epsilon_min", "fraccion_exploracion"):
        if not 0 <= float(params["agente"][key]) <= 1:
            raise ValueError(f"agente.{key} fuera de [0, 1]")
    if mode == "pipeline" and params["nivel"] != level:
        raise ValueError(f"El modo pipeline entrena el nivel de params.yaml ({params['nivel']}), no {level}")
    cfg = niveles.resolve(level)
    probs = sum(cfg["datos"]["prob_tipos"].values())
    if abs(probs - 1) > 1e-9:
        raise ValueError(f"prob_tipos de {level} suma {probs}")
    env = niveles.make_env(cfg)  # valida bloqueos y construye el ambiente
    reachable = len(env.unwrapped.reachable_shelves)
    if reachable < cfg["datos"]["paquetes_por_pedido"]:
        raise ValueError(f"Solo {reachable} estanterias alcanzables en {level}")
    for name in ("entrenamiento", "evaluacion"):
        path = os.path.join(ROOT, data_path(level, name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta {path} (dvc pull)")
    with open(os.path.join(ROOT, "configs", "criterios.yaml")) as f:
        criteria = yaml.safe_load(f)
    changes = tracking.uncommitted_code_changes()
    if changes:
        log.warning(f"Hay cambios de codigo sin commit: {changes}")
    info = {"nivel": level, "familia": niveles.family(env), "version_ambiente": cfg["version_ambiente"],
            "estanterias_alcanzables": reachable, "criterios_sha256": niveles.sha256_json(criteria),
            "git_commit": tracking.git("rev-parse", "HEAD"), "cambios_sin_commit": changes, "modo": mode, "run": run_spec}
    log.info(f"Perfil valido: {info}")
    return info


# ---------------------------------------------------------------- 2. pruebas

@task(name="2-pruebas")
def run_tests():
    out = sh(PYTHON, "-m", "pytest", "-q")
    summary = out.strip().splitlines()[-1]
    get_run_logger().info(summary)
    return summary


# ---------------------------------------------------------------- 3. candidato

@task(name="3-candidato", retries=1, retry_delay_seconds=30)
def candidate(mode, run_spec, level, family):
    from publicar_modelo import find_run

    log = get_run_logger()
    if mode == "pipeline":
        sh(DVC, "repro", timeout=7200)
        sh(DVC, "push")
        with open(os.path.join(ROOT, "metrics", "mlflow_run.json")) as f:
            run_spec = json.load(f)["run_id"]
        log.info(f"Entrenado con DVC: run {run_spec}")
    run = find_run(run_spec)
    run_family = run.data.tags.get("familia_modelo")
    if run_family and run_family != family:
        raise ValueError(f"El checkpoint {run.info.run_name} ({run_family}) no es compatible con {level} ({family})")
    if run.data.tags.get("nivel", level) != level:
        log.warning(f"El checkpoint se entreno en {run.data.tags.get('nivel')}; se usara en {level} (misma familia)")
    return {"run_id": run.info.run_id, "run_name": run.info.run_name, "nivel_entrenamiento": run.data.tags.get("nivel"),
            "eval_tasa_exito_registrada": run.data.metrics.get("eval.tasa_exito")}


# ---------------------------------------------------------------- 4. evaluar

@task(name="4-evaluar")
def evaluate_suite(cand, level, workdir):
    from publicar_modelo import download_model, find_run
    from warehouse_rush import niveles
    from warehouse_rush.politicas import QTablePolicy
    from warehouse_rush.simulacion import evaluate

    log = get_run_logger()
    cfg = resolve_params(dict(load_params(os.path.join(ROOT, "params.yaml")), nivel=level))
    cases = load_orders(os.path.join(ROOT, data_path(level, "evaluacion")))
    suite_id = f"{level}-{niveles.sha256_json(cases)[:12]}"
    q = np.load(download_model(find_run(cand["run_id"]), os.path.join(workdir, "modelo")))
    metrics, per_case = evaluate(make_env(cfg), QTablePolicy(q), cases)
    with open(os.path.join(workdir, "evaluacion_por_caso.json"), "w") as f:
        json.dump({"suite_id": suite_id, "casos": per_case}, f)
    registered = cand["eval_tasa_exito_registrada"]
    same_level = cand["nivel_entrenamiento"] in (None, level)
    if same_level and registered is not None and abs(registered - metrics["tasa_exito"]) > 1e-9:
        raise ValueError(f"La evaluacion ({metrics['tasa_exito']}) no coincide con la registrada en MLflow ({registered})")
    log.info(f"Suite {suite_id}: exito {metrics['tasa_exito']:.3f}, fallos {metrics['fallos_por_causa']}")
    return {"suite_id": suite_id, "tasa_exito": metrics["tasa_exito"], "fallos_por_causa": metrics["fallos_por_causa"],
            "coincide_con_mlflow": same_level and registered is not None}


# ---------------------------------------------------------------- 5. registrar

@task(name="5-registrar", retries=2, retry_delay_seconds=20)
def register(cand):
    import publicar_modelo

    model_id = publicar_modelo.publish(cand["run_id"])
    manifest = publicar_modelo._manifest(model_id)
    get_run_logger().info(f"Modelo {model_id} · sha256 {manifest['model_sha256'][:12]} · {manifest['model_uri']}")
    return {"model_id": model_id, "model_sha256": manifest["model_sha256"], "model_uri": manifest["model_uri"]}


# ---------------------------------------------------------------- 6. criterios

@task(name="6-criterios")
def apply_criteria(model, level, promote):
    import publicar_modelo

    catalog = publicar_modelo.load_catalog()
    before = catalog["recomendados"].get(level)
    if not promote:
        return {"promovido": False, "motivo": "promocion no solicitada", "recomendado": before}
    if before == model["model_id"]:
        return {"promovido": False, "motivo": "ya era el recomendado", "recomendado": before}
    try:
        publicar_modelo.recommend(model["model_id"])
    except SystemExit as decision:
        get_run_logger().warning(str(decision))
        return {"promovido": False, "motivo": str(decision), "recomendado": before}
    return {"promovido": True, "motivo": "cumple los criterios", "recomendado": model["model_id"], "anterior": before}


# ---------------------------------------------------------------- 7. catalogo

@task(name="7-catalogo", retries=1, retry_delay_seconds=20)
def publish_catalog(publish_git, message):
    sh(PYTHON, os.path.join(ROOT, "pipeline", "exportar_catalogo.py"))
    if not publish_git:
        return {"commit": None, "nota": "catalogo exportado localmente (sin --publicar)"}
    sh("git", "add", "manifests", "metrics", "dvc.lock", "docs")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return {"commit": tracking.git("rev-parse", "HEAD"), "nota": "sin cambios en el catalogo"}
    sh("git", "commit", "-q", "-m", message)
    branch = tracking.git("rev-parse", "--abbrev-ref", "HEAD")
    sh("git", "push", "-q", "origin", f"{branch}:main")
    return {"commit": tracking.git("rev-parse", "HEAD"), "nota": f"push de {branch} a main"}


# ---------------------------------------------------------------- 8. verificar

def _get(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


@task(name="8-verificar")
def verify_served(url, commit, level, expected_model, wait_s=600):
    log = get_run_logger()
    deadline = time.time() + wait_s
    health = {}
    while time.time() < deadline:
        try:
            health = _get(f"{url}/api/salud")
            if commit is None or health["git_commit"] == commit:
                break
        except Exception as error:  # noqa: BLE001
            log.info(f"Esperando el despliegue: {error}")
        time.sleep(15)
    else:
        raise TimeoutError(f"La URL sigue sirviendo {health.get('git_commit')} y se esperaba {commit}")
    served = health.get("modelos_recomendados", {}).get(level)
    if expected_model and served != expected_model:
        raise ValueError(f"La URL recomienda {served} en {level} y se esperaba {expected_model}")
    result = _get(f"{url}/api/evaluar", {"nivel": level, "politica": "modelo", "casos": 20})
    log.info(f"Servido: commit {health['git_commit'][:7]}, modelo {served}, exito en 20 casos {result['metricas']['tasa_exito']}")
    return {"commit_servido": health["git_commit"], "modelo_servido": served,
            "exito_20_casos": result["metricas"]["tasa_exito"], "latencia_media_ms": result["metricas"]["latencia_media_ms"]}


@task(name="8b-rollback")
def rollback(level, publish_git):
    import publicar_modelo

    publicar_modelo.rollback(level)
    return publish_catalog.fn(publish_git, f"Rollback automatico del recomendado de {level}")


# ---------------------------------------------------------------- flujo

@flow(name="warehouse-rush-pipeline", log_prints=True)
def pipeline(nivel="n2", modo="checkpoint", run=None, promover=False, publicar=False, url=URL):
    os.chdir(ROOT)
    started = datetime.datetime.now(datetime.timezone.utc)
    evidence = {"flujo": "warehouse-rush-pipeline", "flow_run_id": str(flow_run.get_id()), "inicio": started.isoformat(),
                "parametros": {"nivel": nivel, "modo": modo, "run": run, "promover": promover, "publicar": publicar},
                "etapas": {}}
    path = os.path.join(ROOT, "docs", "ejecuciones", f"{started:%Y%m%d-%H%M%S}-{evidence['flow_run_id'][:8]}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(status):
        evidence["estado"] = status
        evidence["fin"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)

    try:
        with tempfile.TemporaryDirectory() as workdir:
            info = evidence["etapas"]["1-validar"] = validate(nivel, modo, run)
            evidence["etapas"]["2-pruebas"] = run_tests()
            cand = evidence["etapas"]["3-candidato"] = candidate(modo, run, nivel, info["familia"])
            evidence["etapas"]["4-evaluar"] = evaluate_suite(cand, nivel, workdir)
            model = evidence["etapas"]["5-registrar"] = register(cand)
            decision = evidence["etapas"]["6-criterios"] = apply_criteria(model, nivel, promover)
        save("verificando")
        catalog = evidence["etapas"]["7-catalogo"] = publish_catalog(
            publicar, f"Flujo Prefect {evidence['flow_run_id'][:8]}: {model['model_id']} en {nivel} "
                      f"({'promovido' if decision['promovido'] else 'no promovido'})")
        if publicar:
            try:
                evidence["etapas"]["8-verificar"] = verify_served(url, catalog["commit"], nivel, decision["recomendado"])
            except Exception as error:
                evidence["etapas"]["8-verificar"] = {"error": str(error)}
                if decision["promovido"]:
                    evidence["etapas"]["8b-rollback"] = rollback(nivel, publicar)
                raise
        else:
            evidence["etapas"]["8-verificar"] = {"nota": "sin --publicar: se verifica la URL actual sin esperar commit",
                                                 **verify_served(url, None, nivel, None, wait_s=60)}
        save("completado")
    except Exception as error:
        evidence["error"] = str(error)
        save("fallido")
        raise
    print(f"Evidencia: {os.path.relpath(path, ROOT)}")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nivel", default="n2")
    parser.add_argument("--modo", choices=["pipeline", "checkpoint"], default="checkpoint")
    parser.add_argument("--run", help="run de MLflow a usar en modo checkpoint (nombre o id)")
    parser.add_argument("--promover", action="store_true", help="promover si cumple los criterios")
    parser.add_argument("--publicar", action="store_true", help="commit + push del catalogo y verificar la URL")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    if args.modo == "checkpoint" and not args.run:
        raise SystemExit("--modo checkpoint necesita --run")
    pipeline(args.nivel, args.modo, args.run, args.promover, args.publicar, args.url)
