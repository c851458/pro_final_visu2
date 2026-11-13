#!/usr/bin/env python3
"""Corre experimentos (variantes de params.yaml sobre un nivel) y los registra en MLflow.

Cada experimento entrena y evalua con el mismo codigo del pipeline y queda como un run de MLflow con su
hipotesis, parametros resueltos, curvas de entrenamiento y de evaluacion, metricas, resultados por caso,
modelo, commit de git, version del ambiente y del dataset. Los intentos que no mejoran tambien se conservan.

Uso:
    ./experimentar.py -e alpha-alto agente.alpha=0.2 --hipotesis "Aprender mas rapido converge antes"
    ./experimentar.py -e n1-base nivel=n1 -e n3-base nivel=n3 -j 2
    ./experimentar.py -e n3-intervencion nivel=n3 --desde pipeline-n2      # continua desde otro run
    ./experimentar.py --baselines                                          # aleatoria y regla, todos los niveles
    ./experimentar.py --evaluar pipeline-n2 --nivel n3                     # modelo congelado en otro nivel (incidente)
    ./experimentar.py --contraste contraste-n3                             # finalistas x semillas de configs/criterios.yaml
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import tracking  # noqa: E402  (antes que mlflow: configura su entorno)
import mlflow  # noqa: E402
from common import data_path, load_orders, load_params, make_env, resolve_params  # noqa: E402
from core import train  # noqa: E402
from warehouse_rush import niveles  # noqa: E402
from warehouse_rush.politicas import QTablePolicy, RandomPolicy, RulePolicy  # noqa: E402
from warehouse_rush.simulacion import evaluate  # noqa: E402


def resolve_with_overrides(overrides):
    """params.yaml + cambios. `nivel=...` se aplica primero; el resto sobre la configuracion resuelta."""
    params = load_params()
    for item in overrides:
        if item.startswith("nivel="):
            params["nivel"] = item.partition("=")[2]
    cfg = resolve_params(params)
    for item in overrides:
        if item.startswith("nivel="):
            continue
        key, _, raw = item.partition("=")
        node = cfg
        *path, last = key.split(".")
        for part in path:
            if part not in node:
                raise SystemExit(f"Parametro desconocido: {key}")
            node = node[part]
        if last not in node and key not in ("agente.epsilon_inicial",):
            raise SystemExit(f"Parametro desconocido: {key}")
        node[last] = yaml.safe_load(raw)
    return cfg


def dataset(cfg, overrides):
    """Usa el dataset versionado con DVC del nivel; si se cambian parametros de datos, lo genera aparte."""
    files = [data_path(cfg["nivel"], "entrenamiento"), data_path(cfg["nivel"], "evaluacion")]
    if not any(o.startswith("datos.") for o in overrides) and all(os.path.exists(f) for f in files):
        return load_orders(files[0]), load_orders(files[1]), "dvc"
    env = make_env(cfg)
    training, evaluation = niveles.generate_orders(cfg["datos"], env.unwrapped.reachable_shelves)
    return training, evaluation, "generado"


def run_experiment(job):
    """Entrena y evalua en un proceso aparte; el registro en MLflow lo hace el proceso principal."""
    name, overrides, initial_model, workdir = job["name"], job["overrides"], job["initial_model"], job["workdir"]
    cfg = resolve_with_overrides(overrides)
    training_orders, eval_orders, source = dataset(cfg, overrides)
    curve_cases = eval_orders[:cfg["entrenamiento"]["casos_curva"]]
    initial_q = np.load(initial_model) if initial_model else None
    start = time.time()
    agent, curve = train(make_env(cfg), cfg, training_orders, curve_cases, initial_q=initial_q)
    elapsed = time.time() - start
    metrics, per_case = evaluate(make_env(cfg), QTablePolicy(agent.q), eval_orders)
    model_path = os.path.join(workdir, name, "q_table.npy")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"  terminado: {name} (exito {metrics['tasa_exito'] * 100:.1f}%)", flush=True)
    return {"job": job, "name": name, "overrides": overrides, "cfg": cfg, "curve": curve, "metrics": metrics,
            "per_case": per_case, "model_path": model_path, "dataset": source, "tiempo": elapsed,
            "suite_id": f"{cfg['nivel']}-{niveles.sha256_json(eval_orders)[:12]}"}


def log_to_mlflow(result):
    job = result["job"]
    workdir, parent = job["workdir"], job["parent"]
    tags = {"cambios": " ".join(result["overrides"]) or "(base)", "dataset": result["dataset"],
            "hipotesis": job["hypothesis"] or "", "estado_modelo": "checkpoint final disponible", **job["tags"]}
    if parent:
        tags.update({"parent_run_id": parent["run_id"], "origen_modelo": f"continuacion de {parent['run_name']}"})
    else:
        tags["origen_modelo"] = "desde cero"
    with tracking.start_run(result["name"], result["cfg"], origin="experimentar.py", extra_tags=tags) as run:
        for row in result["curve"]:
            tracking.log_curve(row)
        mlflow.log_metric("tiempo_entrenamiento_s", result["tiempo"])
        tracking.log_evaluation(result["metrics"], result["per_case"], result["suite_id"])
        folder = os.path.join(workdir, result["name"])
        with open(os.path.join(folder, "entrenamiento.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["curve"][0]))
            writer.writeheader()
            writer.writerows(result["curve"])
        mlflow.log_artifact(result["model_path"], "modelo")
        mlflow.log_artifact(os.path.join(folder, "entrenamiento.csv"))
        return run.info.run_id


def log_failed(job, error):
    """Un experimento que falla antes de producir pesos queda registrado con su diagnostico."""
    try:
        cfg = resolve_with_overrides(job["overrides"])
    except SystemExit:
        return
    tags = {"cambios": " ".join(job["overrides"]), "hipotesis": job["hypothesis"] or "",
            "estado_modelo": "sin checkpoint disponible", "error": str(error)[:500], **job["tags"]}
    with tracking.start_run(job["name"], cfg, origin="experimentar.py", extra_tags=tags):
        mlflow.end_run(status="FAILED")


def download_parent_model(spec, workdir):
    """Modelo inicial para continuar el entrenamiento: nombre o id de un run de MLflow."""
    from publicar_modelo import find_run, download_model

    parent = find_run(spec)
    return parent, download_model(parent, os.path.join(workdir, f"padre-{parent.info.run_id}"))


def make_job(name, overrides, workdir, hypothesis=None, parent_spec=None, tags=None):
    job = {"name": name, "overrides": list(overrides), "workdir": workdir, "hypothesis": hypothesis,
           "tags": tags or {}, "initial_model": None, "parent": None}
    if parent_spec:
        parent, path = download_parent_model(parent_spec, workdir)
        job.update(initial_model=path, parent={"run_id": parent.info.run_id, "run_name": parent.info.run_name})
    return job


def execute(jobs, processes):
    """Entrena los trabajos en paralelo y registra cada uno en MLflow en cuanto termina.

    Si un proceso muere (por ejemplo, sin memoria), ProcessPoolExecutor lo detecta en vez de quedarse esperando:
    los trabajos afectados quedan registrados como fallidos con su diagnostico y los ya terminados no se pierden.
    """
    for job in jobs:  # valida los parametros antes de lanzar procesos
        resolve_with_overrides(job["overrides"])
    print(f"Corriendo {len(jobs)} experimento(s) con {min(processes, len(jobs))} proceso(s)...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=min(processes, len(jobs))) as pool:
        futures = {pool.submit(run_experiment, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except BrokenProcessPool as error:
                print(f"  FALLO: {job['name']}: el proceso de entrenamiento murio ({error})", flush=True)
                log_failed(job, f"proceso de entrenamiento terminado abruptamente (posible falta de memoria): {error}")
                continue
            except Exception as error:  # noqa: BLE001  (se registra el intento fallido)
                print(f"  FALLO: {job['name']}: {error}", flush=True)
                log_failed(job, error)
                continue
            result["run_id"] = log_to_mlflow(result)
            print(f"  registrado en MLflow: {job['name']} -> {result['run_id']}", flush=True)
            results.append(result)
    print(f"\n{'experimento':26s} {'nivel':5s} {'exito':>7s} {'pasos':>7s} {'rescates':>9s}  run_id")
    for r in sorted(results, key=lambda r: -r["metrics"]["tasa_exito"]):
        m = r["metrics"]
        print(f"{r['name']:26s} {r['cfg']['nivel']:5s} {m['tasa_exito'] * 100:6.1f}% {m['pasos_medios']:7.1f} "
              f"{m['rescates_medios']:9.3f}  {r['run_id']}")
    return results


def contrast_jobs(name, workdir):
    """Brazos x semillas de un contraste definido en configs/criterios.yaml (hipotesis fijadas de antemano)."""
    with open(os.path.join(ROOT, "configs", "criterios.yaml")) as f:
        criteria = yaml.safe_load(f)
    contrast = criteria["contrastes"][name]
    jobs = []
    for arm_id, arm in contrast["brazos"].items():
        for seed in contrast["semillas"]:
            jobs.append(make_job(
                f"{name}-{arm_id}-s{seed}", arm["cambios"] + [f"entrenamiento.semilla={seed}"], workdir,
                hypothesis=arm["hipotesis"], parent_spec=arm.get("desde"),
                tags={"grupo": name, "brazo": arm_id, "brazo_nombre": arm["nombre"], "tipo": "finalista",
                      "criterios_sha256": niveles.sha256_json(criteria)}))
    return jobs


def run_baselines(levels=None):
    """Registra las referencias aleatoria y de regla en cada nivel (evaluadas en la misma suite)."""
    tracking.setup()
    for level in levels or niveles.level_ids():
        cfg = resolve_params(dict(load_params(), nivel=level))
        cases, suite_id = niveles.evaluation_suite(cfg)
        for policy in (RandomPolicy(seed=0), RulePolicy()):
            metrics, per_case = evaluate(make_env(cfg), policy, cases)
            name = f"referencia-{policy.kind}-{level}"
            tags = {"tipo": "referencia", "politica": policy.kind, "estado_modelo": "sin pesos (politica fija)",
                    "hipotesis": "Referencia para comparar: lo que logra una politica sin aprendizaje."}
            with tracking.start_run(name, cfg, origin="experimentar.py --baselines", extra_tags=tags):
                tracking.log_evaluation(metrics, per_case, suite_id)
            print(f"{name:32s} exito {metrics['tasa_exito'] * 100:5.1f}%  pasos {metrics['pasos_medios']:6.1f}")


def run_cross_evaluation(run_spec, level, hypothesis):
    """Evalua un modelo ya entrenado (congelado) en la suite de otro nivel compatible: mide degradacion."""
    from publicar_modelo import find_run

    source = find_run(run_spec)
    cfg = resolve_params(dict(load_params(), nivel=level))
    env = make_env(cfg)
    family = niveles.family(env)
    if source.data.tags.get("familia_modelo", family) != family:
        raise SystemExit(f"El modelo de {source.info.run_name} no es compatible con el nivel {level}")
    cases, suite_id = niveles.evaluation_suite(cfg)
    with tempfile.TemporaryDirectory() as workdir:
        _, model_path = download_parent_model(run_spec, workdir)
        metrics, per_case = evaluate(env, QTablePolicy(np.load(model_path)), cases)
        name = f"evaluacion-{source.info.run_name}-en-{level}"
        tags = {"tipo": "evaluacion", "parent_run_id": source.info.run_id, "estado_modelo": "pesos del run de origen",
                "origen_modelo": f"modelo congelado de {source.info.run_name} (nivel {source.data.tags.get('nivel', 'n2')})",
                "hipotesis": hypothesis or f"Medir como se comporta el modelo de {source.info.run_name} en {level}.",
                "cambios": f"evaluar en nivel={level}"}
        with tracking.start_run(name, cfg, origin="experimentar.py --evaluar", extra_tags=tags):
            tracking.log_evaluation(metrics, per_case, suite_id)
            mlflow.log_artifact(model_path, "modelo")
    print(f"{name}: exito {metrics['tasa_exito'] * 100:.1f}% | fallos {metrics['fallos_por_causa']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--experimento", nargs="+", action="append", metavar=("NOMBRE", "PARAM=VALOR"),
                        help="nombre del experimento y cambios (nivel=n3, agente.alpha=0.2, ...); se puede repetir")
    parser.add_argument("--hipotesis", help="hipotesis del experimento (se guarda como tag en MLflow)")
    parser.add_argument("--desde", help="continuar el entrenamiento desde el modelo de otro run (nombre o id)")
    parser.add_argument("--baselines", nargs="*", metavar="NIVEL",
                        help="registrar las referencias aleatoria y de regla (en todos los niveles o en los indicados)")
    parser.add_argument("--evaluar", metavar="RUN", help="evaluar el modelo de un run (congelado) en --nivel")
    parser.add_argument("--nivel", help="nivel para --evaluar")
    parser.add_argument("--contraste", help="corre los brazos y semillas de un contraste de configs/criterios.yaml")
    parser.add_argument("-j", "--procesos", type=int, default=4, help="experimentos en paralelo")
    args = parser.parse_args()
    os.chdir(ROOT)
    tracking.setup()
    print(f"MLflow: {tracking.tracking_uri()}")

    if args.baselines is not None:
        run_baselines(args.baselines)
    if args.evaluar:
        if not args.nivel:
            raise SystemExit("--evaluar necesita --nivel")
        run_cross_evaluation(args.evaluar, args.nivel, args.hipotesis)
    if not (args.experimento or args.contraste):
        return
    with tempfile.TemporaryDirectory() as workdir:
        if args.contraste:
            jobs = contrast_jobs(args.contraste, workdir)
        else:
            names = [e[0] for e in args.experimento]
            if len(set(names)) != len(names):
                raise SystemExit("Los nombres de experimento deben ser distintos")
            jobs = [make_job(e[0], e[1:], workdir, args.hipotesis, args.desde) for e in args.experimento]
        execute(jobs, args.procesos)

if __name__ == "__main__":
    main()
