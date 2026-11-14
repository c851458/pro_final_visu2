"""Aplica los criterios fijados (configs/criterios.yaml) a los finalistas de un contraste y registra la decision.

1. Agrupa los runs del contraste por brazo (3 semillas cada uno) y calcula media y desviacion estandar.
2. Aplica los requisitos y la regla de seleccion fijados de antemano; elige la semilla a publicar.
3. Con --test-reservado, evalua UNA vez todos los modelos en el test final reservado (semilla secreta en .env).
4. Escribe docs/resultados/<contraste>.md y .json, y un run "decision-<contraste>" en MLflow.

Uso: python pipeline/comparar_finalistas.py contraste-n3 [--test-reservado]
"""

import argparse
import json
import os
import statistics
import tempfile

import mlflow
import numpy as np
import yaml

import tracking
from common import load_params, make_env, resolve_params
from warehouse_rush import niveles
from warehouse_rush.modelos import load_policy
from warehouse_rush.simulacion import evaluate

METRICS = ("tasa_exito", "pasos_medios_exitosos", "rescates_medios", "fragiles_rotos_medios", "choques_medios")
OUT_DIR = os.path.join(tracking.ROOT, "docs", "resultados")


def mean_std(values):
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize_arm(runs):
    summary = {"runs": [], "n_semillas": len(runs)}
    for run in sorted(runs, key=lambda r: int(r.data.tags.get("seed", 0))):
        m = run.data.metrics
        summary["runs"].append({"run_id": run.info.run_id, "nombre": run.info.run_name, "semilla": int(run.data.tags["seed"]),
                                "episodios": int(run.data.params.get("entrenamiento.episodios") or run.data.params["entrenamiento.pasos"]),
                                "tiempo_s": m.get("tiempo_entrenamiento_s"),
                                **{k: m.get(f"eval.{k}") for k in METRICS}})
    for k in METRICS + ("tiempo_s",):
        summary[k] = mean_std([r[k] for r in summary["runs"]])
    summary["episodios"] = summary["runs"][0]["episodios"]  # presupuesto (la clave se conserva por compatibilidad)
    params = runs[0].data.params
    summary["unidad_presupuesto"] = "episodios" if params.get("entrenamiento.episodios") else "pasos del ambiente"
    return summary


def decide(arms, criteria):
    req = criteria["requisitos"]
    for arm in arms.values():
        reasons = []
        if arm["tasa_exito"][0] < req["tasa_exito_min"]:
            reasons.append(f"tasa_exito media {arm['tasa_exito'][0]:.3f} < {req['tasa_exito_min']}")
        if arm["fragiles_rotos_medios"][0] > req["fragiles_rotos_medios_max"]:
            reasons.append(f"fragiles rotos {arm['fragiles_rotos_medios'][0]:.3f} > {req['fragiles_rotos_medios_max']}")
        if arm["rescates_medios"][0] > req["rescates_medios_max"]:
            reasons.append(f"rescates {arm['rescates_medios'][0]:.3f} > {req['rescates_medios_max']}")
        arm["cumple_requisitos"] = not reasons
        arm["incumplimientos"] = reasons
    passing = {k: a for k, a in arms.items() if a["cumple_requisitos"]}
    if not passing:
        return None, "Ningun brazo cumple los requisitos: se mantiene el modelo actual."
    best = max(passing.values(), key=lambda a: a["tasa_exito"][0])
    tied = [k for k, a in passing.items() if best["tasa_exito"][0] - a["tasa_exito"][0] < 0.01]
    if len(tied) > 1:
        tied.sort(key=lambda k: (passing[k]["episodios"], passing[k]["pasos_medios_exitosos"][0]))
        return tied[0], (f"Empate en exito (<1 pp) entre {', '.join(tied)}: gana {tied[0]} por menor presupuesto "
                         f"({passing[tied[0]]['episodios']} {passing[tied[0]].get('unidad_presupuesto', 'episodios')}).")
    winner = next(k for k, a in passing.items() if a is best)
    return winner, f"Gana {winner} por mayor exito medio ({best['tasa_exito'][0] * 100:.1f}%)."


def reserved_test(arms, level, n_cases):
    tracking.load_dotenv()
    seed = os.environ.get("TEST_RESERVADO_SEMILLA")
    if not seed:
        raise SystemExit("Falta TEST_RESERVADO_SEMILLA en .env")
    cfg = resolve_params(dict(load_params(), nivel=level))
    cases, suite_id = niveles.reserved_suite(cfg, int(seed), n_cases)
    from publicar_modelo import download_model, find_run

    with tempfile.TemporaryDirectory() as tmp:
        for arm in arms.values():
            for r in arm["runs"]:
                policy = load_policy(download_model(find_run(r["run_id"]), os.path.join(tmp, r["run_id"])))
                metrics, _ = evaluate(make_env(cfg), policy, cases)
                r["test_reservado"] = {k: metrics[k] for k in METRICS}
            arm["test_reservado_tasa_exito"] = mean_std([r["test_reservado"]["tasa_exito"] for r in arm["runs"]])
    return suite_id


def write_report(name, contrast, arms, winner, reason, selected, suite_reserved):
    os.makedirs(OUT_DIR, exist_ok=True)
    pct = lambda ms: f"{ms[0] * 100:.1f} % ± {ms[1] * 100:.1f}"
    num = lambda ms, d=2: f"{ms[0]:.{d}f} ± {ms[1]:.{d}f}"
    lines = [f"# {name}: {contrast['pregunta'].strip()}", "",
             "Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). "
             "Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel "
             f"{contrast['nivel'].upper()} (500 pedidos).", "",
             "| Brazo | Presupuesto | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |",
             "|---|---|---|---|---|---|---|---|"]
    for k, a in arms.items():
        lines.append(f"| **{k}** · {a['nombre']} | {a['episodios']:,} {a.get('unidad_presupuesto', 'episodios')} | "
                     f"{pct(a['tasa_exito'])} | "
                     f"{num(a['pasos_medios_exitosos'], 1)} | {num(a['rescates_medios'])} | "
                     f"{num(a['fragiles_rotos_medios'])} | {num(a['choques_medios'])} | "
                     f"{'sí' if a['cumple_requisitos'] else 'no: ' + '; '.join(a['incumplimientos'])} |")
    lines += ["", "## Por semilla", "", "| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |", "|---|---|---|---|---|---|"]
    for a in arms.values():
        for r in a["runs"]:
            lines.append(f"| `{r['nombre']}` | {r['semilla']} | {r['tasa_exito'] * 100:.1f} % | "
                         f"{r['pasos_medios_exitosos']:.1f} | {r['rescates_medios']:.3f} | {r['tiempo_s']:.0f} s |")
    lines += ["", "## Decisión", "", reason]
    if selected:
        lines.append(f"Modelo a publicar (mejor semilla del brazo ganador): `{selected['nombre']}` "
                     f"(run `{selected['run_id']}`, {selected['tasa_exito'] * 100:.1f} %).")
    if suite_reserved:
        lines += ["", "## Test final reservado", "",
                  f"Suite `{suite_reserved}` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de "
                  "la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.", "",
                  "| Brazo | Éxito en test reservado |", "|---|---|"]
        lines += [f"| **{k}** · {a['nombre']} | {pct(a['test_reservado_tasa_exito'])} |" for k, a in arms.items()]
    path_md = os.path.join(OUT_DIR, f"{name}.md")
    with open(path_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump({"contraste": name, "brazos": arms, "ganador": winner, "motivo": reason, "seleccionado": selected,
                   "test_reservado_suite_id": suite_reserved}, f, indent=2, ensure_ascii=False)
    return path_md


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("contraste")
    parser.add_argument("--test-reservado", action="store_true", help="evaluar una vez en el test final reservado")
    args = parser.parse_args()
    with open(os.path.join(tracking.ROOT, "configs", "criterios.yaml")) as f:
        criteria = yaml.safe_load(f)
    contrast = criteria["contrastes"][args.contraste]

    tracking.setup()
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], output_format="list", max_results=1000,
                              filter_string=f"tags.grupo = '{args.contraste}' and attributes.status = 'FINISHED'")
    arms = {}
    for arm_id, arm_cfg in contrast["brazos"].items():
        arm_runs = [r for r in runs if r.data.tags.get("brazo") == arm_id]
        if len(arm_runs) != len(contrast["semillas"]):
            raise SystemExit(f"El brazo {arm_id} tiene {len(arm_runs)} runs terminados; se esperaban {len(contrast['semillas'])}")
        arms[arm_id] = dict(summarize_arm(arm_runs), nombre=arm_cfg["nombre"], hipotesis=arm_cfg["hipotesis"])

    winner, reason = decide(arms, criteria)
    selected = None
    if winner:
        selected = max(arms[winner]["runs"], key=lambda r: (r["tasa_exito"], -r["pasos_medios_exitosos"]))
    suite_reserved = reserved_test(arms, contrast["nivel"], criteria["test_reservado"]["casos"]) if args.test_reservado else None
    report = write_report(args.contraste, contrast, arms, winner, reason, selected, suite_reserved)

    cfg = resolve_params(dict(load_params(), nivel=contrast["nivel"]))
    tags = {"tipo": "decision", "grupo": args.contraste, "ganador": winner or "ninguno", "motivo_decision": reason,
            "criterios_sha256": niveles.sha256_json(criteria), "estado_modelo": "sin pesos (decision)",
            "hipotesis": contrast["pregunta"].strip(), "cambios": f"comparacion de brazos {', '.join(arms)}"}
    if selected:
        tags["run_seleccionado"] = selected["run_id"]
    if suite_reserved:
        tags["test_reservado_suite_id"] = suite_reserved
    with tracking.start_run(f"decision-{args.contraste}", cfg, origin="pipeline/comparar_finalistas.py", extra_tags=tags):
        for k, a in arms.items():
            mlflow.log_metrics({f"{k}.tasa_exito_media": a["tasa_exito"][0], f"{k}.tasa_exito_std": a["tasa_exito"][1]})
            if suite_reserved:
                mlflow.log_metric(f"{k}.test_reservado_tasa_exito_media", a["test_reservado_tasa_exito"][0])
        mlflow.log_artifact(report)
        mlflow.log_artifact(report.replace(".md", ".json"))
    print(open(report).read())


if __name__ == "__main__":
    main()
