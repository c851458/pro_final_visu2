"""Flujo de Prefect de vigilancia: sondea el modelo servido en la URL publica y revierte los niveles en alerta.

  1. sondear    GET /api/monitoreo de la URL: el modelo recomendado de cada nivel se ejecuta sobre pedidos de su
                suite y se compara con los requisitos de promocion y con lo registrado al publicarlo
  2. decidir    niveles en alerta o con error de inferencia; se revierten los que tienen un recomendado anterior
  3. revertir   publicar_modelo.rollback por nivel (catalogo y alias del Model Registry) y commit + push
  4. verificar  esperar a que Vercel sirva el commit y volver a sondear: el nivel debe quedar en "ok"

La evidencia queda en docs/ejecuciones/<fecha>-vigilancia-<flujo>.json (se sube a main si hubo rollback).

Uso:
    python pipeline/vigilar.py                     # sondea y revierte lo que este en alerta
    python pipeline/vigilar.py --solo-observar     # sondea sin revertir
"""

import argparse
import datetime
import json
import os
import subprocess
import time

from flujo import URL, _get, publish_catalog, sh  # noqa: E402  (configura Prefect y las rutas)
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402

import tracking  # noqa: E402

ROOT = tracking.ROOT
PROBE_FIELDS = ("tasa_exito", "otif", "rescates_medios", "fragiles_rotos_medios", "utilidad_media_cop", "casos")


def summarize(report, levels=None):
    """Resumen por nivel del informe de /api/monitoreo (estado, modelo servido, sonda y alertas)."""
    return {n["nivel"]: {"estado": n["estado"], "modelo": (n.get("modelo") or {}).get("model_id"),
                         "sonda": {k: (n.get("sonda") or {}).get(k) for k in PROBE_FIELDS},
                         "alertas": n["alertas"], "anteriores": n["historial"]}
            for n in report["niveles"] if levels is None or n["nivel"] in levels}


@task(name="1-sondear", retries=2, retry_delay_seconds=20)
def probe(url, cases, levels=None):
    report = _get(f"{url}/api/monitoreo?casos={cases}", timeout=120)
    result = summarize(report, levels)
    for level, r in result.items():
        get_run_logger().info(f"{level}: {r['estado']} ({r['modelo']}) {'; '.join(r['alertas'])}")
    return {"commit_servido": report["git_commit"], "generado": report["generado"], "niveles": result}


@task(name="2-decidir")
def decide(state, revert):
    log = get_run_logger()
    bad = {lv: r for lv, r in state["niveles"].items() if r["estado"] in ("alerta", "error")}
    to_revert = [lv for lv, r in bad.items() if r["anteriores"]] if revert else []
    manual = [lv for lv in bad if lv not in to_revert]
    for lv in to_revert:
        log.warning(f"{lv}: {bad[lv]['estado']} -> rollback a {bad[lv]['anteriores'][-1]}")
    for lv in manual:
        log.warning(f"{lv}: {bad[lv]['estado']} sin rollback automatico "
                    f"({'no hay un recomendado anterior' if revert else 'modo solo observar'})")
    return {"revertir": to_revert, "requieren_intervencion": manual,
            "motivos": {lv: bad[lv]["alertas"] for lv in bad}}


@task(name="3-revertir")
def revert_levels(levels, reasons):
    import publicar_modelo

    changes = {}
    for level in levels:
        before = publicar_modelo.load_catalog()["recomendados"][level]
        publicar_modelo.rollback(level)
        after = publicar_modelo.load_catalog()["recomendados"][level]
        changes[level] = {"desde": before, "hacia": after, "motivo": "; ".join(reasons[level])}
    detail = ", ".join(f"{lv} {c['desde']} -> {c['hacia']}" for lv, c in changes.items())
    catalog = publish_catalog.fn(True, f"Rollback por vigilancia: {detail}")
    return {"cambios": changes, "commit": catalog["commit"]}


@task(name="4-verificar")
def verify(url, commit, changes, cases, wait_s=600):
    log = get_run_logger()
    deadline = time.time() + wait_s
    while True:
        try:
            if _get(f"{url}/api/salud")["git_commit"] == commit:
                break
        except Exception as error:  # noqa: BLE001
            log.info(f"Esperando el despliegue: {error}")
        if time.time() > deadline:
            raise TimeoutError(f"La URL no sirve {commit} tras {wait_s} s")
        time.sleep(15)
    after = probe.fn(url, cases, list(changes))
    for level, change in changes.items():
        served = after["niveles"][level]
        if served["modelo"] != change["hacia"]:
            raise ValueError(f"{level}: la URL sirve {served['modelo']} y se esperaba {change['hacia']}")
        if served["estado"] != "ok":
            raise ValueError(f"{level}: tras el rollback sigue en {served['estado']}: {served['alertas']}")
        log.info(f"{level}: la URL sirve {served['modelo']} y esta en ok")
    return after


@flow(name="warehouse-rush-vigilancia")
def watch(url=URL, casos=50, revertir=True, niveles=None, publicar_evidencia=True):
    os.chdir(ROOT)
    started = datetime.datetime.now(datetime.timezone.utc)
    evidence = {"flujo": "warehouse-rush-vigilancia", "flow_run_id": str(flow_run.get_id()),
                "inicio": started.isoformat(), "parametros": {"url": url, "casos": casos, "revertir": revertir,
                                                              "niveles": niveles}, "etapas": {}}
    path = os.path.join(ROOT, "docs", "ejecuciones",
                        f"{started:%Y%m%d-%H%M%S}-vigilancia-{evidence['flow_run_id'][:8]}.json")

    def save(status):
        evidence.update(estado=status, fin=datetime.datetime.now(datetime.timezone.utc).isoformat())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)

    try:
        state = evidence["etapas"]["1-sondear"] = probe(url, casos, niveles)
        decision = evidence["etapas"]["2-decidir"] = decide(state, revertir)
        if decision["revertir"]:
            done = evidence["etapas"]["3-revertir"] = revert_levels(decision["revertir"], decision["motivos"])
            evidence["etapas"]["4-verificar"] = verify(url, done["commit"], done["cambios"], casos)
        save("completado")
        if decision["revertir"] and publicar_evidencia:
            sh("git", "add", path)
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
                sh("git", "commit", "-q", "-m", f"Evidencia de la vigilancia {evidence['flow_run_id'][:8]}")
                sh("git", "push", "-q", "origin", f"{tracking.git('rev-parse', '--abbrev-ref', 'HEAD')}:main")
    except Exception as error:
        evidence["error"] = str(error)
        save("fallido")
        raise
    print(f"Evidencia: {os.path.relpath(path, ROOT)}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--casos", type=int, default=50)
    parser.add_argument("--nivel", action="append", help="limitar a estos niveles (por defecto todos)")
    parser.add_argument("--solo-observar", action="store_true", help="no revertir; solo informar")
    args = parser.parse_args()
    watch(url=args.url, casos=args.casos, revertir=not args.solo_observar, niveles=args.nivel)


if __name__ == "__main__":
    main()
