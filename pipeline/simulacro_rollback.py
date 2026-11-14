"""Simulacro de rollback de extremo a extremo sobre la URL publica (flujo de Prefect).

Escenario: un operador fuerza a produccion un modelo que no cumple los requisitos (por defecto la red PPO de la
semilla 0 de v2-n2: 99.8% de exito pero 0.41 rescates por pedido; los criterios la rechazaron). La vigilancia lo
detecta en la URL y revierte al recomendado anterior.

  1. estado-inicial     sondear la URL: el nivel debe estar en "ok"
  2. publicar           publicar el modelo defectuoso (Hugging Face + manifiesto + Model Registry)
  3. promover-forzado   promoverlo con --forzar, commit + push y verificar que la URL lo sirve
  4. vigilancia         flujo warehouse-rush-vigilancia: detecta la alerta, revierte, publica y verifica
  5. estado-final       sondear la URL: el nivel vuelve a "ok" con el modelo original

Evidencia: docs/ejecuciones/<fecha>-simulacro-rollback-<flujo>.json y docs/resultados/simulacro-rollback.md.

Uso: python pipeline/simulacro_rollback.py [--nivel v2-n2 --run <run_id o nombre>]
"""

import argparse
import datetime
import json
import os
import subprocess

from flujo import URL, publish_catalog, sh, verify_served  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402
from vigilar import probe, watch  # noqa: E402

import tracking  # noqa: E402

ROOT = tracking.ROOT
REPORT = os.path.join(ROOT, "docs", "resultados", "simulacro-rollback.md")


@task(name="2-publicar-defectuoso")
def publish_bad(run_spec):
    import publicar_modelo

    model_id = publicar_modelo.publish(run_spec)
    metrics = publicar_modelo._manifest(model_id)["metricas_evaluacion"]
    get_run_logger().info(f"Publicado {model_id}: exito {metrics['tasa_exito']:.3f}, rescates {metrics['rescates_medios']:.2f}")
    return {"model_id": model_id, "metricas_registradas": {k: metrics.get(k) for k in
                                                            ("tasa_exito", "otif", "rescates_medios", "utilidad_media_cop")}}


@task(name="3-promover-forzado")
def force_promote(model_id, level, reason):
    import publicar_modelo

    before = publicar_modelo.load_catalog()["recomendados"].get(level)
    publicar_modelo.recommend(model_id, force=True)
    catalog = publicar_modelo.load_catalog()
    for m in catalog["modelos"]:
        if m["model_id"] == model_id:
            m["motivo"] = reason
    publicar_modelo.save_catalog(catalog)
    published = publish_catalog.fn(True, f"Simulacro de rollback: promocion forzada de {model_id} en {level}")
    return {"anterior": before, "forzado": model_id, "motivo": reason, "commit": published["commit"]}


def write_report(ev):
    s = ev["etapas"]
    level = ev["parametros"]["nivel"]
    initial = s["1-estado-inicial"]["niveles"][level]
    forced = s["3-promover-forzado"]
    served_bad = s["3b-verificar-forzado"]
    watch_ev = s["4-vigilancia"]["etapas"]
    alarm = watch_ev["1-sondear"]["niveles"][level]
    change = watch_ev["3-revertir"]["cambios"][level]
    final = s["5-estado-final"]["niveles"][level]

    def row(name, r):
        p = r["sonda"]
        return (f"| {name} | `{r['modelo']}` | {r['estado']} | {p['tasa_exito']:.1%} | {p['otif']:.1%} | "
                f"{p['rescates_medios']:.2f} | ${p['utilidad_media_cop']:,.0f} |")

    lines = [
        f"# Simulacro de rollback en {level.upper()}",
        "",
        f"Ejecutado el {ev['inicio'][:16].replace('T', ' ')} UTC sobre {ev['parametros']['url']} con el flujo de Prefect "
        f"`warehouse-rush-simulacro-rollback` (`{ev['flow_run_id']}`). Evidencia completa: "
        f"[`{os.path.relpath(ev['ruta'], os.path.dirname(REPORT))}`]({os.path.relpath(ev['ruta'], os.path.dirname(REPORT))}).",
        "",
        "**Escenario:** un operador fuerza a producción un modelo que los criterios habían rechazado. "
        f"{forced['motivo']} La vigilancia sondea la URL pública, detecta la alerta y revierte al recomendado anterior "
        "sin intervención manual.",
        "",
        f"Sonda: {ev['parametros']['casos']} pedidos de la suite de evaluación de {level}, ejecutados por la API desplegada.",
        "",
        "| Momento | Modelo servido | Estado | Éxito | OTIF | Rescates | Utilidad/pedido |",
        "|---|---|---|---|---|---|---|",
        row("1. Antes", initial),
        row("3. Tras la promoción forzada", alarm),
        row("5. Tras el rollback", final),
        "",
        "## Secuencia",
        "",
        f"1. **Estado inicial:** la URL sirve `{initial['modelo']}` en {level} y la sonda está en `{initial['estado']}` "
        f"(commit `{s['1-estado-inicial']['commit_servido'][:7]}`).",
        f"2. **Publicación:** se publica `{forced['forzado']}` en Hugging Face con su manifiesto.",
        f"3. **Promoción forzada:** `publicar_modelo --recomendar --forzar` → commit `{forced['commit'][:7]}`. "
        f"La URL lo sirve (commit `{served_bad['commit_servido'][:7]}`).",
        f"4. **Vigilancia** (`warehouse-rush-vigilancia`): la sonda marca `{alarm['estado']}`: "
        f"{'; '.join(alarm['alertas'])}. Rollback `{change['desde']}` → `{change['hacia']}` "
        f"(commit `{watch_ev['3-revertir']['commit'][:7]}`); el modelo forzado queda como `revertido` en el catálogo y "
        "en el Model Registry.",
        f"5. **Verificación:** la URL vuelve a servir `{final['modelo']}` y la sonda está en `{final['estado']}`.",
        "",
        f"Duración total: {ev['duracion_s'] / 60:.1f} min (incluye dos despliegues de Vercel).",
        "",
    ]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))


@flow(name="warehouse-rush-simulacro-rollback")
def drill(nivel="v2-n2", run=None, url=URL, casos=50):
    os.chdir(ROOT)
    log = get_run_logger()
    started = datetime.datetime.now(datetime.timezone.utc)
    flow_id = str(flow_run.get_id())
    path = os.path.join(ROOT, "docs", "ejecuciones", f"{started:%Y%m%d-%H%M%S}-simulacro-rollback-{flow_id[:8]}.json")
    ev = {"flujo": "warehouse-rush-simulacro-rollback", "flow_run_id": flow_id, "inicio": started.isoformat(),
          "parametros": {"nivel": nivel, "run": run, "url": url, "casos": casos}, "etapas": {}, "ruta": path}

    def save(status):
        ev.update(estado=status, fin=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  duracion_s=(datetime.datetime.now(datetime.timezone.utc) - started).total_seconds())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: v for k, v in ev.items() if k != "ruta"}, f, indent=2, ensure_ascii=False)

    try:
        initial = ev["etapas"]["1-estado-inicial"] = probe(url, casos, [nivel])
        if initial["niveles"][nivel]["estado"] != "ok":
            raise RuntimeError(f"{nivel} no esta en ok antes del simulacro: {initial['niveles'][nivel]}")
        bad = ev["etapas"]["2-publicar-defectuoso"] = publish_bad(run)
        reg = bad["metricas_registradas"]
        reason = (f"Simulacro: modelo rechazado por los criterios ({reg['rescates_medios']:.2f} rescates por pedido, "
                  f"maximo 0.30) promovido con --forzar.")
        forced = ev["etapas"]["3-promover-forzado"] = force_promote(bad["model_id"], nivel, reason)
        ev["etapas"]["3b-verificar-forzado"] = verify_served(url, forced["commit"], nivel, bad["model_id"])
        log.info("El modelo defectuoso esta en produccion; se lanza la vigilancia")
        ev["etapas"]["4-vigilancia"] = watch(url=url, casos=casos, revertir=True, niveles=[nivel],
                                             publicar_evidencia=False)
        ev["etapas"]["5-estado-final"] = probe(url, casos, [nivel])
        save("completado")
        write_report(ev)
        sh("git", "add", path, REPORT, os.path.join(ROOT, "docs", "ejecuciones"))
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
            sh("git", "commit", "-q", "-m", f"Evidencia del simulacro de rollback {flow_id[:8]} en {nivel}")
            sh("git", "push", "-q", "origin", f"{tracking.git('rev-parse', '--abbrev-ref', 'HEAD')}:main")
    except Exception as error:
        ev["error"] = str(error)
        save("fallido")
        raise
    print(f"Evidencia: {os.path.relpath(path, ROOT)} · informe: {os.path.relpath(REPORT, ROOT)}")
    return ev


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nivel", default="v2-n2")
    parser.add_argument("--run", help="run del modelo defectuoso (id o nombre); por defecto ppo-v2-n2-A-s0")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--casos", type=int, default=50)
    args = parser.parse_args()
    run = args.run
    if run is None:
        with open(os.path.join(ROOT, "runs", "ppo", "ppo-v2-n2-A-s0", "registrado.json")) as f:
            run = json.load(f)["run_id"]
    drill(nivel=args.nivel, run=run, url=args.url, casos=args.casos)


if __name__ == "__main__":
    main()
