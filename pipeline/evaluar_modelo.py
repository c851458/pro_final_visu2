"""Reconstruye la evaluacion de un modelo publicado a partir de su manifiesto, sin credenciales.

Pasos (los mismos que sigue la API desplegada):
  1. lee manifests/modelos/<model_id>.json (pesos, SHA-256, nivel, version del ambiente y familia);
  2. descarga los pesos de Hugging Face (URL inmutable con el commit del repositorio de modelos);
  3. verifica el SHA-256 contra el manifiesto;
  4. reconstruye el ambiente y la suite de evaluacion del nivel desde configs/niveles.yaml y comprueba que el
     id de la suite y la version del ambiente coinciden con los del manifiesto;
  5. evalua el modelo congelado (sin exploracion) y compara las metricas con las registradas al publicarlo.

Uso:
    .venv/bin/python pipeline/evaluar_modelo.py v3-n2-imitacion-v3-n2-A-s2-4cc5fefd
    .venv/bin/python pipeline/evaluar_modelo.py n2-pipeline-n2-92bff281 --casos 100
    .venv/bin/python pipeline/evaluar_modelo.py n2-pipeline-n2-92bff281 --nivel n3   # otro nivel de la misma familia
    .venv/bin/python pipeline/evaluar_modelo.py --recomendados                       # los recomendados de cada nivel
    .venv/bin/python pipeline/evaluar_modelo.py <model_id> --casos-json salida.json  # guarda los resultados por caso
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from warehouse_rush import niveles  # noqa: E402
from warehouse_rush.modelos import FILES, TABULAR, load_policy  # noqa: E402
from warehouse_rush.simulacion import evaluate  # noqa: E402

MANIFESTS = os.path.join(ROOT, "manifests")
SHOWN = ("tasa_exito", "otif", "cumplimiento_express", "camiones_perdidos_medios", "rescates_medios",
         "fragiles_rotos_medios", "utilidad_media_cop", "pasos_medios", "latencia_media_ms")


def manifest(model_id):
    path = os.path.join(MANIFESTS, "modelos", f"{model_id}.json")
    if not os.path.exists(path):
        raise SystemExit(f"No hay manifiesto para {model_id} en {path}")
    with open(path) as f:
        return json.load(f)


def download(info, folder):
    """Descarga los pesos y verifica su SHA-256. Devuelve la ruta local."""
    path = os.path.join(folder, FILES[info.get("tipo_modelo", TABULAR)])
    request = urllib.request.Request(info["model_download_url"])
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != info["model_sha256"]:
        raise SystemExit(f"SHA-256 distinto: descargado {digest}, manifiesto {info['model_sha256']}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def run(model_id, level=None, n_cases=None, cases_json=None):
    info = manifest(model_id)
    level = level or info["level_id"]
    cfg = niveles.resolve(level)
    env = niveles.make_env(cfg)
    families = niveles.compatible_families(env)
    if info["familia_modelo"] not in families:
        raise SystemExit(f"{model_id} ({info['familia_modelo']}) no es compatible con {level} ({', '.join(families)})")
    cases, suite_id = niveles.evaluation_suite(cfg)
    print(f"Modelo   {model_id}  ({info.get('tipo_modelo', TABULAR)}, run {info['run_name']}, commit {info['git_commit'][:7]})")
    print(f"Ambiente {cfg['ambiente']} v{cfg['version_ambiente']} · nivel {level} · familia {info['familia_modelo']}")
    print(f"Pesos    {info['model_uri']}")
    if level == info["level_id"]:
        if cfg["version_ambiente"] != info["environment_version"]:
            raise SystemExit(f"Version del ambiente {cfg['version_ambiente']} distinta de la del manifiesto "
                             f"{info['environment_version']}: usar el commit {info['git_commit']}")
        if suite_id != info["evaluation_suite_id"]:
            raise SystemExit(f"La suite reconstruida {suite_id} no es la del manifiesto {info['evaluation_suite_id']}")
        print(f"Suite    {suite_id} (coincide con el manifiesto)")
    else:
        print(f"Suite    {suite_id} (nivel distinto del de entrenamiento: {info['level_id']})")

    with tempfile.TemporaryDirectory() as tmp:
        policy = load_policy(download(info, tmp))
    print("SHA-256  verificado")
    selected = cases[:n_cases] if n_cases else cases
    metrics, per_case = evaluate(env, policy, selected)
    registered = info.get("metricas_evaluacion", {}) if level == info["level_id"] else {}
    print(f"\n{len(selected)} casos de {len(cases)}" + ("" if n_cases else " (suite completa)"))
    print(f"{'metrica':<28}{'ahora':>14}{'registrado':>14}")
    for k in SHOWN:
        if k in metrics:
            ref = registered.get(k)
            fmt = (lambda v: f"{v:,.0f}") if k == "utilidad_media_cop" else (lambda v: f"{v:.4f}")
            print(f"{k:<28}{fmt(metrics[k]):>14}{fmt(ref) if ref is not None else '-':>14}")
    if metrics["fallos_por_causa"]:
        print(f"fallos por causa: {metrics['fallos_por_causa']}")
    if cases_json:
        with open(cases_json, "w") as f:
            json.dump({"model_id": model_id, "nivel": level, "suite_id": suite_id, "metricas": metrics,
                       "casos": per_case}, f, indent=1, ensure_ascii=False)
        print(f"resultados por caso: {cases_json}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", nargs="?")
    parser.add_argument("--nivel", help="nivel donde evaluar (por defecto el de entrenamiento)")
    parser.add_argument("--casos", type=int, help="solo los primeros N casos de la suite")
    parser.add_argument("--casos-json", help="archivo donde guardar los resultados por caso")
    parser.add_argument("--recomendados", action="store_true", help="evalua el recomendado de cada nivel")
    args = parser.parse_args()
    if args.recomendados:
        with open(os.path.join(MANIFESTS, "catalogo.json")) as f:
            for level, model_id in json.load(f)["recomendados"].items():
                print("=" * 70)
                run(model_id, level, args.casos)
        return
    if not args.model_id:
        parser.error("indica un model_id o --recomendados")
    run(args.model_id, args.nivel, args.casos, args.casos_json)


if __name__ == "__main__":
    main()
