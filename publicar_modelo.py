#!/usr/bin/env python3
"""Publica un modelo en Hugging Face Hub, con su ficha (model card), y lo enlaza a su run de MLflow.

Cada publicacion es un commit en el repositorio de Hugging Face (historial de versiones);
con --tag ademas se marca la version (por ejemplo v1.0).

Uso:
    ./publicar_modelo.py                          # modelo del ultimo `dvc repro` (run "pipeline")
    ./publicar_modelo.py --run alpha-alto         # modelo de un experimento (nombre o id del run)
    ./publicar_modelo.py --tag v1.0 --publico     # con tag y repositorio publico (por defecto: privado)

Cargar un modelo publicado:
    ./train3d.py --demo-only --modelo hf:USUARIO/warehouse-rush-qlearning
    ./train3d.py --demo-only --modelo hf:USUARIO/warehouse-rush-qlearning@v1.0
"""

import argparse
import json
import os
import re
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from huggingface_hub import HfApi  # noqa: E402

import tracking  # noqa: E402  (antes que mlflow: configura su entorno)
import mlflow  # noqa: E402
from warehouse_rush import QLearningAgent  # noqa: E402
from warehouse_rush.env_v1 import ACTION_NAMES  # noqa: E402
from warehouse_rush.hub import MODEL_FILE  # noqa: E402

REPO_NAME = "warehouse-rush-qlearning"

METRIC_LABELS = {
    "tasa_exito": "Pedidos completos",
    "paquetes_entregados_medios": "Paquetes entregados por pedido",
    "pasos_medios": "Pasos por pedido",
    "recompensa_media": "Recompensa media",
    "rescates_medios": "Rescates por batería agotada",
    "fragiles_rotos_medios": "Paquetes frágiles rotos",
    "cargas_medias": "Cargas de batería",
    "choques_medios": "Choques",
}


def find_run(spec):
    """Busca el run por id o por nombre (el mas reciente con ese nombre)."""
    if re.fullmatch(r"[0-9a-f]{32}", spec):
        return mlflow.get_run(spec)
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], filter_string=f"tags.mlflow.runName = '{spec}'",
                              order_by=["start_time DESC"], max_results=1, output_format="list")
    if not runs:
        raise SystemExit(f"No hay ningun run de MLflow llamado '{spec}'")
    return runs[0]


def model_card(repo_id, run, params, metrics, shape):
    p = params
    rows = "\n".join(f"| {METRIC_LABELS[k]} | {metrics[k] * 100:.1f} % |" if k == "tasa_exito"
                     else f"| {METRIC_LABELS[k]} | {metrics[k]:.3f} |" for k in METRIC_LABELS)
    return f"""---
tags:
- reinforcement-learning
- q-learning
- gymnasium
- warehouse-rush
model-index:
- name: {repo_id.split('/')[1]}
  results:
  - task:
      type: reinforcement-learning
      name: Reinforcement Learning
    dataset:
      type: WarehouseRush-v1
      name: WarehouseRush-v1 ({metrics['pedidos_evaluados']} pedidos de evaluación)
    metrics:
    - type: tasa_exito
      value: {metrics['tasa_exito']:.4f}
      name: Pedidos completos
---

# Warehouse Rush · Q-learning

Agente de **Q-learning tabular** que controla un robot de bodega en el entorno `WarehouseRush-v1` (Gymnasium).
Cada episodio es un pedido de {p['datos']['paquetes_por_pedido']} paquetes que el robot debe recoger de las estanterías
y entregar, en orden, en la zona de su color. Hay paquetes normales, pesados (gastan el doble de batería)
y frágiles (se rompen si el robot choca), y el robot debe recargar su batería en la estación de carga.

## Evaluación

Sin exploración, sobre {metrics['pedidos_evaluados']} pedidos que no se usaron en el entrenamiento:

| Métrica | Valor |
|---|---|
{rows}

## Entrenamiento

| Parámetro | Valor |
|---|---|
| Episodios | {p['entrenamiento']['episodios']} |
| Tasa de aprendizaje (α) | {p['agente']['alpha']} |
| Factor de descuento (γ) | {p['agente']['gamma']} |
| ε mínimo | {p['agente']['epsilon_min']} (alcanzado al {p['agente']['fraccion_exploracion'] * 100:.0f} % del entrenamiento) |
| Batería máxima | {p['entorno']['bateria_max'] if p['entorno']['bateria'] else 'sin batería'} |
| Pasos máximos por pedido | {p['entorno']['max_pasos']} |

Run de MLflow: `{run.info.run_id}` ({run.info.run_name}) ·
commit de git: `{run.data.tags.get('git.commit', '')[:7]}`.

## Formato

`{MODEL_FILE}`: tabla Q (`float32`, forma `{list(shape)}`), indexada por la observación
`[fila, columna, nivel de batería, lleva paquete, tipo del paquete que lleva, objetivo]` y la acción
({', '.join(f'{i} {a}' for i, a in enumerate(ACTION_NAMES))}).

```python
import numpy as np
from huggingface_hub import hf_hub_download

q = np.load(hf_hub_download("{repo_id}", "{MODEL_FILE}"))
accion = int(np.argmax(q[tuple(observacion)]))
```

Para verlo en 3D desde el proyecto: `./train3d.py --demo-only --modelo hf:{repo_id}`
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="nombre o id del run de MLflow (por defecto, el del pipeline)")
    parser.add_argument("--repo", help=f"repositorio de Hugging Face (por defecto USUARIO/{REPO_NAME})")
    parser.add_argument("--tag", help="etiqueta de version, por ejemplo v1.0")
    parser.add_argument("--publico", action="store_true", help="crear el repositorio como publico")
    parser.add_argument("--mensaje", help="mensaje del commit en Hugging Face")
    args = parser.parse_args()
    os.chdir(ROOT)

    tracking.setup()
    if args.run:
        run = find_run(args.run)
    else:
        with open("metrics/mlflow_run.json") as f:
            run = mlflow.get_run(json.load(f)["run_id"])
    if "eval.tasa_exito" not in run.data.metrics:
        raise SystemExit(f"El run {run.info.run_id} no tiene evaluacion; corre el pipeline completo (dvc repro)")

    api = HfApi()
    repo_id = args.repo or f"{api.whoami()['name']}/{REPO_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        src = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=os.path.join(tmp, "mlflow"))
        with open(os.path.join(src, "params.yaml")) as f:
            params = yaml.safe_load(f)
        with open(os.path.join(src, "evaluacion.json")) as f:
            metrics = json.load(f)
        agent = QLearningAgent([1], 1)
        agent.load(os.path.join(src, "modelo", "q_table_v1.pkl"))

        out = os.path.join(tmp, "hub")
        os.makedirs(out)
        np.save(os.path.join(out, MODEL_FILE), agent.q, allow_pickle=False)
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(model_card(repo_id, run, params, metrics, agent.q.shape))
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"entorno": "WarehouseRush-v1", "acciones": ACTION_NAMES, "forma_tabla_q": list(agent.q.shape),
                       "params": params, "mlflow_run_id": run.info.run_id}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out, "evaluacion.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        api.create_repo(repo_id, private=not args.publico, exist_ok=True)
        message = args.mensaje or (f"Modelo del run {run.info.run_name} ({run.info.run_id[:8]}): "
                                   f"{metrics['tasa_exito'] * 100:.1f}% de pedidos completos")
        commit = api.upload_folder(repo_id=repo_id, folder_path=out, commit_message=message)
    if args.tag:
        api.create_tag(repo_id, tag=args.tag, revision=commit.oid, tag_message=message)

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tags({"hf.repo": repo_id, "hf.commit": commit.oid, "hf.url": commit.commit_url,
                         **({"hf.tag": args.tag} if args.tag else {})})
    print(f"Publicado en https://huggingface.co/{repo_id} (commit {commit.oid[:8]}"
          f"{', tag ' + args.tag if args.tag else ''})")
    print(f"Cargar: ./train3d.py --demo-only --modelo hf:{repo_id}{'@' + args.tag if args.tag else ''}")


if __name__ == "__main__":
    main()
