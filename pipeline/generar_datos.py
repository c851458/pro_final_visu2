"""Etapa 1: genera el dataset de pedidos de un nivel (entrenamiento y evaluacion).

Cada pedido tiene sus paquetes (estanteria, zona de destino, tipo) y la bateria inicial. Los pedidos de
evaluacion no se repiten en entrenamiento y solo usan estanterias alcanzables (el nivel N3 tiene bloqueos).

Uso: python pipeline/generar_datos.py n2
Salidas: data/<nivel>/pedidos_entrenamiento.json, data/<nivel>/pedidos_evaluacion.json
"""

import sys

from common import data_path, write_json
from warehouse_rush import niveles
from warehouse_rush.env_v1 import TYPE_NAMES


def main(level):
    cfg = niveles.resolve(level)
    env = niveles.make_env(cfg)
    training, evaluation = niveles.generate_orders(cfg["datos"], env.unwrapped.reachable_shelves)
    meta = {"tipos": TYPE_NAMES, "paquetes_por_pedido": cfg["datos"]["paquetes_por_pedido"]}
    write_json(data_path(level, "entrenamiento"), dict(meta, pedidos=training))
    write_json(data_path(level, "evaluacion"), dict(meta, pedidos=evaluation))
    print(f"{level}: {len(training)} pedidos de entrenamiento, {len(evaluation)} de evaluacion")


if __name__ == "__main__":
    main(sys.argv[1])
