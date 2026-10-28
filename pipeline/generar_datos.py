"""Etapa 1: genera el dataset de pedidos de entrenamiento y de evaluacion.

Cada pedido tiene sus paquetes (estanteria, zona de destino, tipo) y la bateria inicial.
Los pedidos de evaluacion no se repiten en entrenamiento.

Salidas: data/pedidos_entrenamiento.json, data/pedidos_evaluacion.json
"""

from common import load_params, write_json
from core import generate_orders
from warehouse_rush.env_v1 import TYPE_NAMES


def main():
    datos = load_params()["datos"]
    training, evaluation = generate_orders(datos)
    meta = {"tipos": TYPE_NAMES, "paquetes_por_pedido": datos["paquetes_por_pedido"]}
    write_json("data/pedidos_entrenamiento.json", dict(meta, pedidos=training))
    write_json("data/pedidos_evaluacion.json", dict(meta, pedidos=evaluation))
    print(f"Pedidos: {len(training)} de entrenamiento, {len(evaluation)} de evaluacion")


if __name__ == "__main__":
    main()
