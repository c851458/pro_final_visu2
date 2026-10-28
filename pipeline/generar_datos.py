"""Etapa 1: genera el dataset de pedidos de entrenamiento y de evaluacion.

Cada pedido tiene sus paquetes (estanteria, zona de destino, tipo) y la bateria inicial.
Los pedidos de evaluacion no se repiten en entrenamiento.

Salidas: data/pedidos_entrenamiento.json, data/pedidos_evaluacion.json
"""

import numpy as np

from common import load_params, write_json
from warehouse_rush.env_v1 import SHELF, LAYOUT, TYPE_NAMES, random_order


def order_key(order):
    return tuple((p["shelf"], p["zone"], p["type"]) for p in order["paquetes"]) + (order["bateria_inicial"],)


def main():
    params = load_params()["datos"]
    rng = np.random.default_rng(params["semilla"])
    n_shelves = sum(row.count(SHELF) for row in LAYOUT)
    probs = [params["prob_tipos"][name] for name in ("normal", "pesado", "fragil")]
    assert abs(sum(probs) - 1) < 1e-9, "prob_tipos debe sumar 1"

    def generate(n, exclude=frozenset()):
        orders, seen = [], set(exclude)
        while len(orders) < n:
            order = random_order(rng, params["paquetes_por_pedido"], n_shelves, probs)
            if order_key(order) not in seen:
                seen.add(order_key(order))
                orders.append(order)
        return orders

    evaluation = generate(params["pedidos_evaluacion"])
    training = generate(params["pedidos_entrenamiento"], exclude={order_key(o) for o in evaluation})
    meta = {"tipos": TYPE_NAMES, "paquetes_por_pedido": params["paquetes_por_pedido"]}
    write_json("data/pedidos_entrenamiento.json", dict(meta, pedidos=training))
    write_json("data/pedidos_evaluacion.json", dict(meta, pedidos=evaluation))
    print(f"Pedidos: {len(training)} de entrenamiento, {len(evaluation)} de evaluacion")


if __name__ == "__main__":
    main()
