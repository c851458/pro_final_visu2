#!/usr/bin/env python3
"""Control manual del robot en Warehouse Rush v1 con visualizacion 3D (three.js en el navegador).

Cada episodio es un pedido de varios paquetes (normal, pesado o fragil) que van a distintas
zonas de entrega. La bateria baja al moverse y se recarga en la estacion de carga.

Controles en la pagina:
    Flechas / WASD   mover el robot (relativo a la camara)
    R / Espacio      recoger el paquete (junto a la estanteria que lo tiene)
    E / Enter        entregar el paquete (en su zona de entrega)
    C                cargar la bateria (en la estacion de carga)
    N                nuevo pedido
    Raton            arrastrar = girar la camara, rueda = zoom, clic derecho = desplazar

Uso:
    ./play3d.py
    ./play3d.py --paquetes 2 --sin-bateria --seed 7 --port 8770
"""

import argparse
import threading

import gymnasium as gym

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v1)
from warehouse_rush.env_v1 import ACTION_NAMES
from warehouse_rush.web3d import WebApp, env_config, snapshot, step_frame, wait_forever


class ManualApp(WebApp):
    def __init__(self, seed, max_steps, n_packages, use_battery):
        self.env = gym.make("WarehouseRush-v1", max_steps=max_steps, n_packages=n_packages,
                            use_battery=use_battery)
        super().__init__(env_config(self.env, "manual"))
        self.seed = seed
        self.episode = 0
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.episode += 1
        self.env.reset(seed=self.seed + self.episode - 1)
        self.frame = dict(snapshot(self.env), episode=self.episode)
        print(f"--- Pedido {self.episode} ---", flush=True)
        return self.frame

    def handle_get(self, path, query):
        if path == "/api/state":
            return self.frame
        return super().handle_get(path, query)

    def handle_post(self, path, body):
        with self.lock:
            if path == "/api/reset":
                return self.reset()
            if path == "/api/step":
                if self.frame["terminated"] or self.frame["truncated"]:
                    return self.frame
                action = int(body["action"])
                self.frame = dict(step_frame(self.env, action)[4], episode=self.episode)
                f = self.frame
                print(f"Paso {f['steps']:3d} | {ACTION_NAMES[action]:9s} | {f['event']:11s} | "
                      f"recompensa {f['reward']:+6.1f} | acumulada {f['total']:+7.1f} | "
                      f"bateria {f['battery']:3d}", flush=True)
                if f["terminated"] or f["truncated"]:
                    result = "PEDIDO COMPLETO" if f["success"] else "TIEMPO AGOTADO"
                    print(f"=> {result} en {f['steps']} pasos | recompensa total {f['total']:+.1f} | "
                          f"rescates {f['rescues']}", flush=True)
                return self.frame
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--paquetes", type=int, default=3, help="paquetes por pedido")
    parser.add_argument("--sin-bateria", action="store_true", help="desactiva la bateria")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automaticamente")
    args = parser.parse_args()

    app = ManualApp(args.seed, args.max_steps, args.paquetes, not args.sin_bateria)
    app.serve(args.port, open_page=not args.no_browser)
    wait_forever()
