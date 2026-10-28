#!/usr/bin/env python3
"""Control manual del robot en Warehouse Rush con visualizacion 3D (three.js en el navegador).

Controles en la pagina:
    Flechas / WASD   mover el robot (relativo a la camara)
    R / Espacio      recoger el paquete (junto a la estanteria que lo tiene)
    E / Enter        entregar el paquete (en la zona de entregas)
    N                nuevo episodio
    Raton            arrastrar = girar la camara, rueda = zoom, clic derecho = desplazar

Uso:
    ./play3d.py
    ./play3d.py --seed 7 --max-steps 300 --port 8770
"""

import argparse
import threading

import gymnasium as gym

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v0)
from warehouse_rush.renderer import ACTION_NAMES
from warehouse_rush.web3d import WebApp, env_config, snapshot, step_frame, wait_forever


class ManualApp(WebApp):
    def __init__(self, seed, max_steps):
        self.env = gym.make("WarehouseRush-v0", max_steps=max_steps)
        super().__init__(env_config(self.env, "manual"))
        self.seed = seed
        self.episode = 0
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.episode += 1
        self.env.reset(seed=self.seed + self.episode - 1)
        self.frame = dict(snapshot(self.env), episode=self.episode)
        print(f"--- Episodio {self.episode} ---", flush=True)
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
                self.frame = dict(step_frame(self.env, action), episode=self.episode)
                f = self.frame
                print(f"Paso {f['steps']:3d} | {ACTION_NAMES[action]:9s} | recompensa {f['reward']:+5.1f} | "
                      f"acumulada {f['total']:+6.1f}", flush=True)
                if f["terminated"] or f["truncated"]:
                    result = "ENTREGADO" if f["terminated"] else "TIEMPO AGOTADO"
                    print(f"=> {result} en {f['steps']} pasos | recompensa total {f['total']:+.1f}", flush=True)
                return self.frame
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automaticamente")
    args = parser.parse_args()

    ManualApp(args.seed, args.max_steps).serve(args.port, open_page=not args.no_browser)
    wait_forever()
