"""Servidor local para la visualizacion 3D (three.js) de Warehouse Rush.

Python ejecuta el entorno y el agente; el navegador solo dibuja la escena.
La pagina (carpeta web/) habla con este servidor mediante una pequena API JSON:

    GET  /api/config         mapa de la bodega y modo ("manual" o "auto")
    GET  /api/state          (manual) estado actual
    POST /api/step           (manual) {"action": 0..5} -> fotograma
    POST /api/reset          (manual) nuevo episodio -> fotograma
    GET  /api/status         (auto) progreso del entrenamiento
    GET  /api/next?after=ID  (auto) siguiente episodio grabado para animar
"""

import json
import os
import platform
import subprocess
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from warehouse_rush.env import LAYOUT, MOVES, PICK, DROP

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def env_config(env, mode):
    base = env.unwrapped
    return {
        "mode": mode,
        "layout": LAYOUT,
        "shelf_cells": [list(p) for p in base.shelf_cells],
        "delivery_cells": sorted(list(map(int, p)) for p in base.delivery_cells),
        "start": list(base.start_pos),
        "max_steps": base.max_steps,
    }


def snapshot(env, action=None, reward=0.0, prev=None, terminated=False, truncated=False):
    """Fotograma serializable del estado del entorno, con el evento ocurrido en el ultimo paso."""
    base = env.unwrapped
    pos = [int(base.agent_pos[0]), int(base.agent_pos[1])]
    if action is None:
        event = "reset"
    elif action in MOVES:
        event = "move" if tuple(pos) != tuple(prev["pos"]) else "bump"
    elif action == PICK:
        event = "pick" if base.carrying and not prev["carrying"] else "pick_fail"
    elif action == DROP:
        event = "drop" if base.delivered else "drop_fail"
    return {
        "pos": pos,
        "carrying": bool(base.carrying),
        "delivered": bool(base.delivered),
        "item_idx": int(base.item_idx),
        "steps": int(base.steps),
        "action": None if action is None else int(action),
        "reward": float(reward),
        "total": float(base.total_reward),
        "event": event,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def step_frame(env, action):
    prev = snapshot(env)
    _, reward, terminated, truncated, _ = env.step(action)
    return snapshot(env, action, reward, prev, terminated, truncated)


class WebApp:
    """Base de los modos manual y automatico: maneja la API y sirve la carpeta web/."""

    def __init__(self, config):
        self.config = config
        self.last_seen = 0.0  # ultima peticion del navegador

    def viewer_alive(self, timeout=3.0):
        return time.time() - self.last_seen < timeout

    def handle_get(self, path, query):
        if path == "/api/config":
            return self.config
        return None

    def handle_post(self, path, body):
        return None

    def serve(self, port, open_page=True):
        app = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=WEB_DIR, **kwargs)

            def log_message(self, *args):
                pass

            def end_headers(self):
                self.send_header("Cache-Control", "no-store")
                super().end_headers()

            def _json(self, data):
                if data is None:
                    self.send_error(404)
                    return
                raw = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path.startswith("/api/"):
                    app.last_seen = time.time()
                    self._json(app.handle_get(url.path, parse_qs(url.query)))
                else:
                    super().do_GET()

            def do_POST(self):
                url = urlparse(self.path)
                app.last_seen = time.time()
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._json(app.handle_post(url.path, body))

        # si el puerto esta ocupado (otro programa), prueba los siguientes
        for candidate in range(port, port + 20):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
                break
            except OSError:
                print(f"El puerto {candidate} esta ocupado, probando el siguiente...", flush=True)
        else:
            raise OSError(f"No hay puertos libres entre {port} y {port + 19}; usa --port")
        url = f"http://localhost:{candidate}/"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Visualizacion 3D en {url}  (Ctrl+C para salir)", flush=True)
        if open_page:
            open_browser(url)
        return server


def open_browser(url):
    """Abre la pagina; en WSL usa el navegador de Windows."""
    if "microsoft" in platform.release().lower():
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", url], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, cwd="/mnt/c")
            return
        except OSError:
            pass
    webbrowser.open(url)


def wait_forever():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCerrando servidor.")
