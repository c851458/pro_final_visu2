#!/usr/bin/env python3
"""Entrena el robot con Q-learning en Warehouse Rush v1 y anima el aprendizaje en 3D (three.js).

Cada episodio es un pedido de varios paquetes (normal, pesado o fragil) con distintas zonas
de entrega, y el robot debe administrar su bateria.

Durante el entrenamiento se graba 1 de cada N episodios y el navegador lo reproduce;
el entrenamiento espera a que termine la animacion (si la pagina esta abierta).
Al final se muestran episodios de demostracion con el agente entrenado (sin exploracion).

Uso:
    ./train3d.py
    ./train3d.py --episodes 40000 --watch-every 2000
    ./train3d.py --paquetes 2 --sin-bateria
    ./train3d.py --demo-only --demo-episodes 5      # usa la tabla Q guardada
    ./train3d.py --demo-only --modelo models/q_table_v1.pkl   # modelo del pipeline (dvc repro)
    ./train3d.py --demo-only --modelo hf:USUARIO/warehouse-rush-qlearning@v1.0   # modelo de Hugging Face
"""

import argparse
import sys
import threading
import time

import gymnasium as gym
import numpy as np

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v1)
from warehouse_rush import QLearningAgent
from warehouse_rush.hub import resolve_model
from warehouse_rush.web3d import WebApp, env_config, snapshot, step_frame, wait_forever

Q_PATH = "q_table_v1.pkl"
EPSILON_MIN = 0.05


class AutoApp(WebApp):
    def __init__(self, env):
        super().__init__(env_config(env, "auto"))
        self.cond = threading.Condition()
        self.latest = None  # ultimo episodio grabado
        self.consumed = -1  # id del ultimo episodio que el navegador termino de animar
        self.next_id = 0
        self.status = {"phase": "esperando", "episode": 0, "episodes": 0, "epsilon": 1.0,
                       "history": [], "demo": None}

    def handle_get(self, path, query):
        if path == "/api/status":
            return self.status
        if path == "/api/next":
            after = int(query.get("after", ["-1"])[0])
            with self.cond:
                self.consumed = max(self.consumed, after)
                self.cond.notify_all()
                episode = self.latest if self.latest and self.latest["id"] > after else None
            return {"episode": episode, "status": self.status}
        return super().handle_get(path, query)

    def publish(self, episode):
        """Entrega un episodio grabado al navegador y espera a que lo anime."""
        with self.cond:
            episode["id"] = self.next_id
            self.next_id += 1
            self.latest = episode
            while self.consumed < episode["id"] and self.viewer_alive():
                self.cond.wait(timeout=0.5)

    def wait_viewer(self, seconds):
        end = time.time() + seconds
        while not self.viewer_alive() and time.time() < end:
            time.sleep(0.2)
        if not self.viewer_alive():
            print("No hay navegador abierto: se entrena sin animacion "
                  "(abre la pagina para ver los siguientes episodios).", flush=True)


def run_episode(env, agent, seed, learn, greedy=False, record=False):
    obs, _ = env.reset(seed=seed)
    frames = [snapshot(env)] if record else None
    done = False
    while not done:
        action = agent.act(obs, greedy=greedy)
        if record:
            next_obs, reward, terminated, truncated, frame = step_frame(env, action)
            frames.append(frame)
        else:
            next_obs, reward, terminated, truncated, _ = env.step(action)
        if learn:
            agent.update(obs, action, reward, next_obs, terminated)
        obs = next_obs
        done = terminated or truncated
    base = env.unwrapped
    result = {"success": base.success, "delivered": base.current, "rescues": base.rescues,
              "steps": base.steps, "total": base.total_reward}
    return result, frames


def train(app, env, agent, episodes, seed, watch_every):
    results = []
    app.status.update(phase="entrenamiento", episodes=episodes)
    for ep in range(episodes):
        watch = watch_every > 0 and (ep % watch_every == 0 or ep == episodes - 1)
        record = watch and app.viewer_alive()
        epsilon = agent.epsilon
        result, frames = run_episode(env, agent, seed + ep, learn=True, record=record)
        agent.decay_epsilon()
        results.append(result)
        app.status.update(episode=ep + 1, epsilon=agent.epsilon)

        if (ep + 1) % 1000 == 0:
            last = results[-1000:]
            entry = {"episode": ep + 1,
                     "success_rate": float(np.mean([r["success"] for r in last])),
                     "mean_reward": float(np.mean([r["total"] for r in last])),
                     "rescues": float(np.mean([r["rescues"] for r in last])),
                     "epsilon": agent.epsilon}
            app.status["history"] = app.status["history"] + [entry]
            print(f"Episodio {ep + 1:6d} | recompensa media {entry['mean_reward']:7.2f} | "
                  f"pedidos completos {entry['success_rate'] * 100:5.1f}% | "
                  f"rescates/ep {entry['rescues']:.2f} | epsilon {agent.epsilon:.3f}", flush=True)
        if record:
            app.publish(dict(result, kind="train", number=ep + 1, of=episodes, epsilon=epsilon, frames=frames))

    agent.save(Q_PATH)
    print(f"Tabla Q guardada en {Q_PATH}", flush=True)


def demo(app, env, agent, seed, episodes):
    app.status.update(phase="demo")
    for i in range(episodes):
        app.status["demo"] = {"number": i + 1, "of": episodes}
        result, frames = run_episode(env, agent, seed + i, learn=False, greedy=True, record=True)
        print(f"Demo {i + 1}: {'Pedido completo' if result['success'] else 'Incompleto'} "
              f"({result['delivered']}/{env.unwrapped.n_packages} paquetes) en {result['steps']} pasos | "
              f"recompensa {result['total']:.1f} | rescates {result['rescues']}", flush=True)
        app.publish(dict(result, kind="demo", number=i + 1, of=episodes, epsilon=0.0, frames=frames))
    app.status.update(phase="terminado")
    print("Listo. La pagina sigue abierta; Ctrl+C para salir.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch-every", type=int, default=2000,
                        help="anima 1 de cada N episodios durante el entrenamiento (0 = no)")
    parser.add_argument("--paquetes", type=int, default=3, help="paquetes por pedido")
    parser.add_argument("--sin-bateria", action="store_true", help="desactiva la bateria")
    parser.add_argument("--demo-only", action="store_true", help="usa la tabla Q guardada, sin entrenar")
    parser.add_argument("--modelo", default=Q_PATH,
                        help="tabla Q a cargar con --demo-only (ruta local o hf:usuario/repo[@revision])")
    parser.add_argument("--demo-episodes", type=int, default=3)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automaticamente")
    args = parser.parse_args()

    env = gym.make("WarehouseRush-v1", n_packages=args.paquetes, use_battery=not args.sin_bateria)
    # epsilon llega a su minimo al 70% del entrenamiento
    decay = EPSILON_MIN ** (1 / max(1, 0.7 * args.episodes))
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n, epsilon_min=EPSILON_MIN,
                           epsilon_decay=decay, seed=args.seed)
    if args.demo_only:
        expected = agent.q.shape
        try:
            agent.load(resolve_model(args.modelo))
        except FileNotFoundError:
            sys.exit(f"No existe {args.modelo}: entrena primero (./train3d.py o dvc repro)")
        if agent.q.shape != expected:
            sys.exit(f"{args.modelo} se entreno con otra configuracion (con/sin bateria); vuelve a entrenar.")

    app = AutoApp(env)
    app.serve(args.port, open_page=not args.no_browser)
    print("Esperando a que se abra la pagina...", flush=True)
    app.wait_viewer(20)

    def run():
        if not args.demo_only:
            train(app, env, agent, args.episodes, args.seed, args.watch_every)
        demo(app, env, agent, args.seed + 12345, args.demo_episodes)

    threading.Thread(target=run, daemon=True).start()
    wait_forever()
