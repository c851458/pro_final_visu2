#!/usr/bin/env python3
"""Entrena el robot con Q-learning y anima el aprendizaje en 3D (three.js en el navegador).

Durante el entrenamiento se graba 1 de cada N episodios y el navegador lo reproduce;
el entrenamiento espera a que termine la animacion (si la pagina esta abierta).
Al final se muestran episodios de demostracion con el agente entrenado (sin exploracion).

Uso:
    ./train3d.py
    ./train3d.py --episodes 20000 --watch-every 1000
    ./train3d.py --demo-only --demo-episodes 5      # usa la tabla Q guardada
"""

import argparse
import threading
import time

import gymnasium as gym
import numpy as np

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v0)
from warehouse_rush import QLearningAgent
from warehouse_rush.web3d import WebApp, env_config, snapshot, step_frame, wait_forever

Q_PATH = "q_table.pkl"


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
    total, done, terminated = 0.0, False, False
    while not done:
        action = agent.act(obs, greedy=greedy)
        if record:
            frame = step_frame(env, action)
            next_obs, reward = env.unwrapped._obs(), frame["reward"]
            terminated, truncated = frame["terminated"], frame["truncated"]
            frames.append(frame)
        else:
            next_obs, reward, terminated, truncated, _ = env.step(action)
        if learn:
            agent.update(obs, action, reward, next_obs, terminated)
        obs, total = next_obs, total + reward
        done = terminated or truncated
    return total, terminated, env.unwrapped.steps, frames


def train(app, env, agent, episodes, seed, watch_every):
    returns, delivered = [], []
    app.status.update(phase="entrenamiento", episodes=episodes)
    for ep in range(episodes):
        watch = watch_every > 0 and (ep % watch_every == 0 or ep == episodes - 1)
        record = watch and app.viewer_alive()
        epsilon = agent.epsilon
        total, ok, steps, frames = run_episode(env, agent, seed + ep, learn=True, record=record)
        agent.decay_epsilon()
        returns.append(total)
        delivered.append(ok)
        app.status.update(episode=ep + 1, epsilon=agent.epsilon)

        if (ep + 1) % 1000 == 0:
            entry = {"episode": ep + 1, "mean_reward": float(np.mean(returns[-1000:])),
                     "delivery_rate": float(np.mean(delivered[-1000:])), "epsilon": agent.epsilon}
            app.status["history"] = app.status["history"] + [entry]
            print(f"Episodio {ep + 1:6d} | recompensa media {entry['mean_reward']:7.2f} | "
                  f"entregas {entry['delivery_rate'] * 100:5.1f}% | epsilon {agent.epsilon:.3f}", flush=True)
        if record:
            app.publish({"kind": "train", "number": ep + 1, "of": episodes, "epsilon": epsilon,
                         "frames": frames, "delivered": ok, "steps": steps, "total": total})

    agent.save(Q_PATH)
    print(f"Tabla Q guardada en {Q_PATH}", flush=True)


def demo(app, env, agent, seed, episodes):
    app.status.update(phase="demo")
    for i in range(episodes):
        app.status["demo"] = {"number": i + 1, "of": episodes}
        total, ok, steps, frames = run_episode(env, agent, seed + i, learn=False, greedy=True, record=True)
        print(f"Demo {i + 1}: {'Entregado' if ok else 'No entregado'} en {steps} pasos | "
              f"recompensa {total:.1f}", flush=True)
        app.publish({"kind": "demo", "number": i + 1, "of": episodes, "epsilon": 0.0,
                     "frames": frames, "delivered": ok, "steps": steps, "total": total})
    app.status.update(phase="terminado")
    print("Listo. La pagina sigue abierta; Ctrl+C para salir.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch-every", type=int, default=1000,
                        help="anima 1 de cada N episodios durante el entrenamiento (0 = no)")
    parser.add_argument("--demo-only", action="store_true", help="usa la tabla Q guardada, sin entrenar")
    parser.add_argument("--demo-episodes", type=int, default=3)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador automaticamente")
    args = parser.parse_args()

    env = gym.make("WarehouseRush-v0")
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n, seed=args.seed)
    app = AutoApp(env)
    app.serve(args.port, open_page=not args.no_browser)
    print("Esperando a que se abra la pagina...", flush=True)
    app.wait_viewer(20)

    def run():
        if args.demo_only:
            agent.load(Q_PATH)
        else:
            train(app, env, agent, args.episodes, args.seed, args.watch_every)
        demo(app, env, agent, args.seed + 12345, args.demo_episodes)

    threading.Thread(target=run, daemon=True).start()
    wait_forever()
