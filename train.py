"""Entrena el agente Q-learning en Warehouse Rush y muestra un episodio de demostracion.

Uso:
    python train.py --episodes 20000
    python train.py --watch-every 1000          # anima en la terminal 1 de cada 1000 episodios
    python train.py --watch-every 500 --delay 0.1
    python train.py --demo-only                 # usa la tabla Q guardada
    python train.py --gui --watch-every 1000    # igual, pero en una ventana de pygame
    python train.py --gui --demo-only
"""

import argparse
import time

import gymnasium as gym
import numpy as np

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v0)
from warehouse_rush import QLearningAgent

Q_PATH = "q_table.pkl"
ACTION_NAMES = ["arriba", "abajo", "izquierda", "derecha", "recoger", "entregar"]
CLEAR = "\033[H\033[J"

# Colores ANSI para cada simbolo del mapa
STYLE = {
    "#": "\033[90m#\033[0m",
    "S": "\033[33mS\033[0m",
    "D": "\033[42;30mD\033[0m",
    "$": "\033[1;93m$\033[0m",
    "A": "\033[1;94mA\033[0m",
    "C": "\033[1;91mC\033[0m",
    ".": "\033[2m.\033[0m",
}


def colorize(frame):
    return "".join(STYLE.get(ch, ch) for ch in frame)


def show_frame(env, header, step, action, reward, total, info, delay):
    lines = [
        header,
        "",
        colorize(env.render()),
        "",
        f"Paso {step:3d} | accion {ACTION_NAMES[action] if action is not None else '-':9s} | "
        f"recompensa {reward:+6.1f} | acumulada {total:+7.1f}",
        f"Paquete en {info['item_pos']} | {'CARGANDO' if info['carrying'] else 'sin carga'}",
        "",
        "Leyenda: A agente  C agente con paquete  $ paquete  S estanteria  D entregas  # pared",
    ]
    print(CLEAR + "\n".join(lines), flush=True)
    time.sleep(delay)


def run_episode(env, agent, seed, learn, greedy=False, watch=False, header="", delay=0.0):
    obs, info = env.reset(seed=seed)
    total, done, step, terminated = 0.0, False, 0, False
    if watch:
        show_frame(env, header, step, None, 0.0, total, info, delay)
    while not done:
        action = agent.act(obs, greedy=greedy)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if learn:
            agent.update(obs, action, reward, next_obs, terminated)
        obs, total, step = next_obs, total + reward, step + 1
        done = terminated or truncated
        if watch:
            show_frame(env, header, step, action, reward, total, info, delay)
    return total, terminated, step


def make_gui_env(delay):
    return gym.make("WarehouseRush-v0", render_mode="human", render_fps=1 / max(delay, 0.01))


def train(episodes, seed, watch_every, delay, gui):
    env = gym.make("WarehouseRush-v0", render_mode="ansi")
    gui_env = make_gui_env(delay) if gui and watch_every > 0 else None
    agent = QLearningAgent(env.observation_space.nvec, env.action_space.n, seed=seed)
    returns, delivered, history = [], [], []

    for ep in range(episodes):
        watch = watch_every > 0 and (ep % watch_every == 0 or ep == episodes - 1)
        header = f"ENTRENAMIENTO | episodio {ep + 1}/{episodes} | epsilon {agent.epsilon:.3f}"
        if watch and gui_env is not None and not gui_env.unwrapped.closed_by_user:
            # la ventana se dibuja sola en cada paso (render_mode="human")
            gui_env.unwrapped.hud_extra = [header] + history[-2:]
            total, ok, steps = run_episode(gui_env, agent, seed + ep, learn=True)
            gui_env.unwrapped.hold(seconds=max(delay * 8, 0.6))
        else:
            header += "\n" + "\n".join(history[-5:])
            total, ok, steps = run_episode(env, agent, seed + ep, learn=True,
                                           watch=watch and not gui, header=header, delay=delay)
        agent.decay_epsilon()
        returns.append(total)
        delivered.append(ok)
        if watch and not gui:
            print(f"\n=> {'ENTREGADO' if ok else 'NO ENTREGADO'} en {steps} pasos", flush=True)
            time.sleep(max(delay * 8, 0.5))

        if (ep + 1) % 1000 == 0:
            line = (f"Episodio {ep + 1:6d} | recompensa media {np.mean(returns[-1000:]):7.2f} | "
                    f"entregas {np.mean(delivered[-1000:]) * 100:5.1f}% | epsilon {agent.epsilon:.3f}")
            history.append(line)
            print(line, flush=True)

    agent.save(Q_PATH)
    print(f"Tabla Q guardada en {Q_PATH}")
    if gui_env is not None:
        gui_env.close()
    return agent


def demo(agent, seed, delay, gui, episodes=1):
    header = "DEMO | agente entrenado (sin exploracion)"
    env = make_gui_env(delay) if gui else gym.make("WarehouseRush-v0", render_mode="ansi")
    for i in range(episodes):
        if gui:
            if env.unwrapped.closed_by_user:
                break
            env.unwrapped.hud_extra = [f"{header} | episodio {i + 1}/{episodes}"]
            total, ok, steps = run_episode(env, agent, seed + i, learn=False, greedy=True)
            env.unwrapped.hold(seconds=1.0 if i < episodes - 1 else None)
        else:
            total, ok, steps = run_episode(env, agent, seed + i, learn=False, greedy=True,
                                           watch=True, header=header, delay=delay)
        print(f"{'Entregado' if ok else 'No entregado'} en {steps} pasos | recompensa {total:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.15, help="segundos entre fotogramas")
    parser.add_argument("--watch-every", type=int, default=0,
                        help="anima 1 de cada N episodios durante el entrenamiento (0 = no)")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--gui", action="store_true", help="visualizar con pygame en vez de la terminal")
    parser.add_argument("--demo-episodes", type=int, default=1)
    args = parser.parse_args()

    if args.demo_only:
        env = gym.make("WarehouseRush-v0")
        agent = QLearningAgent(env.observation_space.nvec, env.action_space.n)
        agent.load(Q_PATH)
    else:
        agent = train(args.episodes, args.seed, args.watch_every, args.delay, args.gui)
    demo(agent, seed=args.seed + 12345, delay=args.delay, gui=args.gui, episodes=args.demo_episodes)
