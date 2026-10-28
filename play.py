"""Control manual del agente en Warehouse Rush con el teclado.

Controles:
    Flechas   mover el agente
    R         recoger el paquete (estando junto a la estanteria que lo tiene)
    E         entregar el paquete (estando en la zona de entregas)
    N         nuevo episodio
    ESC       salir

Uso:
    python play.py
    python play.py --seed 7 --max-steps 300
"""

import argparse

import gymnasium as gym
import pygame

import warehouse_rush  # noqa: F401  (registra WarehouseRush-v0)
from warehouse_rush.env import UP, DOWN, LEFT, RIGHT, PICK, DROP
from warehouse_rush.renderer import ACTION_NAMES

KEY_TO_ACTION = {
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_r: PICK,
    pygame.K_e: DROP,
}
CONTROLS = "Flechas mover  |  R recoger  |  E entregar  |  N nuevo episodio"


def main(seed, max_steps):
    env = gym.make("WarehouseRush-v0", render_mode="human", render_fps=30, max_steps=max_steps)
    base = env.unwrapped
    episode = 1
    env.reset(seed=seed)
    base.renderer.allow_pause = False
    base.renderer.hint = "ESC salir"
    base.hud_extra = [f"MODO MANUAL | episodio {episode}", CONTROLS]
    done = False

    while not base.closed_by_user:
        env.render()  # dibuja la ventana y recoge las teclas pulsadas
        if base.closed_by_user:
            break
        for key in base.renderer.pop_keys():
            if key == pygame.K_n:
                episode += 1
                env.reset(seed=seed + episode - 1)
                base.hud_extra = [f"MODO MANUAL | episodio {episode}", CONTROLS]
                done = False
                print(f"--- Episodio {episode} ---")
            elif key in KEY_TO_ACTION and not done:
                action = KEY_TO_ACTION[key]
                _, reward, terminated, truncated, info = env.step(action)
                print(f"Paso {info['steps']:3d} | {ACTION_NAMES[action]:9s} | recompensa {reward:+5.1f} | "
                      f"acumulada {base.total_reward:+6.1f}")
                if terminated or truncated:
                    done = True
                    result = "ENTREGADO" if terminated else "TIEMPO AGOTADO"
                    print(f"=> {result} en {info['steps']} pasos | recompensa total {base.total_reward:+.1f}")
                    base.hud_extra = [f"MODO MANUAL | episodio {episode} terminado: {result}",
                                      "Pulsa N para un nuevo episodio o ESC para salir"]
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    main(args.seed, args.max_steps)
