"""Entorno Warehouse Rush: una bodega de carga en cuadricula.

Leyenda del mapa:
    #  pared (limite de la bodega)
    .  piso libre (pasillos y corredores)
    S  estanteria (no transitable)
    D  zona de entregas

La bodega tiene 4 estanterias separadas por 3 pasillos, un corredor
perimetral y una zona de entregas en la parte inferior.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

LAYOUT = [
    "###############",
    "#.............#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.............#",
    "#.............#",
    "#.....DDD.....#",
    "###############",
]

WALL, FLOOR, SHELF, DELIVERY = "#", ".", "S", "D"

# Acciones
UP, DOWN, LEFT, RIGHT, PICK, DROP = range(6)
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Recompensas
STEP_PENALTY = -0.1
COLLISION_PENALTY = -1.0
INVALID_ACTION_PENALTY = -1.0
PICK_REWARD = 10.0
DELIVERY_REWARD = 20.0


class WarehouseRushEnv(gym.Env):
    """El agente debe recoger un paquete de una estanteria y llevarlo a la zona de entregas.

    Observacion: MultiDiscrete [fila, columna, cargando (0/1), indice del paquete].
    Acciones: 0 arriba, 1 abajo, 2 izquierda, 3 derecha, 4 recoger, 5 entregar.
    El paquete se recoge estando en una casilla adyacente a la estanteria que lo contiene.
    El episodio termina al entregar el paquete en la zona D.

    Modos de render: "human" (ventana pygame), "rgb_array" (imagen pygame) y "ansi" (texto).
    `hud_extra` es una lista de lineas que se muestran en el panel de la ventana.
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode=None, max_steps=200, random_start=False, cell_size=48, render_fps=None):
        self.grid = np.array([list(row) for row in LAYOUT])
        self.height, self.width = self.grid.shape
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.random_start = random_start
        self.cell_size = cell_size
        self.render_fps = render_fps or self.metadata["render_fps"]
        self.renderer = None
        self.hud_extra = []

        self.shelf_cells = [tuple(int(x) for x in p) for p in np.argwhere(self.grid == SHELF)]
        self.free_cells = [tuple(int(x) for x in p) for p in np.argwhere(self.grid == FLOOR)]
        self.delivery_cells = {tuple(p) for p in np.argwhere(self.grid == DELIVERY)}
        self.start_pos = (8, 7)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.MultiDiscrete(
            [self.height, self.width, 2, len(self.shelf_cells)]
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            self.agent_pos = self.free_cells[self.np_random.integers(len(self.free_cells))]
        else:
            self.agent_pos = self.start_pos
        self.item_idx = int(self.np_random.integers(len(self.shelf_cells)))
        self.carrying = False
        self.delivered = False
        self.steps = 0
        self.last_action, self.last_reward, self.total_reward = None, 0.0, 0.0
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        self.steps += 1
        reward = STEP_PENALTY
        terminated = False

        if action in MOVES:
            dr, dc = MOVES[action]
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self.grid[nxt] in (WALL, SHELF):
                reward += COLLISION_PENALTY
            else:
                self.agent_pos = nxt
        elif action == PICK:
            if not self.carrying and self._adjacent(self.agent_pos, self.shelf_cells[self.item_idx]):
                self.carrying = True
                reward += PICK_REWARD
            else:
                reward += INVALID_ACTION_PENALTY
        elif action == DROP:
            if self.carrying and self.agent_pos in self.delivery_cells:
                self.carrying = False
                self.delivered = True
                reward += DELIVERY_REWARD
                terminated = True
            else:
                reward += INVALID_ACTION_PENALTY

        truncated = not terminated and self.steps >= self.max_steps
        self.last_action, self.last_reward = int(action), reward
        self.total_reward += reward
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info()

    def _obs(self):
        return np.array(
            [self.agent_pos[0], self.agent_pos[1], int(self.carrying), self.item_idx],
            dtype=np.int64,
        )

    def _info(self):
        return {"item_pos": self.shelf_cells[self.item_idx], "carrying": self.carrying, "steps": self.steps}

    @staticmethod
    def _adjacent(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @property
    def closed_by_user(self):
        return self.renderer is not None and self.renderer.closed_by_user

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode in ("human", "rgb_array"):
            if self.closed_by_user:
                return None
            if self.renderer is None:
                from warehouse_rush.renderer import PygameRenderer
                self.renderer = PygameRenderer(self, self.cell_size, self.render_fps,
                                               human=self.render_mode == "human")
            if self.render_mode == "human":
                self.renderer.show()
                return None
            return self.renderer.rgb_array()
        return None

    def hold(self, seconds=None):
        """Deja la ventana abierta mostrando el ultimo fotograma (None = hasta que el usuario la cierre)."""
        if self.render_mode == "human" and self.renderer is not None and not self.closed_by_user:
            self.renderer.hold(seconds)

    def close(self):
        if self.renderer is not None and not self.renderer.closed_by_user:
            self.renderer.close()
        self.renderer = None

    def _render_text(self):
        canvas = self.grid.copy()
        if not self.carrying and not self.delivered:
            canvas[self.shelf_cells[self.item_idx]] = "$"
        canvas[self.agent_pos] = "A" if not self.carrying else "C"
        return "\n".join("".join(row) for row in canvas)
