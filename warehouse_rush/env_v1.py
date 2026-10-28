"""Warehouse Rush v1: pedidos de varios paquetes, bateria, zonas de entrega y tipos de paquete.

Leyenda del mapa:
    #      pared (limite de la bodega)
    .      piso libre
    S      estanteria (no transitable)
    B      estacion de carga de bateria
    1 2 3  zonas de entrega A, B y C

Cada episodio es un pedido de N paquetes que el robot entrega en orden, uno a la vez.
Cada paquete esta en una estanteria, tiene una zona de destino y un tipo:
    normal   sin efectos extra
    pesado   mientras se lleva, moverse gasta el doble de bateria y cuesta mas
    fragil   si el robot choca mientras lo lleva, se rompe y hay que volver a buscarlo

La bateria baja con cada movimiento; si llega a 0, un operario rescata al robot y lo lleva a la
estacion de carga (con una penalizacion fuerte), conservando el paquete que llevaba.
En la estacion de carga, la accion "cargar" la llena.
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

LAYOUT = [
    "###############",
    "#......B......#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.SS.SS.SS.SS.#",
    "#.............#",
    "#.............#",
    "#111..222..333#",
    "###############",
]

WALL, FLOOR, SHELF, CHARGER = "#", ".", "S", "B"
ZONE_CHARS = "123"
ZONE_NAMES = ["A", "B", "C"]

# Tipos de paquete
NORMAL, HEAVY, FRAGILE = range(3)
TYPE_NAMES = ["normal", "pesado", "fragil"]
TYPE_PROBS = [0.5, 0.25, 0.25]

# Acciones
UP, DOWN, LEFT, RIGHT, PICK, DROP, CHARGE = range(7)
ACTION_NAMES = ["arriba", "abajo", "izquierda", "derecha", "recoger", "entregar", "cargar"]
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Bateria
BATTERY_MAX = 100
BATTERY_LEVELS = 5  # niveles que ve el agente: 0 vacia, 1-4 cuartos de bateria
MOVE_COST = 1
HEAVY_MOVE_COST = 2

# Recompensas
STEP_PENALTY = -0.1
HEAVY_STEP_PENALTY = -0.2
COLLISION_PENALTY = -1.0
BREAK_PENALTY = -10.0
INVALID_ACTION_PENALTY = -1.0
PICK_REWARD = 5.0
DELIVERY_REWARD = 15.0
ORDER_BONUS = 20.0
RESCUE_PENALTY = -25.0


def random_order(rng, n_packages, n_shelves, type_probs=TYPE_PROBS):
    """Genera un pedido al azar: paquetes en estanterias distintas y bateria inicial (fraccion)."""
    shelves = rng.choice(n_shelves, n_packages, replace=False)
    return {
        "paquetes": [
            {"shelf": int(s), "zone": int(rng.integers(len(ZONE_NAMES))),
             "type": int(rng.choice(len(TYPE_NAMES), p=type_probs))}
            for s in shelves
        ],
        "bateria_inicial": int(rng.integers(50, 101)) / 100,
    }


class WarehouseRushV1Env(gym.Env):
    """Observacion: MultiDiscrete [fila, columna, nivel de bateria, cargando (0/1),
    tipo del paquete que lleva, objetivo].

    `objetivo` es la estanteria del paquete actual si no lo lleva, o su zona de destino si lo lleva.
    El tipo solo se informa mientras lleva el paquete (antes vale 0): asi la tabla Q se mantiene
    pequena y el agente no tiene que aprender a navegar por separado para cada tipo.
    Acciones: 0 arriba, 1 abajo, 2 izquierda, 3 derecha, 4 recoger, 5 entregar, 6 cargar bateria.
    Modo de render: "ansi" (texto). La visualizacion 3D esta en web3d.py.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    def __init__(self, render_mode=None, n_packages=3, use_battery=True, max_steps=400, battery_max=BATTERY_MAX):
        self.grid = np.array([list(row) for row in LAYOUT])
        self.height, self.width = self.grid.shape
        self.render_mode = render_mode
        self.n_packages = n_packages
        self.use_battery = use_battery
        self.max_steps = max_steps
        self.battery_max = battery_max

        cells = lambda ch: [tuple(int(x) for x in p) for p in np.argwhere(self.grid == ch)]
        self.shelf_cells = cells(SHELF)
        self.zone_cells = [set(cells(z)) for z in ZONE_CHARS]
        self.charger = cells(CHARGER)[0]
        self.start_pos = (8, 7)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.MultiDiscrete([
            self.height, self.width, BATTERY_LEVELS if use_battery else 1, 2,
            len(TYPE_NAMES), len(self.shelf_cells),
        ])

    def reset(self, *, seed=None, options=None):
        """`options={"pedido": {...}}` usa un pedido fijo (del dataset); si no, se genera uno al azar."""
        super().reset(seed=seed)
        order = (options or {}).get("pedido") or random_order(self.np_random, self.n_packages, len(self.shelf_cells))
        self.order = [dict(p, status="pendiente", broken=0) for p in order["paquetes"]]
        self.n_packages = len(self.order)
        self.current = 0
        self.agent_pos = self.start_pos
        self.carrying = False
        self.battery = round(order["bateria_inicial"] * self.battery_max)
        self.steps = 0
        self.total_reward = 0.0
        self.event = "reset"
        self.success = False
        self.rescues = 0
        return self._obs(), self._info()

    @property
    def package(self):
        return self.order[min(self.current, self.n_packages - 1)]

    def step(self, action):
        self.steps += 1
        pkg = self.package
        reward = STEP_PENALTY
        terminated = False

        if action in MOVES:
            heavy = self.carrying and pkg["type"] == HEAVY
            if heavy:
                reward = HEAVY_STEP_PENALTY
            self._drain(HEAVY_MOVE_COST if heavy else MOVE_COST)
            dr, dc = MOVES[action]
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self.grid[nxt] in (WALL, SHELF):
                reward += COLLISION_PENALTY
                self.event = "bump"
                if self.carrying and pkg["type"] == FRAGILE:
                    # el paquete se rompe: hay que volver a la estanteria por otro igual
                    reward += BREAK_PENALTY
                    self.carrying = False
                    pkg["status"] = "pendiente"
                    pkg["broken"] += 1
                    self.event = "break"
            else:
                self.agent_pos = nxt
                self.event = "move"
        elif action == PICK:
            if not self.carrying and self._adjacent(self.agent_pos, self.shelf_cells[pkg["shelf"]]):
                self.carrying = True
                pkg["status"] = "cargando"
                reward += PICK_REWARD
                self.event = "pick"
            else:
                reward += INVALID_ACTION_PENALTY
                self.event = "pick_fail"
        elif action == DROP:
            if self.carrying and self.agent_pos in self.zone_cells[pkg["zone"]]:
                self.carrying = False
                pkg["status"] = "entregado"
                self.current += 1
                reward += DELIVERY_REWARD
                self.event = "drop"
                if self.current == self.n_packages:
                    reward += ORDER_BONUS
                    terminated = self.success = True
            else:
                reward += INVALID_ACTION_PENALTY
                in_other_zone = self.carrying and any(self.agent_pos in z for z in self.zone_cells)
                self.event = "wrong_zone" if in_other_zone else "drop_fail"
        elif action == CHARGE:
            if self.use_battery and self.agent_pos == self.charger:
                self.battery = self.battery_max
                self.event = "charge"
            else:
                reward += INVALID_ACTION_PENALTY
                self.event = "charge_fail"

        if self.use_battery and self.battery <= 0 and not terminated:
            # bateria agotada: un operario lleva el robot a la estacion de carga
            reward += RESCUE_PENALTY
            self.agent_pos = self.charger
            self.battery = self.battery_max
            self.rescues += 1
            self.event = "rescue"

        truncated = not terminated and self.steps >= self.max_steps
        self.total_reward += reward
        return self._obs(), reward, terminated, truncated, self._info()

    def _drain(self, amount):
        if self.use_battery:
            self.battery = max(0, self.battery - amount)

    def battery_level(self):
        if not self.use_battery:
            return 0
        return math.ceil(self.battery * (BATTERY_LEVELS - 1) / self.battery_max)

    def _obs(self):
        pkg = self.package
        target = pkg["zone"] if self.carrying else pkg["shelf"]
        return np.array(
            [self.agent_pos[0], self.agent_pos[1], self.battery_level(), int(self.carrying),
             pkg["type"] if self.carrying else 0, target],
            dtype=np.int64,
        )

    def _info(self):
        return {"steps": self.steps, "delivered": self.current, "battery": self.battery,
                "carrying": self.carrying, "success": self.success, "rescues": self.rescues}

    @staticmethod
    def _adjacent(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def state_dict(self):
        """Estado completo serializable (para la visualizacion 3D)."""
        return {
            "pos": [int(self.agent_pos[0]), int(self.agent_pos[1])],
            "battery": int(self.battery),
            "carrying": bool(self.carrying),
            "current": int(self.current),
            "order": [dict(p) for p in self.order],
            "steps": int(self.steps),
            "total": float(self.total_reward),
            "event": self.event,
            "success": bool(self.success),
            "rescues": int(self.rescues),
        }

    def render(self):
        if self.render_mode != "ansi":
            return None
        canvas = self.grid.copy()
        for i, p in enumerate(self.order):
            if p["status"] == "pendiente":
                canvas[self.shelf_cells[p["shelf"]]] = "$" if i == self.current else "p"
        canvas[self.agent_pos] = "C" if self.carrying else "A"
        pkg = self.package
        status = (f"Bateria {self.battery}/{self.battery_max} | paquete {self.current + 1}/{self.n_packages} "
                  f"{TYPE_NAMES[pkg['type']]} -> zona {ZONE_NAMES[pkg['zone']]}")
        return "\n".join("".join(row) for row in canvas) + "\n" + status
