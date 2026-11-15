"""Warehouse Rush v3 (version 3.0.0): centro de distribucion de dos pisos con hora de corte de los camiones.

Contexto de negocio (el mismo de v2: utilidad en COP y servicio Express/Estandar), con dos cambios:

1. Dos pisos. El segundo piso replica las estanterias del primero (40 + 40 ubicaciones) y se conecta por un
   montacargas en el muro oriental. Los muelles de despacho y la estacion de carga estan en el primer piso, asi que
   un paquete del segundo piso exige subir, recogerlo, bajar y llevarlo al muelle. Un viaje en el montacargas dura
   TIEMPO_MONTACARGAS pasos (puertas y recorrido de un piso) y no gasta bateria del robot.
2. Hora de corte. Cada ruta (muelles A, B y C) tiene un camion que sale a una hora fija. Al llegar la orden, a cada
   camion le faltan entre corte_pasos[0] y corte_pasos[1] pasos para salir. El sistema de gestion de bodega (WMS)
   secuencia los paquetes de la orden por la salida de su camion (el que sale primero va primero). Un paquete que
   llega al muelle despues de que su camion salio se despacha al dia siguiente: llega tarde al cliente
   (compensacion del SLA) y genera un costo de reproceso (almacenamiento y re-manipulacion).

Las reglas de movimiento, bateria, fragiles y rescate son las de v1; la recompensa y el SLA, los de v2 con plazos
mas amplios (los recorridos entre pisos son mas largos). Valores y fuentes: docs/ficha_ambiente.md (seccion v3).
"""

import heapq

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from warehouse_rush.env_v1 import (
    ACTION_NAMES as ACTION_NAMES_V1, BATTERY_LEVELS, BATTERY_MAX, BLOCKING, CHARGE, CHARGER, DROP, FLOOR, FRAGILE,
    HEAVY, HEAVY_MOVE_COST, MOVE_COST, MOVES, OBSTACLE, PICK, SHELF, TYPE_NAMES, TYPE_PROBS, ZONE_CHARS, ZONE_NAMES,
)
from warehouse_rush.env_v2 import COSTOS as COSTOS_V2, EXPRESS, EXPRESS_PROB, SERVICE_NAMES, STANDARD, WarehouseRushV2Env

ELEVATOR_CHAR = "E"
ELEVATOR = 7  # accion: usar el montacargas (sube o baja un piso)
ACTION_NAMES = ACTION_NAMES_V1 + ["montacargas"]
FLOOR_NAMES = ["piso 1", "piso 2"]

LAYOUTS = [
    [  # piso 1: muelles de despacho, estacion de carga y montacargas (E, en el muro oriental)
        "###############",
        "#......B......#",
        "#.SS.SS.SS.SS.#",
        "#.SS.SS.SS.SS.#",
        "#.SS.SS.SS.SS.E",
        "#.SS.SS.SS.SS.#",
        "#.SS.SS.SS.SS.#",
        "#.............#",
        "#.............#",
        "#111..222..333#",
        "###############",
    ],
    [  # piso 2: las mismas estanterias; sin muelles ni estacion de carga
        "###############",
        "#.............#",
        "#.SS.SS.SS.SS.#",
        "#.SS.SS.SS.SS.#",
        "#.SS.SS.SS.SS.E",
        "#.SS.SS.SS.SS.#",
        "#.SS.SS.SS.SS.#",
        "#.............#",
        "#.............#",
        "#.............#",
        "###############",
    ],
]

TIEMPO_MONTACARGAS = 8  # pasos (~16 s): cerrar puertas, subir o bajar un piso (~4 m) y abrir
SLA_STEPS = {STANDARD: 150, EXPRESS: 90}  # plazos desde que el paquete pasa a ser el actual
CUTOFF_RANGE = (120, 360)  # pasos que le faltan a cada camion para salir cuando llega la orden

COSTOS = dict(
    COSTOS_V2,
    costo_montacargas=50,             # energia de un viaje: montacargas de ~10 kW x 16 s x ~1.050 COP/kWh
    costo_reproceso_camion=3_750,     # 15 min de un operario (~15.000 COP/h): almacenar, re-etiquetar y re-cargar
)


def random_order_v3(rng, n_packages, shelves, type_probs=TYPE_PROBS, express_prob=EXPRESS_PROB,
                    cutoff_range=CUTOFF_RANGE):
    """Pedido al azar: paquetes en estanterias de los dos pisos, servicio, bateria inicial y salida de cada camion.

    Los paquetes quedan secuenciados por la salida del camion de su ruta (orden estable)."""
    shelves = [shelves[i] for i in rng.choice(len(shelves), n_packages, replace=False)]
    packages = [
        {"shelf": int(s), "zone": int(rng.integers(len(ZONE_NAMES))),
         "type": int(rng.choice(len(TYPE_NAMES), p=type_probs)), "service": int(rng.random() < express_prob)}
        for s in shelves
    ]
    battery = int(rng.integers(50, 101)) / 100
    cutoff = [int(x) for x in rng.integers(cutoff_range[0], cutoff_range[1] + 1, len(ZONE_NAMES))]
    packages.sort(key=lambda p: cutoff[p["zone"]])
    return {"paquetes": packages, "bateria_inicial": battery, "corte": cutoff}


class WarehouseRushV3Env(WarehouseRushV2Env):
    """Observacion: MultiDiscrete [piso, fila, columna, nivel de bateria, cargando, tipo del paquete que lleva,
    objetivo (estanteria 0-79 de los dos pisos, o zona si lleva el paquete), servicio del paquete actual].
    Acciones: las 7 de v1 y 7 = montacargas. Recompensa en COP.

    Una posicion es (piso, fila, columna); `agent_pos` guarda (fila, columna) y `floor` el piso, para que el
    codigo comun (trayectorias, fotogramas) siga leyendo la fila y la columna igual que en v1 y v2."""

    FAMILY_PREFIX = "warehouse-rush-v3"
    N_FLOORS = len(LAYOUTS)
    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    def __init__(self, render_mode=None, n_packages=3, use_battery=True, max_steps=600, battery_max=BATTERY_MAX,
                 blocked_cells=(), express_prob=EXPRESS_PROB, cutoff_range=CUTOFF_RANGE):
        # No llama a WarehouseRushV1Env.__init__: el mapa tiene dos pisos.
        self.grids = [np.array([list(row) for row in layout]) for layout in LAYOUTS]
        self.blocked_cells = []
        for cell in blocked_cells:
            f, r, c = (int(x) for x in (cell if len(cell) == 3 else (0, *cell)))
            if self.grids[f][r, c] != FLOOR:
                raise ValueError(f"Solo se puede bloquear piso libre; ({f}, {r}, {c}) es '{self.grids[f][r, c]}'")
            self.grids[f][r, c] = OBSTACLE
            self.blocked_cells.append((f, r, c))
        self.height, self.width = self.grids[0].shape
        self.render_mode = render_mode
        self.n_packages = n_packages
        self.use_battery = use_battery
        self.max_steps = max_steps
        self.battery_max = battery_max
        self.express_prob = express_prob
        self.cutoff_range = tuple(cutoff_range)

        def cells(floor, ch):
            return [tuple(int(x) for x in p) for p in np.argwhere(self.grids[floor] == ch)]

        self.shelf_cells, self.shelf_floor = [], []
        for f in range(self.N_FLOORS):
            for cell in cells(f, SHELF):
                self.shelf_cells.append(cell)
                self.shelf_floor.append(f)
        self.zone_cells = [set(cells(0, z)) for z in ZONE_CHARS]
        self.charger = cells(0, CHARGER)[0]
        self.elevator = cells(0, ELEVATOR_CHAR)[0]
        if any(cells(f, ELEVATOR_CHAR) != [self.elevator] for f in range(self.N_FLOORS)):
            raise ValueError("El montacargas debe estar en la misma casilla en todos los pisos")
        self.start_pos = (8, 7)
        self.floor = 0
        self.reachable_shelves = self._reachable_shelves()

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.MultiDiscrete([
            self.N_FLOORS, self.height, self.width, BATTERY_LEVELS if use_battery else 1, 2,
            len(TYPE_NAMES), len(self.shelf_cells), len(SERVICE_NAMES),
        ])

    @property
    def grid(self):
        """Mapa del piso donde esta el robot."""
        return self.grids[self.floor]

    # ---------------------------------------------------------------- mapa de dos pisos

    def passable(self, floor, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grids[floor][r, c] not in BLOCKING

    def neighbors(self, node):
        """Vecinos de una posicion (piso, fila, columna): [(accion, vecino, pasos de tiempo, unidades de movimiento)].

        El montacargas cambia de piso en TIEMPO_MONTACARGAS pasos sin moverse (no gasta bateria)."""
        f, r, c = node
        result = []
        for action, (dr, dc) in MOVES.items():
            if self.passable(f, r + dr, c + dc):
                result.append((action, (f, r + dr, c + dc), 1, 1))
        if (r, c) == self.elevator:
            for g in range(self.N_FLOORS):
                if g != f:
                    result.append((ELEVATOR, (g, r, c), TIEMPO_MONTACARGAS, 0))
        return result

    def distances(self, sources):
        """Dijkstra desde un conjunto de posiciones: {posicion: (pasos de tiempo, unidades de movimiento)}.

        El grafo es simetrico, asi que tambien es la distancia de cada posicion hasta el conjunto."""
        dist = {s: (0, 0) for s in sources}
        heap = [(0, 0, s) for s in sources]
        heapq.heapify(heap)
        while heap:
            t, m, node = heapq.heappop(heap)
            if dist.get(node, (np.inf, np.inf)) < (t, m):
                continue
            for _, nxt, dt, dm in self.neighbors(node):
                cand = (t + dt, m + dm)
                if cand < dist.get(nxt, (np.inf, np.inf)):
                    dist[nxt] = cand
                    heapq.heappush(heap, (cand[0], cand[1], nxt))
        return dist

    def pick_faces(self, shelf):
        """Posiciones desde las que se recoge un paquete de una estanteria (casillas vecinas libres, mismo piso)."""
        f, (r, c) = self.shelf_floor[shelf], self.shelf_cells[shelf]
        return frozenset((f, r + dr, c + dc) for dr, dc in MOVES.values() if self.passable(f, r + dr, c + dc))

    def _reachable_shelves(self):
        seen = self.distances([(0, *self.start_pos)])
        return [i for i in range(len(self.shelf_cells)) if any(face in seen for face in self.pick_faces(i))]

    @property
    def position(self):
        return (self.floor, *self.agent_pos)

    # ---------------------------------------------------------------- dinamica

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)  # siembra np_random (los reset de v1 y v2 no aplican)
        order = (options or {}).get("pedido") or random_order_v3(
            self.np_random, self.n_packages, self.reachable_shelves, express_prob=self.express_prob,
            cutoff_range=self.cutoff_range)
        self.order = [dict(p, status="pendiente", broken=0, picked_once=False, deadline=None, late=None,
                           delivered_at=None, camion_perdido=None) for p in order["paquetes"]]
        self.cutoff = [int(x) for x in order["corte"]]
        self.n_packages = len(self.order)
        self.current = 0
        self.agent_pos = self.start_pos
        self.floor = 0
        self.carrying = False
        self.battery = round(order["bateria_inicial"] * self.battery_max)
        self.steps = 0
        self.total_reward = 0.0
        self.event = "reset"
        self.success = False
        self.rescues = 0
        self.ledger = {"ingresos": 0.0, "costos_operacion": 0.0, "danos": 0.0, "intervenciones": 0.0,
                       "compensaciones_sla": 0.0, "energia": 0.0, "reproceso_camion": 0.0}
        self._start_package()
        return self._obs(), self._info()

    def _start_package(self):
        if self.current < self.n_packages:
            pkg = self.order[self.current]
            pkg["deadline"] = self.steps + SLA_STEPS[pkg["service"]]

    def step(self, action):
        self.steps += 1
        pkg = self.package
        c = COSTOS
        terminated = False
        heavy = self.carrying and pkg["type"] == HEAVY
        reward = -self._book("costos_operacion", c["costo_paso_pesado"] if heavy else c["costo_paso"])

        if action in MOVES:
            self._drain(HEAVY_MOVE_COST if heavy else MOVE_COST)
            dr, dc = MOVES[action]
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if not self.passable(self.floor, *nxt):
                reward -= self._book("costos_operacion", c["costo_choque"])
                self.event = "bump"
                if self.carrying and pkg["type"] == FRAGILE:
                    reward -= self._book("danos", c["indemnizacion_fragil"])
                    self.carrying = False
                    pkg["status"] = "pendiente"
                    pkg["broken"] += 1
                    self.event = "break"
            else:
                self.agent_pos = nxt
                self.event = "move"
        elif action == PICK:
            shelf = pkg["shelf"]
            if (not self.carrying and self.floor == self.shelf_floor[shelf]
                    and self._adjacent(self.agent_pos, self.shelf_cells[shelf])):
                self.carrying = True
                pkg["status"] = "cargando"
                if not pkg["picked_once"]:
                    reward += self._book("ingresos", c["hito_recogida"])
                    pkg["picked_once"] = True
                self.event = "pick"
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                self.event = "pick_fail"
        elif action == DROP:
            on_dock_floor = self.floor == 0
            if self.carrying and on_dock_floor and self.agent_pos in self.zone_cells[pkg["zone"]]:
                self.carrying = False
                pkg["status"] = "entregado"
                pkg["delivered_at"] = self.steps
                pkg["camion_perdido"] = self.steps > self.cutoff[pkg["zone"]]
                pkg["late"] = self.steps > pkg["deadline"] or pkg["camion_perdido"]
                fare = c["tarifa_express"] if pkg["service"] == EXPRESS else c["tarifa_estandar"]
                reward += self._book("ingresos", fare - c["hito_recogida"])
                if pkg["late"]:  # tarde por el plazo o porque sale en el camion del dia siguiente (una sola vez)
                    comp = c["compensacion_express_tarde"] if pkg["service"] == EXPRESS else c["compensacion_estandar_tarde"]
                    reward -= self._book("compensaciones_sla", comp)
                if pkg["camion_perdido"]:
                    reward -= self._book("reproceso_camion", c["costo_reproceso_camion"])
                self.current += 1
                self.event = "drop"
                if self.current == self.n_packages:
                    reward += self._book("ingresos", c["bono_pedido_completo"])
                    terminated = self.success = True
                else:
                    self._start_package()
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                in_other_zone = self.carrying and on_dock_floor and any(self.agent_pos in z for z in self.zone_cells)
                self.event = "wrong_zone" if in_other_zone else "drop_fail"
        elif action == CHARGE:
            if self.use_battery and self.floor == 0 and self.agent_pos == self.charger:
                recharged = (self.battery_max - self.battery) / self.battery_max
                reward -= self._book("energia", round(c["costo_carga_completa"] * recharged))
                self.battery = self.battery_max
                self.event = "charge"
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                self.event = "charge_fail"
        elif action == ELEVATOR:
            if self.agent_pos == self.elevator:
                self.floor = (self.floor + 1) % self.N_FLOORS
                extra = TIEMPO_MONTACARGAS - 1  # el paso de esta decision ya se conto
                self.steps += extra
                reward -= self._book("costos_operacion", extra * (c["costo_paso_pesado"] if heavy else c["costo_paso"]))
                reward -= self._book("energia", c["costo_montacargas"])
                self.event = "elevator"
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                self.event = "elevator_fail"

        if self.use_battery and self.battery <= 0 and not terminated:
            reward -= self._book("intervenciones", c["costo_rescate"])
            self.agent_pos = self.charger
            self.floor = 0
            self.battery = self.battery_max
            self.rescues += 1
            self.event = "rescue"

        truncated = not terminated and self.steps >= self.max_steps
        self.total_reward += reward
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _obs(self):
        pkg = self.package
        target = pkg["zone"] if self.carrying else pkg["shelf"]
        return np.array(
            [self.floor, self.agent_pos[0], self.agent_pos[1], self.battery_level(), int(self.carrying),
             pkg["type"] if self.carrying else 0, target, pkg["service"]],
            dtype=np.int64,
        )

    def _info(self):
        return dict(super()._info(), floor=self.floor)

    def sla_summary(self):
        return dict(super().sla_summary(), camiones_perdidos=sum(1 for p in self.order if p["camion_perdido"]))

    def state_dict(self):
        data = super().state_dict()
        data.update(piso=int(self.floor), corte=list(self.cutoff),
                    camiones=["salio" if self.steps >= t else "esperando" for t in self.cutoff])
        return data

    def map_info(self):
        """Mapa para la visualizacion: el piso 1 en `layout` (compatibilidad) y todos en `pisos`.
        Las estanterias y los bloqueos llevan el piso como tercer elemento: [fila, columna, piso]."""
        return {
            "layout": LAYOUTS[0], "pisos": [list(layout) for layout in LAYOUTS],
            "shelf_cells": [[r, c, f] for (r, c), f in zip(self.shelf_cells, self.shelf_floor)],
            "zone_cells": [sorted(list(p) for p in z) for z in self.zone_cells], "charger": list(self.charger),
            "start": list(self.start_pos), "bloqueos": [[r, c, f] for f, r, c in self.blocked_cells],
            "elevador": list(self.elevator), "tiempo_montacargas": TIEMPO_MONTACARGAS,
        }

    def render(self):
        if self.render_mode != "ansi":
            return None
        canvases = [g.copy() for g in self.grids]
        for i, p in enumerate(self.order):
            if p["status"] == "pendiente":
                canvases[self.shelf_floor[p["shelf"]]][self.shelf_cells[p["shelf"]]] = "$" if i == self.current else "p"
        canvases[self.floor][self.agent_pos] = "C" if self.carrying else "A"
        rows = ["   ".join("".join(canvases[f][r]) for f in range(self.N_FLOORS)) for r in range(self.height)]
        pkg = self.package
        trucks = " ".join(f"{z}:{t - self.steps:+d}" for z, t in zip(ZONE_NAMES, self.cutoff))
        status = (f"{FLOOR_NAMES[self.floor]} | Bateria {self.battery}/{self.battery_max} | paquete "
                  f"{self.current + 1}/{self.n_packages} {TYPE_NAMES[pkg['type']]} -> zona {ZONE_NAMES[pkg['zone']]} | "
                  f"camiones {trucks}")
        return "\n".join(rows) + "\n" + status
