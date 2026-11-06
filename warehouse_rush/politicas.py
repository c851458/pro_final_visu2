"""Politicas que pueden controlar el robot: aleatoria, de regla (referencia) y tabla Q (modelo aprendido).

Todas exponen `act(env, obs) -> accion`. La de regla usa el estado completo del ambiente (informacion
privilegiada): sirve como referencia de lo que logra un planificador sin aprendizaje.
"""

from collections import deque

import numpy as np

from warehouse_rush.env_v1 import BLOCKING, CHARGE, DROP, HEAVY, HEAVY_MOVE_COST, MOVE_COST, MOVES, PICK


class RandomPolicy:
    kind = "aleatoria"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return int(self.rng.integers(env.action_space.n))


class QTablePolicy:
    """Modelo aprendido: accion de mayor valor en la tabla Q, sin exploracion."""

    kind = "modelo"

    def __init__(self, q):
        self.q = q

    def act(self, env, obs):
        return int(np.argmax(self.q[tuple(obs)]))


class RulePolicy:
    """Planificador por busqueda en anchura (BFS): va al paquete, lo entrega y carga cuando no le alcanza."""

    kind = "regla"
    MARGIN = 4  # unidades de bateria de reserva

    def act(self, env, obs):
        base = env.unwrapped
        pkg = base.package
        if base.carrying:
            goals, final = base.zone_cells[pkg["zone"]], DROP
        else:
            r, c = base.shelf_cells[pkg["shelf"]]
            goals = {(r + dr, c + dc) for dr, dc in MOVES.values() if base.grid[r + dr, c + dc] not in BLOCKING}
            final = PICK

        if base.use_battery and self._needs_charge(base, goals):
            if base.agent_pos == base.charger:
                return CHARGE
            return self._step_towards(base, {base.charger})
        if base.agent_pos in goals:
            return final
        return self._step_towards(base, goals)

    def _needs_charge(self, base, goals):
        if base.agent_pos == base.charger and base.battery >= base.battery_max:
            return False
        heavy = base.carrying and base.package["type"] == HEAVY
        cost = HEAVY_MOVE_COST if heavy else MOVE_COST
        to_goal = self._distance(base, base.agent_pos, goals)
        goal = min(goals, key=lambda g: self._distance(base, base.agent_pos, {g}))
        back = self._distance(base, goal, {base.charger})
        # al recoger, el regreso puede ser con un paquete pesado
        back_cost = HEAVY_MOVE_COST if (heavy or (not base.carrying and base.package["type"] == HEAVY)) else MOVE_COST
        return base.battery < to_goal * cost + back * back_cost + self.MARGIN

    @staticmethod
    def _bfs(base, start, goals):
        """Devuelve (distancia, primera accion) del camino mas corto de start a cualquier meta."""
        if start in goals:
            return 0, None
        seen = {start}
        queue = deque([(start, 0, None)])
        while queue:
            (r, c), dist, first = queue.popleft()
            for action, (dr, dc) in MOVES.items():
                n = (r + dr, c + dc)
                if n in seen or base.grid[n] in BLOCKING:
                    continue
                if n in goals:
                    return dist + 1, first if first is not None else action
                seen.add(n)
                queue.append((n, dist + 1, first if first is not None else action))
        return float("inf"), None

    def _distance(self, base, start, goals):
        return self._bfs(base, start, goals)[0]

    def _step_towards(self, base, goals):
        _, action = self._bfs(base, base.agent_pos, goals)
        return action if action is not None else CHARGE
