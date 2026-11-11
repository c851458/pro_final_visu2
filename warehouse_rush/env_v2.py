"""Warehouse Rush v2 (version 2.0.0): centro de distribucion de paqueteria con recompensa en pesos colombianos
y nivel de servicio (SLA) Express / Estandar.

Contexto de negocio: un robot movil autonomo (AMR) de un centro de distribucion tipo Servientrega o Coordinadora
recoge los paquetes de una orden y los lleva al muelle de despacho de su ruta (zonas A, B y C). La recompensa es
la utilidad de la operacion en COP: ingresos por envio menos costos de operacion, danos, intervenciones humanas y
compensaciones por incumplir el SLA.

Cambios frente a v1 (las reglas de movimiento, bateria, fragiles y rescate son las mismas):
  - Cada paquete tiene un servicio: Estandar o Express (entrega el mismo dia, tarifa mayor y plazo corto).
  - El plazo (SLA) de cada paquete corre desde que pasa a ser el paquete actual del pedido.
  - La observacion agrega el servicio del paquete actual (familia de modelos nueva).
  - Recoger da su hito una sola vez por paquete (un fragil roto y vuelto a recoger no vuelve a sumar).
  - Cargar la bateria cuesta la energia recargada.

Valores y fuentes: ver docs/ficha_ambiente.md (seccion v2) y COSTOS abajo.
"""

import numpy as np
from gymnasium import spaces

from warehouse_rush.env_v1 import (
    BATTERY_LEVELS, BLOCKING, CHARGE, DROP, FRAGILE, HEAVY, HEAVY_MOVE_COST, MOVE_COST, MOVES, PICK, TYPE_NAMES,
    TYPE_PROBS, WarehouseRushV1Env,
)

SERVICE_NAMES = ["estandar", "express"]
STANDARD, EXPRESS = range(2)
EXPRESS_PROB = 0.3

# Plazos (SLA) en pasos de decision desde que el paquete pasa a ser el actual (1 paso ~ 2 s de operacion del AMR).
SLA_STEPS = {STANDARD: 100, EXPRESS: 40}

# Valores en pesos colombianos (COP, 2026). Un paso de decision ~ 2 segundos de operacion del robot.
COSTOS = {
    # Ingresos por paquete despachado (tarifa urbana del operador)
    "tarifa_estandar": 11_000,      # envio urbano estandar 1-2 kg: rango 9.000-13.000 (Coordinadora, Interrapidisimo)
    "tarifa_express": 24_000,       # Servientrega "hoy mismo" urbano, primer kilo (tarifario 2026)
    "hito_recogida": 2_000,         # parte de la tarifa que se reconoce al recoger (una sola vez por paquete)
    # Incumplimiento del SLA
    "compensacion_express_tarde": 13_000,  # se devuelve el sobrecosto Express: se cobra como estandar
    "compensacion_estandar_tarde": 3_300,  # descuento del 30% por entrega fuera de plazo
    # Costos de operacion
    "costo_paso": 5,                # AMR: ~7.000 COP/h de amortizacion+mantenimiento + 1.500 COP/h de supervision
    "costo_paso_pesado": 10,        # doble energia y desgaste con carga pesada
    "costo_choque": 1_500,          # inspeccion del robot por un supervisor (~6 min a ~15.000 COP/h)
    "costo_accion_invalida": 150,   # escaneo o maniobra fallida (tiempo perdido del robot)
    "indemnizacion_fragil": 60_000,  # valor declarado promedio de un paquete fragil roto
    "costo_rescate": 8_000,         # tecnico 20 min (SMMLV 2026 + prestaciones ~ 15.000 COP/h) + robot detenido
    "costo_carga_completa": 2_000,  # bateria AMR ~2 kWh x ~1.050 COP/kWh (tarifa comercial Enel 2026)
    "bono_pedido_completo": 4_000,  # ahorro por despacho consolidado de la orden completa
}


def random_order_v2(rng, n_packages, shelves, type_probs=TYPE_PROBS, express_prob=EXPRESS_PROB):
    """Pedido al azar con servicio por paquete (Estandar o Express) y bateria inicial."""
    shelves = [shelves[i] for i in rng.choice(len(shelves), n_packages, replace=False)]
    return {
        "paquetes": [
            {"shelf": int(s), "zone": int(rng.integers(3)), "type": int(rng.choice(len(TYPE_NAMES), p=type_probs)),
             "service": int(rng.random() < express_prob)}
            for s in shelves
        ],
        "bateria_inicial": int(rng.integers(50, 101)) / 100,
    }


class WarehouseRushV2Env(WarehouseRushV1Env):
    """Observacion: MultiDiscrete [fila, columna, nivel de bateria, cargando, tipo del paquete que lleva,
    objetivo, servicio del paquete actual (0 estandar, 1 express)]. Recompensa en COP."""

    FAMILY_PREFIX = "warehouse-rush-v2"

    def __init__(self, *args, express_prob=EXPRESS_PROB, **kwargs):
        super().__init__(*args, **kwargs)
        self.express_prob = express_prob
        self.observation_space = spaces.MultiDiscrete([
            self.height, self.width, BATTERY_LEVELS if self.use_battery else 1, 2,
            len(TYPE_NAMES), len(self.shelf_cells), len(SERVICE_NAMES),
        ])

    def reset(self, *, seed=None, options=None):
        super(WarehouseRushV1Env, self).reset(seed=seed)
        order = (options or {}).get("pedido") or random_order_v2(
            self.np_random, self.n_packages, self.reachable_shelves, express_prob=self.express_prob)
        self.order = [dict(p, status="pendiente", broken=0, picked_once=False, deadline=None, late=None,
                           delivered_at=None) for p in order["paquetes"]]
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
        self.ledger = {"ingresos": 0.0, "costos_operacion": 0.0, "danos": 0.0, "intervenciones": 0.0,
                       "compensaciones_sla": 0.0, "energia": 0.0}
        self._start_package()
        return self._obs(), self._info()

    def _start_package(self):
        if self.current < self.n_packages:
            pkg = self.order[self.current]
            pkg["deadline"] = self.steps + SLA_STEPS[pkg["service"]]

    def _book(self, account, amount):
        self.ledger[account] += amount
        return amount

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
            if self.grid[nxt] in BLOCKING:
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
            if not self.carrying and self._adjacent(self.agent_pos, self.shelf_cells[pkg["shelf"]]):
                self.carrying = True
                pkg["status"] = "cargando"
                if not pkg["picked_once"]:  # el hito se premia una sola vez por paquete
                    reward += self._book("ingresos", c["hito_recogida"])
                    pkg["picked_once"] = True
                self.event = "pick"
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                self.event = "pick_fail"
        elif action == DROP:
            if self.carrying and self.agent_pos in self.zone_cells[pkg["zone"]]:
                self.carrying = False
                pkg["status"] = "entregado"
                pkg["delivered_at"] = self.steps
                pkg["late"] = self.steps > pkg["deadline"]
                fare = c["tarifa_express"] if pkg["service"] == EXPRESS else c["tarifa_estandar"]
                reward += self._book("ingresos", fare - c["hito_recogida"])
                if pkg["late"]:
                    comp = c["compensacion_express_tarde"] if pkg["service"] == EXPRESS else c["compensacion_estandar_tarde"]
                    reward -= self._book("compensaciones_sla", comp)
                self.current += 1
                self.event = "drop"  # la tardanza queda en pkg["late"]
                if self.current == self.n_packages:
                    reward += self._book("ingresos", c["bono_pedido_completo"])
                    terminated = self.success = True
                else:
                    self._start_package()
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                in_other_zone = self.carrying and any(self.agent_pos in z for z in self.zone_cells)
                self.event = "wrong_zone" if in_other_zone else "drop_fail"
        elif action == CHARGE:
            if self.use_battery and self.agent_pos == self.charger:
                recharged = (self.battery_max - self.battery) / self.battery_max
                reward -= self._book("energia", round(c["costo_carga_completa"] * recharged))
                self.battery = self.battery_max
                self.event = "charge"
            else:
                reward -= self._book("costos_operacion", c["costo_accion_invalida"])
                self.event = "charge_fail"

        if self.use_battery and self.battery <= 0 and not terminated:
            reward -= self._book("intervenciones", c["costo_rescate"])
            self.agent_pos = self.charger
            self.battery = self.battery_max
            self.rescues += 1
            self.event = "rescue"

        truncated = not terminated and self.steps >= self.max_steps
        self.total_reward += reward
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _obs(self):
        return np.append(super()._obs(), self.package["service"]).astype(np.int64)

    def sla_summary(self):
        delivered = [p for p in self.order if p["status"] == "entregado"]
        express = [p for p in self.order if p["service"] == EXPRESS]
        return {
            "a_tiempo": sum(1 for p in delivered if not p["late"]),
            "tarde": sum(1 for p in delivered if p["late"]),
            "express": len(express),
            "express_a_tiempo": sum(1 for p in express if p["status"] == "entregado" and not p["late"]),
            "pedido_a_tiempo": self.success and all(not p["late"] for p in self.order),
        }

    def state_dict(self):
        data = super().state_dict()
        remaining = None
        if self.current < self.n_packages and self.package["deadline"] is not None:
            remaining = self.package["deadline"] - self.steps
        data.update(moneda="COP", libro=dict(self.ledger), sla_restante=remaining, servicios=SERVICE_NAMES)
        return data
