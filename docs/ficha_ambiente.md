# Ficha del ambiente · Warehouse Rush (`WarehouseRush-v1`, versión 1.0.0)

Un robot de bodega recibe un **pedido** de paquetes que debe recoger de las estanterías y entregar, en orden,
en la zona de entrega de cada paquete, administrando su batería.

Código: [`warehouse_rush/env_v1.py`](../warehouse_rush/env_v1.py) · niveles: [`configs/niveles.yaml`](../configs/niveles.yaml)
· pruebas: [`tests/test_ambiente.py`](../tests/test_ambiente.py).

## Mundo

| Elemento | Definición |
|---|---|
| Espacio | Cuadrícula de 11 × 15 casillas (1 casilla = 1 unidad de distancia). Paredes en todo el borde. |
| Estanterías | 40 casillas `S` (4 bloques de 2 × 5), no transitables. Un paquete se recoge desde una casilla vecina libre. |
| Zonas de entrega | A (columnas 1-3), B (6-8) y C (11-13) en la fila 9. |
| Estación de carga | Casilla `B` en (1, 7). |
| Obstáculos | Casillas `X` del nivel N3, no transitables (pasillo bloqueado). |
| Inicio | El robot empieza en (8, 7) mirando hacia arriba. |

## Dinámica

- **Paso de decisión:** el ambiente es discreto. Cada llamada a `step()` es una decisión y avanza el mundo un paso,
  sin tiempo físico continuo ni `dt`. La animación de la web interpola entre pasos, y su velocidad no cambia las reglas.
- **Movimiento:** una casilla por paso en la dirección elegida. Moverse contra una pared, una estantería o un
  obstáculo deja al robot en su sitio (choque).
- **Batería:** de 0 a `bateria_max` (100) unidades. La batería inicial de cada pedido está entre el 50 % y el 100 %.
  Cada movimiento o choque gasta 1 unidad, o 2 si lleva un paquete pesado. *Cargar* en la estación la llena.
  Si llega a 0, una grúa **rescata** al robot: lo lleva a la estación con la batería llena, conserva el paquete y
  recibe una penalización. El episodio continúa.
- **Pedido:** de 1 a 3 paquetes según el nivel, entregados en orden y uno a la vez. Cada paquete tiene estantería,
  zona de destino y tipo: *normal*; *pesado* (doble gasto de batería y penalización por paso mayor); o *frágil*
  (si el robot choca mientras lo lleva, se rompe y vuelve a la estantería).

## Acciones (`Discrete(7)`)

| Id | Acción | Válida cuando |
|---|---|---|
| 0-3 | arriba, abajo, izquierda, derecha | siempre (puede chocar) |
| 4 | recoger | junto a la estantería del paquete actual y sin carga |
| 5 | entregar | con carga y en la zona del paquete (en otra zona: *zona equivocada*) |
| 6 | cargar | en la estación de carga, en niveles con batería |

Una acción inválida no cambia el estado y se penaliza.

## Observación (`MultiDiscrete`)

| Índice | Variable | Rango | Unidad |
|---|---|---|---|
| 0 | fila | 0-10 | casilla |
| 1 | columna | 0-14 | casilla |
| 2 | nivel de batería | 0-4 (N1: siempre 0) | cuartos de batería, `ceil(4 · batería / máx.)` |
| 3 | lleva paquete | 0-1 | booleano |
| 4 | tipo del paquete que lleva | 0-2 | normal / pesado / frágil (0 si no lleva) |
| 5 | objetivo | 0-39 | estantería del paquete actual, o zona A/B/C (0-2) si lo lleva |

Las observaciones son enteros que se usan directamente como índices de la tabla Q: **no hay normalización,
escalado ni clipping.** La observación es una parte del estado: no incluye los paquetes futuros del pedido, la
batería exacta, la ubicación de los obstáculos ni los conteos de rotos o rescates.

**Pérdida de resolución por discretizar:** la batería exacta (0-100) se observa en 4 niveles, y el tipo del paquete
solo se informa mientras se lleva. Esto mantiene la tabla en 11 × 15 × 5 × 2 × 3 × 40 × 7 valores y hace que el
agente aprenda una sola forma de navegar para todos los tipos. El costo es que al ir a recoger un paquete no sabe
si será pesado.

## Recompensa (versionada con el código)

| Evento | Recompensa |
|---|---|
| Cada paso | -0.1 (-0.2 si lleva un paquete pesado) |
| Choque | -1 |
| Romper un frágil | -10 adicional |
| Acción inválida | -1 |
| Recoger | +5 (un paquete roto se reemplaza por otro nuevo, que vuelve a contar como hito) |
| Entregar en la zona correcta | +15 |
| Completar el pedido | +20 adicional |
| Rescate por batería agotada | -25 |

## Terminación

| Señal | Cuándo |
|---|---|
| `terminated` | Se entregó el último paquete del pedido (**éxito**). No hay fallo terminal: agotar la batería produce un rescate penalizado y el episodio sigue. |
| `truncated` | Se alcanzó el presupuesto externo de pasos (`max_pasos`: 200 en N1, 400 en N2/N3) sin completar el pedido. |

**Causa de fallo** en la evaluación (ver [`warehouse_rush/simulacion.py`](../warehouse_rush/simulacion.py)):
`fragil_roto` si se rompió algún frágil, `bateria_agotada` si hubo rescates, `choques_repetidos` con 20 o más choques,
y `tiempo_agotado` en el resto.

## Niveles

| Nivel | Decisión nueva del agente | Cambio | Familia de modelos |
|---|---|---|---|
| **N1 · Misión base** | Ir a la estantería, recoger y entregar en la zona indicada | 1 paquete normal, sin batería, 200 pasos | `obs-11x15x1x2x3x40` |
| **N2 · Más decisiones** | Orden de varias entregas, tipos de paquete y **cuándo cargar** | 3 paquetes (normal 50 %, pesado 25 %, frágil 25 %), batería, 400 pasos | `obs-11x15x5x2x3x40` |
| **N3 · Operación exigente** | **Recuperarse** de un pasillo bloqueado buscando otra ruta | N2 + obstáculos en (7, 4) y (7, 7) | igual que N2 (mismo contrato) |

- N2 y N3 comparten familia, así que un modelo de N2 se puede evaluar en N3 para **medir la degradación** del incidente.
- N1 tiene otra familia (la batería no existe), y la API rechaza los modelos incompatibles.

Cada nivel tiene su dataset de pedidos de entrenamiento y su **suite de evaluación**: pedidos distintos de los de
entrenamiento, generados con una semilla fija y solo con estanterías alcanzables. Ambos se versionan con DVC en
`data/<nivel>/`, y el identificador de la suite es el hash de sus casos.

## Referencias (sin aprendizaje)

| Política | Qué hace |
|---|---|
| Aleatoria | Acción uniforme al azar, con semilla. |
| Regla | Planificador BFS con el estado completo (información privilegiada): va al paquete, lo entrega y carga cuando la batería no alcanza para el tramo más el regreso a la estación. |

## Simplificaciones

- Movimiento en una cuadrícula, sin física continua, inercia ni giros (el robot "gira" al instante).
- Recoger y entregar son acciones discretas por adyacencia, sin brazo articulado ni estabilidad.
- Un solo robot y obstáculos estáticos dentro de cada nivel.
- La carga de batería es instantánea (una acción).
