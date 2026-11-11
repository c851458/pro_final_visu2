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

---

# Versión 2.0.0 · contexto de negocio (`WarehouseRush-v2`, niveles `v2-n1`, `v2-n2`, `v2-n3`)

El mismo mundo, interpretado como el **centro de distribución de un operador de paquetería** (tipo Servientrega
o Coordinadora): un robot móvil autónomo (AMR) recoge los paquetes de una orden y los lleva al **muelle de despacho**
de su ruta (zonas A, B y C). La recompensa es la **utilidad de la operación en pesos colombianos (COP)** y cada
paquete tiene un **nivel de servicio**. Código: [`warehouse_rush/env_v2.py`](../warehouse_rush/env_v2.py).

La versión 1.0.0 se conserva sin cambios: los modelos entrenados en v1 se siguen evaluando con sus reglas originales.

## Qué cambia frente a v1

| Elemento | v1 | v2 |
|---|---|---|
| Recompensa | puntos abstractos | utilidad en COP (ingresos − costos) |
| Servicio del paquete | — | Estándar (70 %) o Express (30 %), con plazo |
| Observación | 6 variables | 7: se agrega el servicio del paquete actual (0 Estándar, 1 Express) |
| Familia de modelos | `warehouse-rush-v1/...` | `warehouse-rush-v2/obs-...x2/...` (incompatible con v1) |
| Hito de recoger | se repetía al volver a recoger un frágil roto | **una sola vez por paquete** |
| Cargar batería | gratis | cuesta la energía recargada |

## Nivel de servicio (SLA)

El plazo de cada paquete corre **desde que pasa a ser el paquete actual** de la orden. Un paso de decisión equivale
a unos 2 segundos de operación del AMR (aproximadamente 1,5 m a 1 m/s más la maniobra).

| Servicio | Probabilidad | Plazo | Tarifa | Si llega tarde |
|---|---|---|---|---|
| Estándar | 70 % | 100 pasos | $11.000 | descuento del 30 % (−$3.300) |
| Express | 30 % | 40 pasos | $24.000 | se devuelve el sobrecosto Express (−$13.000): se cobra como Estándar |

El agente observa si el paquete actual es Express, pero **no el tiempo restante**. Es una simplificación para
mantener la tabla Q manejable: debe aprender que con un Express los desvíos (por ejemplo, ir a cargar) son más caros.

## Modelo de costos (COP, 2026)

| Concepto | Valor | Supuesto y fuente |
|---|---|---|
| Tarifa Estándar | $11.000 | envío urbano de 1-2 kg; rango de $9.000-$13.000 reportado para Coordinadora e Interrapidísimo ([comparativa 2026](https://www.andreybusiness.com/blog/interrapidisimo-vs-coordinadora-comparativa-2026)) |
| Tarifa Express | $24.000 | servicio urbano "hoy mismo", primer kilo, del tarifario de Servientrega 2026 ([UIS, feb. 2026](https://documentos.uis.edu.co/wp-content/uploads/2026/02/tarifas-de-operador-de-mensajeria-externa-servientrega.pdf)) |
| Hito de recogida | $2.000 | parte de la tarifa que se reconoce al recoger (una sola vez); el resto se cobra al despachar |
| Costo por paso | $5 ($10 con carga pesada) | AMR de ~$140 millones amortizado en 5 años a 2 turnos (~$7.000/h con mantenimiento) más una fracción de supervisión |
| Choque | $1.500 | inspección del robot por un supervisor (~6 min) |
| Acción inválida | $150 | escaneo o maniobra fallida: tiempo perdido del robot |
| Frágil roto | $60.000 | indemnización por el valor declarado promedio |
| Rescate (batería agotada) | $8.000 | técnico 20 min (salario mínimo 2026 de $1.750.905 + $249.095 de auxilio de transporte + prestaciones ≈ $15.000/h, [Buk](https://www.buk.co/blog/salario-minimo-2026-en-colombia)) más el robot detenido |
| Carga de batería | hasta $2.000 | batería AMR de ~2 kWh × ~$1.050/kWh, tarifa comercial de Enel ([tarifario julio 2026](https://www.enel.com.co/content/dam/enel-co/español/personas/1-17-1/2026/Consulta%20el%20tarifario%20de%20energía%20Enel%20-%20Julio%202026.pdf)); proporcional a la energía recargada |
| Orden completa | +$4.000 | ahorro por despacho consolidado de la orden |

**Límites de los valores:** no hay una cifra única y oficial para la tarifa urbana Estándar ni para la tarifa
comercial de energía. El valor de $1.750.905 del salario mínimo fue suspendido y reemplazado por un decreto
transitorio con el mismo valor. Son aproximaciones razonables para comparar políticas, no una contabilidad real.

## KPIs de negocio

| KPI | Definición |
|---|---|
| **OTIF** | órdenes completas con **todos** sus paquetes a tiempo |
| Cumplimiento del SLA | paquetes entregados a tiempo / paquetes totales |
| Cumplimiento Express | Express entregados a tiempo / Express totales |
| Utilidad por orden | recompensa total en COP |
| Libro de la orden | ingresos, operación del robot, daños, intervenciones humanas, compensaciones SLA y energía |
