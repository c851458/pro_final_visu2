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

## Observación para redes neuronales (familia `ppo-caracteristicas-22`)

Los modelos de red neuronal (PPO e imitación) no usan la observación tabular. Reciben un vector de **22
características continuas** calculado del estado del ambiente por `warehouse_rush/ppo.py::features`. La misma
función se usa al entrenar y al servir, así que la API evalúa exactamente lo que se entrenó. Incluye lo que la tabla Q
no ve: la **batería exacta** y el **plazo restante del SLA**.

| Características | Normalización |
|---|---|
| fila, columna | posición / tamaño de la grilla, en [0, 1] |
| batería | batería / batería máxima (1 si el nivel no tiene batería) |
| lleva paquete; tipo normal, pesado o frágil (one-hot); Express | 0 o 1 |
| plazo restante del SLA | (vencimiento − paso actual) / 100, recortado a [−1, 1] |
| distancia al objetivo en filas y columnas, y distancia Manhattan | diferencias / tamaño de la grilla |
| distancia al cargador en filas y columnas | diferencias / tamaño de la grilla |
| bloqueado arriba, abajo, izquierda, derecha | 0 o 1 (pared, estantería u obstáculo) |
| junto a la estantería objetivo; en el muelle de destino; en el cargador | 0 o 1 |
| paquetes restantes del pedido | restantes / total |

El **objetivo** es la estantería del paquete actual o, si ya lo lleva, el centro del muelle de su zona.

- **Familia:** `warehouse-rush-v2/ppo-caracteristicas-22/acciones-7`. Es compatible con `v2-n2` y `v2-n3`, y la API
  rechaza servirla en niveles de otra familia.
- **Red:** 22 → 128 → 128 → 7, con activación tanh. La acción servida es el argmax de los logits, sin exploración.
- **Artefacto:** los pesos se exportan a NumPy (`politica.npz`) y la inferencia es NumPy puro, así que la API no
  necesita PyTorch (~700 MB) y cabe en el límite de 500 MB de Vercel. El manifiesto lo identifica con
  `tipo_modelo: ppo-mlp`.
- **Normalización:** fija y escrita en el código (tabla de arriba), sin estadísticas aprendidas. Por eso no hay un
  archivo de normalización (`normalization_uri` es nulo): el campo `normalizacion` del manifiesto describe estas
  reglas. La recompensa se escaló ×10⁻³ solo durante el entrenamiento PPO.

---

# Versión 3.0.0 · dos pisos y hora de corte (`WarehouseRush-v3`, niveles `v3-n1`, `v3-n2`, `v3-n3`)

El mismo centro de distribución de v2 (recompensa en COP, servicio Express/Estándar), ampliado con dos variables de
la operación real. Código: [`warehouse_rush/env_v3.py`](../warehouse_rush/env_v3.py). Las versiones 1.0.0 y 2.0.0 se
conservan sin cambios: sus datasets se regeneraron y son idénticos byte a byte.

## Qué cambia frente a v2

| Elemento | v2 | v3 |
|---|---|---|
| Edificio | un piso, 40 estanterías | **dos pisos** con las mismas estanterías (40 + 40) |
| Conexión entre pisos | — | **montacargas** en el muro oriental, casilla (4, 14) en ambos pisos |
| Muelles y estación de carga | piso único | solo en el **piso 1**; el piso 2 tiene un corredor libre donde abajo están los muelles |
| Acciones | 7 | **8**: la 7 usa el montacargas (sube o baja un piso) |
| Hora de corte | — | cada ruta (A, B, C) tiene un **camión** que sale a una hora fija |
| Secuencia del pedido | la del generador | el sistema de gestión de bodega (WMS) ordena los paquetes por la salida de su camión |
| Plazos del SLA | Estándar 100 pasos, Express 40 | Estándar 150, Express 90 (los recorridos entre pisos son más largos) |
| Observación tabular | 7 variables | 8: se agrega el piso; el objetivo indexa las 80 estanterías |
| Familia tabular | `warehouse-rush-v2/obs-11x15x5x2x3x40x2/acciones-7` | `warehouse-rush-v3/obs-2x11x15x5x2x3x80x2/acciones-8` |
| Familia de redes | `warehouse-rush-v2/ppo-caracteristicas-22/acciones-7` | `warehouse-rush-v3/ppo-caracteristicas-30/acciones-8` |

## Montacargas

Usar el montacargas en su casilla cambia de piso en **8 pasos** (~16 s: cerrar puertas, recorrer ~4 m y abrir). No
gasta batería del robot (el montacargas tiene su propia alimentación), pero el robot está detenido: se cobra el
costo por paso de esos 8 pasos y **$50** de energía del montacargas (~10 kW × 16 s × ~$1.050/kWh). Usarlo fuera de su
casilla es una acción inválida. Si la batería se agota en el piso 2, el técnico lleva el robot a la estación de carga
del piso 1.

## Hora de corte de los camiones

Al llegar la orden, a cada camión le faltan entre `corte_pasos[0]` y `corte_pasos[1]` pasos para salir (60-200 en
`v3-n1`, 120-360 en `v3-n2` y `v3-n3`). El camión sale aunque falten paquetes. Un paquete que llega al muelle después
de la salida de su camión se despacha al día siguiente:

- **llega tarde al cliente:** se paga la compensación del SLA (la misma de v2), una sola vez aunque también haya
  vencido su plazo;
- **reproceso:** **$3.750** por almacenarlo, re-etiquetarlo y volver a cargarlo (15 min de un operario a
  ~$15.000/h, con el mismo supuesto de salario de v2).

El OTIF exige que todos los paquetes lleguen a tiempo y en su camión. KPIs nuevos: **camiones perdidos por orden**,
**paquetes en su camión** (cumplimiento del corte) y **viajes en montacargas**.

## Calibración (planificador de reglas, sin aprendizaje)

Los plazos y las ventanas de corte se fijaron midiendo al planificador de reglas (BFS sobre los dos pisos) antes de
entrenar. Un paquete del piso 1 tarda ~20-30 pasos y uno del piso 2 ~70 (p95 de 90 a 98). Con las ventanas elegidas el
planificador completa el 100 % de las órdenes y pierde algunos camiones, así que la variable sí pesa sin volver
imposible el nivel:

| Nivel | Éxito | OTIF | Camiones perdidos / orden | Utilidad / orden | Viajes en montacargas |
|---|---|---|---|---|---|
| `v3-n1` | 100 % | 97,5 % | 0,025 | $17.568 | 1,1 |
| `v3-n2` | 100 % | 94,6 % | 0,032 | $45.116 | 2,9 |
| `v3-n3` | 100 % | 87,6 % | 0,086 | $43.996 | 2,9 |

(Suites de evaluación de 200 y 500 pedidos de cada nivel.) El planificador no mira la hora de corte: va paquete por
paquete en el orden del WMS y carga cuando la batería no alcanza. Una política que priorice la salida de los camiones
podría superarlo; queda como trabajo futuro.

## Observación para redes neuronales (familia `ppo-caracteristicas-30`)

Las 22 características de v2 se adaptan a dos pisos y se agregan las del montacargas y el camión
(`warehouse_rush/ppo.py::FEATURE_NAMES_V3`):

| Características nuevas o cambiadas | Normalización |
|---|---|
| piso del robot | 0 o 1 |
| tiempo hasta la salida del camión del paquete actual | (salida − paso actual) / 300, recortado a [−1, 1] |
| el camión ya salió | 0 o 1 |
| el objetivo está en otro piso | 0 o 1 |
| destino en filas, columnas y distancia Manhattan | hacia el objetivo si está en el piso del robot; si no, hacia el montacargas |
| montacargas en filas y columnas; en el montacargas | diferencias / tamaño de la grilla; 0 o 1 |
| el cargador está en otro piso | 0 o 1 |

- **Red:** 30 → 128 → 128 → 8, con activación tanh; el resto (exportación a NumPy, inferencia sin PyTorch) es igual
  que en v2.

## Simplificaciones de v3

- Dos pisos idénticos y un solo montacargas, siempre disponible (sin cola de otros robots ni mantenimiento).
- La hora de corte es la misma para todos los paquetes de una ruta y el camión no espera.
- El robot sigue la secuencia del WMS: no elige qué paquete atender primero.
