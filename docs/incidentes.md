# Informe de incidentes · Warehouse Rush

Registro de los incidentes del proyecto: qué pasó, cómo se detectó, la causa raíz, la respuesta y lo que se cambió
para que no se repita. Las horas están en hora de Colombia (UTC−5). Cada afirmación enlaza su evidencia: commits,
runs de MLflow, informes de decisión o registros de los flujos.

| # | Fecha | Tipo | Impacto en producción | Estado |
|---|---|---|---|---|
| [INC-1](#inc-1--pasillo-bloqueado-en-n3-el-modelo-de-n2-cae-de-986--a-54-) | 2026-11-06 | Cambio en el ambiente (modelo) | Ninguno: detectado antes de servir N3 | Resuelto: reentrenado desde cero (99,1 %) |
| [INC-2](#inc-2--runs-de-negocio-v2-entrenados-con-el-ambiente-v1) | 2026-11-11 | Configuración del entrenamiento | Ninguno: detectado antes de publicar | Resuelto: corrección, prueba de regresión y 10 runs invalidados |
| [INC-3](#inc-3--el-brazo-b-de-imitación-no-arranca-y-se-reintenta-toda-la-noche) | 2026-11-13/14 | Pipeline de entrenamiento | Ninguno; la publicación se retrasó ~8 h | Resuelto: corrección, pruebas y detección de errores de código |
| [INC-4](#inc-4--simulacro-modelo-rechazado-forzado-a-producción) | 2026-11-14 | **Simulacro** planificado | 5 min con un modelo peor en v2-n2 | Revertido automáticamente por la vigilancia |

---

## INC-1 · Pasillo bloqueado en N3: el modelo de N2 cae de 98,6 % a 5,4 %

**Resumen.** El nivel N3 representa un incidente operativo del centro de distribución: un pasillo bloqueado en las
celdas (7, 4) y (7, 7), por ejemplo por un derrame o un montacargas varado. N2 y N3 tienen el mismo contrato de
observaciones y acciones (misma familia de modelos), así que el modelo de N2 puede servirse en N3, pero no ve el bloqueo.

**Detección.** Antes de servir N3, el modelo recomendado de N2 se evaluó en la suite de N3 (run
`evaluacion-pipeline-n2-en-n3`). La hipótesis registrada era: *"el modelo de N2 no observa los bloqueos; se espera una
caída fuerte de éxito en N3"*.

| Modelo | Suite | Éxito |
|---|---|---|
| `pipeline-n2` (Q-learning) | N2 | 98,6 % |
| `pipeline-n2` (Q-learning) | N3 | **5,4 %** |
| Política de reglas (BFS, referencia) | N3 | 100 % |

**Causa raíz.** La tabla Q aprendió rutas fijas que atraviesan el pasillo ahora bloqueado. La observación tabular
incluye la posición del robot, pero no los obstáculos, así que el agente no puede adaptarse sin reentrenar.

**Respuesta.**
1. **Intervención rechazada:** continuar desde el modelo de N2 con poca exploración (ε inicial 0,3, 15.000 episodios)
   llegó a solo **23,6 %** (run `n3-recuperacion`). El modelo no exploraba lo suficiente para desaprender la ruta bloqueada.
2. **Contraste pre-registrado** ([`contraste-n3`](resultados/contraste-n3.md), criterios fijados antes de entrenar en
   el commit `3a00d17`), con 3 semillas por brazo:
   - A, reentrenar desde cero (40.000 episodios): **99,1 % ± 0,1**; test reservado 98,7 % ± 0,5.
   - B, continuar desde N2 con exploración completa (25.000 episodios): 17,5 % ± 3,8.
3. **Promoción:** `contraste-n3-A-s2` (99,2 %) se publicó y se promovió con el flujo de Prefect `99d63c43`
   ([evidencia](ejecuciones/20261106-223406-99d63c43.json)) y quedó en producción.

**Lecciones.**
- Un modelo compatible por contrato no es un modelo válido para el nuevo ambiente. La compatibilidad de familia
  permite *medir* la degradación, no la evita.
- Continuar desde un modelo previo no ahorró presupuesto: las rutas aprendidas en N2 estorbaron más de lo que
  ayudaron, incluso con exploración completa.

---

## INC-2 · Runs de negocio (v2) entrenados con el ambiente v1

**Resumen.** Al introducir el ambiente de negocio v2.0.0 (commit `0165eee`), los entrenamientos de los niveles
`v2-*` se ejecutaron con el ambiente `WarehouseRush-v1`: sin recompensa en pesos y sin SLA.

**Detección.** Al revisar los runs de los niveles v2, el mismo día y antes de publicar ningún modelo de esos niveles
(el primero, `v2-n1`, se promovió el 13 de noviembre).

**Causa raíz.** `pipeline/common.py::resolve_params` no copiaba el campo `ambiente` del nivel a la configuración
resuelta, así que `make_env` usaba su valor por defecto (`WarehouseRush-v1`).

**Respuesta** (commit `39a6820`):
- `resolve_params` incluye `ambiente` y `version_ambiente`.
- Prueba de regresión en [`tests/test_ambiente_v2.py`](../tests/test_ambiente_v2.py): la configuración de
  entrenamiento de cada nivel usa el ambiente de ese nivel.
- Los **10 runs afectados se conservaron en MLflow**, marcados como invalidados con su diagnóstico, en lugar de
  borrarlos: se entrenaron y evaluaron con `WarehouseRush-v1` usando pedidos v2.

**Lección.** Un valor por defecto silencioso en la configuración puede invalidar experimentos sin dar ningún error.
La prueba ahora verifica el ambiente efectivo, no solo el nombre del nivel.

---

## INC-3 · El brazo B de imitación no arranca y se reintenta toda la noche

**Resumen.** En el contraste pre-registrado `imitacion-v2-n2` / `imitacion-v2-n3`, los 6 trabajos del brazo B
(imitación + afinamiento PPO) fallaron al arrancar. El supervisor de entrenamiento los reintentó **144 veces** durante
~3,5 horas sin posibilidad de éxito. El brazo A (imitación) terminó bien. Producción no se afectó (seguía el
modelo anterior), pero la promoción de los modelos de v2-n2 y v2-n3 se retrasó unas 8 horas.

**Cronología** ([`runs/ppo/supervisor.log`](#evidencia-1) y `etapas.log`, locales):

| Hora | Evento |
|---|---|
| 13-nov 22:53 | Se lanzan los 12 trabajos (criterios pre-registrados en `0ddada0`). |
| 22:53–02:17 | Cada trabajo B falla a los ~15 s con el mismo error. El supervisor lo reintenta con espera creciente; al agotar los reintentos, el encadenador lo relanza con los contadores reiniciados (23:56, 00:49 y 01:41). |
| 23:05–23:11 | Los 6 trabajos A terminan y se registran en MLflow (99,8–100 % de éxito). |
| ~02:17–07:01 | El equipo se apaga o suspende (WSL arranca de nuevo a las 07:01). Se detienen el supervisor, el encadenador y el servidor de Prefect. |
| 14-nov 07:04 | Se detecta el fallo al revisar los registros. |
| 07:06 | Corrección y prueba (commit `fdbfd2a`); se relanzan solo los trabajos B (los A ya registrados se saltan). |
| 07:39 | Los 12 trabajos quedan registrados; se aplican los criterios con el test reservado. |
| 07:43 / 07:46 | El encadenador arranca el servidor de Prefect (estaba caído) y el flujo publica y promueve los modelos de v2-n2 y v2-n3; Vercel los sirve. |

**Causa raíz.**
1. **Error de código:** el brazo B pasaba `learning_rate: 0.00005` al entrenamiento como texto `learning_rate=5e-05`, y
   el script lo leía con `yaml.safe_load`. YAML 1.1 solo reconoce la notación científica con punto decimal (`5.0e-05`),
   así que devolvía la cadena `"5e-05"` y Stable-Baselines3 rechazaba la tasa de aprendizaje. La prueba previa al
   lanzamiento pasó porque llamaba a la función de entrenamiento desde Python con el número ya convertido, sin pasar
   por la línea de comandos.
2. **Falla del plan de contingencia:** los reintentos estaban pensados para errores transitorios (memoria, red, procesos
   colgados) y no distinguían un **error determinista**. Reintentar el mismo código con los mismos parámetros no podía
   funcionar, y el encadenador además reiniciaba los contadores.
3. **Factor adicional:** el apagado nocturno del equipo era el único riesgo declarado como no cubierto; ocurrió y
   detuvo también el servidor de Prefect.

**Respuesta y acciones correctivas** (commit `fdbfd2a`):
- `parse_value` en [`pipeline/common.py`](../pipeline/common.py) prueba `int` y `float` antes que YAML. Pruebas en
  [`tests/test_entrenamiento.py`](../tests/test_entrenamiento.py), incluida una que recorre todos los hiperparámetros
  de los contrastes de `configs/criterios.yaml`.
- Se probó el comando real de línea de comandos con los parámetros del brazo B antes de relanzar.
- **Detección de errores de código** en el supervisor: si un trabajo falla igual dos veces seguidas en menos de 3
  minutos, se marca `error_deterministico`, no se reintenta, y el encadenador se detiene de inmediato en vez de relanzar.
- El encadenador **arranca el servidor de Prefect** si no responde. Esta contingencia funcionó al publicar.

**Lecciones.**
- Las pruebas previas deben ejercitar la misma interfaz que usa producción: aquí, la línea de comandos.
- Un reintento sin clasificar el error esconde los fallos. Hay que separar los transitorios (reintentar) de los
  deterministas (detener y avisar).

<a id="evidencia-1"></a>**Evidencia.** Los registros del supervisor quedan en `runs/ppo/` (no versionado, por tamaño).
Las decisiones están en [`imitacion-v2-n2.md`](resultados/imitacion-v2-n2.md) e
[`imitacion-v2-n3.md`](resultados/imitacion-v2-n3.md), y las publicaciones en `docs/ejecuciones/20261114-1240*` y
`20261114-1243*`.

---

## INC-4 · Simulacro: modelo rechazado forzado a producción

**Resumen.** Simulacro planificado para probar el monitoreo y el rollback de extremo a extremo sobre la URL pública.
Un operador fuerza a producción, con `--forzar`, la red PPO de la semilla 0 de v2-n2, que los criterios habían
rechazado: completa el 99,8 % de los pedidos, pero hace 0,41 rescates por pedido (máximo 0,30).

**Detección y respuesta.** El flujo de Prefect `warehouse-rush-vigilancia` sondeó la URL, marcó **alerta** (0,50
rescates por pedido en la sonda de 50 pedidos), revirtió al recomendado anterior, publicó y verificó que la URL
volviera a "ok", **sin intervención manual**.

| Momento | Modelo servido en v2-n2 | Estado | Rescates | Utilidad por pedido (sonda) |
|---|---|---|---|---|
| Antes | `v2-n2-imitacion-v2-n2-A-s0` | ok | 0,00 | $45.893 |
| Promoción forzada | `v2-n2-ppo-v2-n2-A-s0` | **alerta** | 0,50 | $40.444 |
| Tras el rollback | `v2-n2-imitacion-v2-n2-A-s0` | ok | 0,00 | $45.893 |

Duración total: 5,2 minutos, incluidos dos despliegues de Vercel. Commits `4ac7230` (promoción forzada), `4343a74`
(rollback) y `4f3eef7` (evidencia). El modelo forzado quedó como `revertido` en el catálogo y en el Model Registry.
Detalle en [`resultados/simulacro-rollback.md`](resultados/simulacro-rollback.md).

**Lecciones.**
- La sonda detecta problemas de **costo operativo** (rescates) aunque la tasa de éxito parezca buena (98 %): vigilar
  solo el éxito no habría detectado este modelo.
- El rollback depende de que exista un recomendado anterior. Un nivel con un solo modelo promovido queda en alerta
  y requiere intervención manual; el flujo de vigilancia lo informa así.

---

## Riesgos conocidos sin mitigación automática

- **Apagado del equipo durante un entrenamiento largo:** el entrenamiento corre en el PC del equipo (WSL). Los
  checkpoints permiten reanudar con `pipeline/lanzar_ppo.sh` y `pipeline/lanzar_etapas_ppo.sh`, pero hay que
  relanzarlos a mano.
- **Estadísticas de la instancia de la API:** Vercel no guarda estado. Las peticiones, latencias y errores de
  `/api/monitoreo` se reinician en cada arranque en frío. La sonda no depende de ellas.
