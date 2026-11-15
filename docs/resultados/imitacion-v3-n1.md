# imitacion-v3-n1: ¿La receta de imitación de v2 (DAgger, 8 iteraciones de 1500 pedidos, red 128x128) aprende de forma estable a despachar un paquete en el centro de dos pisos con hora de corte?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V3-N1 (500 pedidos).

| Brazo | Presupuesto | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · imitacion DAgger (receta v2) | 765,439 pasos del ambiente | 100.0 % ± 0.0 | 41.5 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |

## Indicadores de negocio

Se informan para el análisis; no forman parte de los requisitos ni de la regla de selección.

| Brazo | OTIF | Express a tiempo | Utilidad / orden | Camiones perdidos / orden | Paquetes en su camión |
|---|---|---|---|---|---|
| **A** · imitacion DAgger (receta v2) | 97.5 % ± 0.0 | 97.9 % ± 0.0 | $17,568 ± 0 | 0.025 ± 0.000 | 97.5 % ± 0.0 |
| Planificador de reglas (maestro, sin aprendizaje) | 97.5 % | 97.9 % | $17,568 | 0.025 | 97.5 % |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `imitacion-v3-n1-A-s0` | 0 | 100.0 % | 41.5 | 0.000 | 387 s |
| `imitacion-v3-n1-A-s1` | 1 | 100.0 % | 41.5 | 0.000 | 378 s |
| `imitacion-v3-n1-A-s2` | 2 | 100.0 % | 41.5 | 0.000 | 345 s |

## Decisión

Gana A por mayor exito medio (100.0%).
Modelo a publicar (mejor semilla del brazo ganador): `imitacion-v3-n1-A-s0` (run `60f1f88c20e04794832a797c1db00573`, 100.0 %).

## Test final reservado

Suite `v3-n1-reservado-0862b6234665` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · imitacion DAgger (receta v2) | 100.0 % ± 0.0 |
