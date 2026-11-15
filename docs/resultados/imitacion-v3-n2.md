# imitacion-v3-n2: ¿La receta de imitación de v2 cumple los requisitos de forma estable en el centro de dos pisos con órdenes de 3 paquetes, batería y hora de corte?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V3-N2 (500 pedidos).

| Brazo | Presupuesto | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · imitacion DAgger (receta v2) | 1,761,902 pasos del ambiente | 100.0 % ± 0.0 | 136.5 ± 0.7 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |

## Indicadores de negocio

Se informan para el análisis; no forman parte de los requisitos ni de la regla de selección.

| Brazo | OTIF | Express a tiempo | Utilidad / orden | Camiones perdidos / orden | Paquetes en su camión |
|---|---|---|---|---|---|
| **A** · imitacion DAgger (receta v2) | 90.7 % ± 0.8 | 93.1 % ± 0.3 | $44,529 ± 48 | 0.053 ± 0.004 | 98.2 % ± 0.1 |
| Planificador de reglas (maestro, sin aprendizaje) | 94.6 % | 96.4 % | $45,116 | 0.032 | 98.9 % |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `imitacion-v3-n2-A-s0` | 0 | 100.0 % | 137.2 | 0.000 | 918 s |
| `imitacion-v3-n2-A-s1` | 1 | 100.0 % | 136.4 | 0.000 | 920 s |
| `imitacion-v3-n2-A-s2` | 2 | 100.0 % | 135.9 | 0.004 | 931 s |

## Decisión

Gana A por mayor exito medio (100.0%).
Modelo a publicar (mejor semilla del brazo ganador): `imitacion-v3-n2-A-s2` (run `1b934b7df5b24ef08022bb563c321c77`, 100.0 %).

## Test final reservado

Suite `v3-n2-reservado-4ad2532c269e` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · imitacion DAgger (receta v2) | 100.0 % ± 0.0 |
