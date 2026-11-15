# imitacion-v3-n3: ¿La receta de imitación de v2 cumple los requisitos de forma estable en el centro de dos pisos con el corredor inferior bloqueado en ambos pisos?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V3-N3 (500 pedidos).

| Brazo | Presupuesto | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · imitacion DAgger (receta v2) | 2,026,833 pasos del ambiente | 100.0 % ± 0.0 | 151.0 ± 0.6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |

## Indicadores de negocio

Se informan para el análisis; no forman parte de los requisitos ni de la regla de selección.

| Brazo | OTIF | Express a tiempo | Utilidad / orden | Camiones perdidos / orden | Paquetes en su camión |
|---|---|---|---|---|---|
| **A** · imitacion DAgger (receta v2) | 85.0 % ± 0.6 | 87.8 % ± 0.6 | $43,496 ± 83 | 0.116 ± 0.006 | 96.1 % ± 0.2 |
| Planificador de reglas (maestro, sin aprendizaje) | 87.6 % | 89.8 % | $43,996 | 0.086 | 97.1 % |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `imitacion-v3-n3-A-s0` | 0 | 100.0 % | 150.6 | 0.000 | 936 s |
| `imitacion-v3-n3-A-s1` | 1 | 100.0 % | 150.7 | 0.000 | 926 s |
| `imitacion-v3-n3-A-s2` | 2 | 100.0 % | 151.7 | 0.000 | 899 s |

## Decisión

Gana A por mayor exito medio (100.0%).
Modelo a publicar (mejor semilla del brazo ganador): `imitacion-v3-n3-A-s0` (run `d12c72f1b96b4fcd89291ebdd8a02622`, 100.0 %).

## Test final reservado

Suite `v3-n3-reservado-4ad2532c269e` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · imitacion DAgger (receta v2) | 100.0 % ± 0.0 |
