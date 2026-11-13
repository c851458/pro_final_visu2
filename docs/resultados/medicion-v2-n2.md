# medicion-v2-n2: ¿Q-learning tabular aprende de forma estable el centro de distribucion (N2, recompensa en COP y SLA) con 80000 episodios, el doble que v1 porque el servicio duplica los estados?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V2-N2 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · Q-learning, 80000 episodios | 80000 | 93.8 % ± 7.2 | 80.5 ± 3.4 | 0.42 ± 0.02 | 0.00 ± 0.00 | 0.01 ± 0.01 | no: tasa_exito media 0.938 < 0.95; rescates 0.419 > 0.3 |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `medicion-v2-n2-A-s0` | 0 | 99.8 % | 84.3 | 0.438 | 490 s |
| `medicion-v2-n2-A-s1` | 1 | 85.8 % | 77.9 | 0.398 | 490 s |
| `medicion-v2-n2-A-s2` | 2 | 95.8 % | 79.3 | 0.422 | 377 s |

## Decisión

Ningun brazo cumple los requisitos: se mantiene el modelo actual.

## Test final reservado

Suite `v2-n2-reservado-1bde9e327dc2` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · Q-learning, 80000 episodios | 94.7 % ± 5.3 |

> Nota: el test reservado usa la misma semilla secreta en v2-n2 y v2-n3, así que son los mismos 500 pedidos evaluados con y sin el pasillo bloqueado.
