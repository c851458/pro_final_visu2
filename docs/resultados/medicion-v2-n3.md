# medicion-v2-n3: ¿Q-learning tabular aprende de forma estable el centro de distribucion con el pasillo bloqueado (N3) con 80000 episodios?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V2-N3 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · Q-learning, 80000 episodios | 80000 | 75.7 % ± 20.1 | 114.5 ± 5.2 | 0.59 ± 0.05 | 0.00 ± 0.00 | 0.01 ± 0.01 | no: tasa_exito media 0.757 < 0.95; rescates 0.593 > 0.3 |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `medicion-v2-n3-A-s0` | 0 | 73.6 % | 111.2 | 0.622 | 635 s |
| `medicion-v2-n3-A-s1` | 1 | 96.8 % | 120.5 | 0.618 | 631 s |
| `medicion-v2-n3-A-s2` | 2 | 56.8 % | 111.9 | 0.538 | 514 s |

## Decisión

Ningun brazo cumple los requisitos: se mantiene el modelo actual.

## Test final reservado

Suite `v2-n3-reservado-1bde9e327dc2` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · Q-learning, 80000 episodios | 76.9 % ± 20.4 |

> Nota: el test reservado usa la misma semilla secreta en v2-n2 y v2-n3, así que son los mismos 500 pedidos evaluados con y sin el pasillo bloqueado.
