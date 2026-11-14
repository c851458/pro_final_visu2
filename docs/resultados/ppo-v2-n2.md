# ppo-v2-n2: ¿Una politica PPO (red 128x128, 22 caracteristicas) entrenada 3 millones de pasos cumple los requisitos en el centro de distribucion N2 de forma estable en tres semillas, donde Q-learning no los cumplio?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V2-N2 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · PPO red neuronal, 3M pasos | 3000000 | 63.5 % ± 55.2 | 49.6 ± 43.0 | 0.40 ± 0.37 | 0.00 ± 0.00 | 0.00 ± 0.00 | no: tasa_exito media 0.635 < 0.95; rescates 0.396 > 0.3 |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `ppo-v2-n2-A-s0` | 0 | 99.8 % | 71.9 | 0.414 | 1349 s |
| `ppo-v2-n2-A-s1` | 1 | 90.8 % | 76.9 | 0.754 | 1364 s |
| `ppo-v2-n2-A-s2` | 2 | 0.0 % | 0.0 | 0.020 | 1389 s |

## Decisión

Ningun brazo cumple los requisitos: se mantiene el modelo actual.

## Test final reservado

Suite `v2-n2-reservado-1bde9e327dc2` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · PPO red neuronal, 3M pasos | 63.1 % ± 54.9 |
