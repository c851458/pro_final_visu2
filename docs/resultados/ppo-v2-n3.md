# ppo-v2-n3: ¿Una politica PPO (red 128x128, 22 caracteristicas) entrenada 3 millones de pasos cumple los requisitos en el centro de distribucion N3 (pasillo bloqueado) de forma estable en tres semillas?

> **Aclaración (2026-11-14), sin cambios en la decisión:** en las redes neuronales el presupuesto de la columna "Episodios" y del motivo de la decisión son **pasos del ambiente**, no episodios. La etiqueta venía del formato de Q-learning y se corrigió en `pipeline/comparar_finalistas.py`; este informe no se regeneró porque el test reservado se evalúa una sola vez.

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V2-N3 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · PPO red neuronal, 3M pasos | 3000000 | 0.0 % ± 0.0 | 0.0 ± 0.0 | 0.03 ± 0.05 | 0.00 ± 0.00 | 0.03 ± 0.05 | no: tasa_exito media 0.000 < 0.95 |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `ppo-v2-n3-A-s0` | 0 | 0.0 % | 0.0 | 0.004 | 1383 s |
| `ppo-v2-n3-A-s1` | 1 | 0.0 % | 0.0 | 0.000 | 1381 s |
| `ppo-v2-n3-A-s2` | 2 | 0.0 % | 0.0 | 0.088 | 1361 s |

## Decisión

Ningun brazo cumple los requisitos: se mantiene el modelo actual.

## Test final reservado

Suite `v2-n3-reservado-1bde9e327dc2` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · PPO red neuronal, 3M pasos | 0.0 % ± 0.0 |
