# imitacion-v2-n2: ¿Una red neuronal pre-entrenada imitando al planificador de reglas (DAgger) cumple los requisitos de forma estable en N2 de negocio, y el afinamiento posterior con PPO la mejora?

> **Aclaración (2026-11-14), sin cambios en la decisión:** en las redes neuronales el presupuesto de la columna "Episodios" y del motivo de la decisión son **pasos del ambiente**, no episodios. La etiqueta venía del formato de Q-learning y se corrigió en `pipeline/comparar_finalistas.py`; este informe no se regeneró porque el test reservado se evalúa una sola vez.

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V2-N2 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · imitacion DAgger | 1213155 | 99.9 % ± 0.1 | 60.1 ± 0.2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |
| **B** · imitacion DAgger + afinamiento PPO 1M pasos | 2213155 | 100.0 % ± 0.0 | 58.5 ± 0.5 | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `imitacion-v2-n2-A-s0` | 0 | 100.0 % | 60.1 | 0.000 | 722 s |
| `imitacion-v2-n2-A-s1` | 1 | 100.0 % | 60.4 | 0.000 | 699 s |
| `imitacion-v2-n2-A-s2` | 2 | 99.8 % | 60.0 | 0.000 | 701 s |
| `imitacion-v2-n2-B-s0` | 0 | 100.0 % | 58.9 | 0.034 | 1537 s |
| `imitacion-v2-n2-B-s1` | 1 | 100.0 % | 58.0 | 0.048 | 1502 s |
| `imitacion-v2-n2-B-s2` | 2 | 100.0 % | 58.6 | 0.030 | 1517 s |

## Decisión

Empate en exito (<1 pp) entre A, B: gana A por menor presupuesto (1213155 episodios).
Modelo a publicar (mejor semilla del brazo ganador): `imitacion-v2-n2-A-s0` (run `fd3db2ea5ccb462bb8836bb665d4e074`, 100.0 %).

## Test final reservado

Suite `v2-n2-reservado-1bde9e327dc2` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · imitacion DAgger | 99.9 % ± 0.1 |
| **B** · imitacion DAgger + afinamiento PPO 1M pasos | 100.0 % ± 0.0 |
