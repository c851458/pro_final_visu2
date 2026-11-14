# imitacion-v2-n3: ¿Una red neuronal pre-entrenada imitando al planificador de reglas (DAgger) cumple los requisitos de forma estable en N3 de negocio (pasillo bloqueado), y el afinamiento posterior con PPO la mejora?

> **Aclaración (2026-11-14), sin cambios en la decisión:** en las redes neuronales el presupuesto de la columna "Episodios" y del motivo de la decisión son **pasos del ambiente**, no episodios. La etiqueta venía del formato de Q-learning y se corrigió en `pipeline/comparar_finalistas.py`; este informe no se regeneró porque el test reservado se evalúa una sola vez.

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel V2-N3 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · imitacion DAgger | 1382791 | 100.0 % ± 0.0 | 81.6 ± 0.1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |
| **B** · imitacion DAgger + afinamiento PPO 1M pasos | 2382791 | 99.7 % ± 0.6 | 84.4 ± 1.5 | 0.03 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | sí |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `imitacion-v2-n3-A-s0` | 0 | 100.0 % | 81.7 | 0.000 | 881 s |
| `imitacion-v2-n3-A-s1` | 1 | 100.0 % | 81.7 | 0.000 | 896 s |
| `imitacion-v2-n3-A-s2` | 2 | 100.0 % | 81.5 | 0.000 | 861 s |
| `imitacion-v2-n3-B-s0` | 0 | 99.0 % | 86.1 | 0.002 | 1772 s |
| `imitacion-v2-n3-B-s1` | 1 | 100.0 % | 84.0 | 0.052 | 1798 s |
| `imitacion-v2-n3-B-s2` | 2 | 100.0 % | 83.2 | 0.038 | 1773 s |

## Decisión

Empate en exito (<1 pp) entre A, B: gana A por menor presupuesto (1382791 episodios).
Modelo a publicar (mejor semilla del brazo ganador): `imitacion-v2-n3-A-s2` (run `08034532ba944837a40c500cca96bc64`, 100.0 %).

## Test final reservado

Suite `v2-n3-reservado-1bde9e327dc2` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · imitacion DAgger | 100.0 % ± 0.0 |
| **B** · imitacion DAgger + afinamiento PPO 1M pasos | 99.8 % ± 0.3 |
