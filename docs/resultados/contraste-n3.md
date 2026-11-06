# contraste-n3: Tras el incidente, ¿conviene reentrenar desde cero en N3 o continuar desde el modelo de N2 con exploracion completa, que usa menos episodios?

Criterios fijados antes de entrenar: [`configs/criterios.yaml`](../../configs/criterios.yaml). Media ± desviación estándar de 3 semillas de entrenamiento, en la suite de evaluación del nivel N3 (500 pedidos).

| Brazo | Episodios | Éxito | Pasos (exitosos) | Rescates | Frágiles rotos | Choques | Cumple |
|---|---|---|---|---|---|---|---|
| **A** · desde cero | 40000 | 99.1 % ± 0.1 | 90.7 ± 0.5 | 0.08 ± 0.02 | 0.00 ± 0.00 | 0.03 ± 0.04 | sí |
| **B** · continuar desde N2 con exploracion completa | 25000 | 17.5 % ± 3.8 | 112.1 ± 2.1 | 0.30 ± 0.04 | 0.00 ± 0.00 | 3.39 ± 0.80 | no: tasa_exito media 0.175 < 0.95 |

## Por semilla

| Run | Semilla | Éxito | Pasos (exitosos) | Rescates | Tiempo |
|---|---|---|---|---|---|
| `contraste-n3-A-s0` | 0 | 99.0 % | 90.2 | 0.098 | 422 s |
| `contraste-n3-A-s1` | 1 | 99.2 % | 91.2 | 0.068 | 420 s |
| `contraste-n3-A-s2` | 2 | 99.2 % | 90.6 | 0.088 | 421 s |
| `contraste-n3-B-s0` | 0 | 17.4 % | 112.8 | 0.330 | 379 s |
| `contraste-n3-B-s1` | 1 | 21.4 % | 109.7 | 0.312 | 373 s |
| `contraste-n3-B-s2` | 2 | 13.8 % | 113.7 | 0.258 | 378 s |

## Decisión

Gana A por mayor exito medio (99.1%).
Modelo a publicar (mejor semilla del brazo ganador): `contraste-n3-A-s2` (run `98cdb38b62fb42259f5cdfa30e9b223c`, 99.2 %).

## Test final reservado

Suite `n3-reservado-72b4b47801c3` (500 pedidos con semilla secreta, ninguno repetido de entrenamiento ni de la suite pública). Se evaluó una sola vez, después de decidir; no se usó para ajustar nada.

| Brazo | Éxito en test reservado |
|---|---|
| **A** · desde cero | 98.7 % ± 0.5 |
| **B** · continuar desde N2 con exploracion completa | 19.2 % ± 6.2 |
