# pro_final_visu2
## Warehouse Rush 3D

Un robot de bodega aprende con Q-learning a completar pedidos.
La visualización 3D usa three.js en el navegador; Python ejecuta el entorno y el Q-learning.

### Reglas (entorno `WarehouseRush-v1`)

- **Pedidos de varios paquetes:** cada episodio es un pedido de 3 paquetes que se entregan en orden, uno a la vez.
- **Varias zonas de entrega:** cada paquete lleva una etiqueta de color (zona A roja, B azul, C morada)
  y solo se puede entregar en su zona.
- **Tipos de paquete:**
  - normal: sin efectos extra;
  - pesado: mientras se lleva, cada paso gasta el doble de batería;
  - frágil: si el robot choca mientras lo lleva, se rompe y hay que volver a buscarlo a la estantería.
- **Batería y estación de carga:** moverse gasta batería, y la acción *cargar* en la estación ⚡ la llena.
  Si la batería se agota, una grúa rescata al robot y lo lleva a la estación (penalización de -25).

| Evento | Recompensa |
|---|---|
| Cada paso | -0.1 (-0.2 con paquete pesado) |
| Chocar | -1 |
| Romper un frágil | -10 adicional |
| Acción inválida (recoger, entregar o cargar donde no corresponde) | -1 |
| Recoger | +5 |
| Entregar en la zona correcta | +15 |
| Completar el pedido | +20 adicional |
| Rescate por batería agotada | -25 |

El agente observa `[fila, columna, nivel de batería (4 niveles), lleva paquete, tipo del paquete que lleva, objetivo]`,
donde *objetivo* es la estantería del paquete actual, o su zona de destino si ya lo lleva.
Con 40 000 episodios completa el 100 % de los pedidos en modo demo, con unos 65 pasos de media.

### Ejecutar

```bash
source .venv/bin/activate

./train3d.py                                   # entrena (40 000 episodios, ~3 min) y anima 1 de cada 2000
./train3d.py --demo-only --demo-episodes 5     # solo demostración con la tabla Q guardada
./train3d.py --paquetes 2 --sin-bateria        # versión más sencilla
./play3d.py                                    # control manual con el teclado
```

Cada script abre el navegador automáticamente (`--no-browser` para no hacerlo) en
`http://localhost:8765` (automático) o `http://localhost:8766` (manual).
Si el puerto está ocupado, se usa automáticamente el siguiente libre (la terminal muestra la dirección).
La página necesita internet para cargar three.js desde el CDN.

- Cámara: arrastrar para girar, rueda para zoom, clic derecho para desplazar; teclas 1-4 cambian de vista.
- Manual: flechas/WASD mover (relativo a la cámara), R recoger, E entregar, C cargar batería, N nuevo pedido.

Las versiones anteriores, con el entorno simple `WarehouseRush-v0` (un paquete, sin batería), siguen disponibles:
`train.py` (terminal o pygame) y `play.py` (pygame).

## Versionado con DVC (datos, modelo, pipeline y experimentos)

| Qué | Dónde | Cómo se versiona |
|---|---|---|
| Dataset de pedidos (entrenamiento y evaluación) | `data/pedidos_*.json` | DVC (git guarda el hash en `dvc.lock`) |
| Dataset de trayectorias del agente evaluado | `data/trayectorias_evaluacion.csv` | DVC |
| Modelo (tabla Q) | `models/q_table_v1.pkl` | DVC |
| Pipeline | `dvc.yaml` + `pipeline/*.py` | git |
| Parámetros | `params.yaml` | git |
| Métricas y curva de aprendizaje | `metrics/evaluacion.json`, `metrics/entrenamiento.csv` | git |
| Experimentos | `dvc exp` | refs de git + caché de DVC |

Pipeline: `generar_datos` → `entrenar` → `evaluar`. DVC solo repite las etapas cuyas dependencias
(código, datos o parámetros) cambiaron.

```bash
source .venv/bin/activate

dvc repro                                  # ejecuta el pipeline (solo lo que cambió)
dvc push                                   # sube datos y modelo al almacenamiento de DVC
dvc pull                                   # descarga los datos y el modelo de la versión actual

dvc exp run -S agente.alpha=0.2            # experimento cambiando un parámetro
dvc exp run -S entorno.bateria_max=60 -n bateria-60   # experimento con nombre
dvc exp show                               # tabla comparando parámetros y métricas
dvc exp diff                               # diferencias con la versión base
dvc plots diff <exp1> <exp2>               # curvas de aprendizaje comparadas (HTML)
dvc exp apply <exp>                        # trae al workspace los archivos de un experimento
dvc exp branch <exp> <rama>                # convierte un experimento en una rama de git

./train3d.py --demo-only --modelo models/q_table_v1.pkl   # ver en 3D el modelo versionado
```

El almacenamiento de DVC está configurado en `/home/carolo06/dvc-storage/pro_final_visu2` (local).
Para compartirlo con otras personas, cámbialo por uno en la nube, por ejemplo Google Drive, S3 o DagsHub:
`dvc remote modify almacen url <url>`.
