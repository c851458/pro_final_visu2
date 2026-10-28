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

## Versionado: datos y pipeline (DVC), experimentos (MLflow), modelos (Hugging Face)

| Qué | Dónde | Herramienta |
|---|---|---|
| Dataset de pedidos (entrenamiento y evaluación) | `data/pedidos_*.json` | DVC (git guarda el hash en `dvc.lock`) |
| Dataset de trayectorias del agente evaluado | `data/trayectorias_evaluacion.csv` | DVC |
| Pipeline `generar_datos` → `entrenar` → `evaluar` | `dvc.yaml`, `pipeline/`, `params.yaml` | DVC + git |
| Experimentos: parámetros, curvas, métricas, modelo de cada corrida, commit y versión del dataset | `mlflow.db`, `mlartifacts/` | MLflow |
| Modelos publicados, con ficha y versiones (commits y tags) | [`c851458/warehouse-rush-qlearning`](https://huggingface.co/c851458/warehouse-rush-qlearning) (privado) | Hugging Face Hub |

```bash
source .venv/bin/activate

# Datos y pipeline (DVC): solo repite las etapas cuyo código, datos o parámetros cambiaron.
# Cada ejecución queda también como run "pipeline" en MLflow.
dvc repro
dvc push                       # sube el dataset al almacenamiento de DVC
dvc pull                       # descarga el dataset de la versión actual

# Experimentos (MLflow): variantes de params.yaml, en paralelo
./experimentar.py -e alpha-alto agente.alpha=0.2 -e gamma-bajo agente.gamma=0.95 -j 2
mlflow ui --backend-store-uri sqlite:///mlflow.db     # comparar en http://localhost:5000

# Modelos (Hugging Face): publica el modelo de un run con su ficha y lo enlaza al run de MLflow
./publicar_modelo.py                         # modelo del pipeline
./publicar_modelo.py --run alpha-alto        # modelo de un experimento
./publicar_modelo.py --tag v1.1              # marca la versión

# Usar un modelo publicado
./train3d.py --demo-only --modelo hf:c851458/warehouse-rush-qlearning@v1.0
```

- La autenticación con Hugging Face usa la sesión guardada (`hf auth login`). Los repositorios se crean privados;
  usa `--publico` para que sean públicos.
- El modelo se publica como `q_table.npy` (sin pickle), con `README.md` (ficha con métricas y parámetros),
  `config.json` y `evaluacion.json`.
- El almacenamiento de DVC es local (`/home/carolo06/dvc-storage/pro_final_visu2`). Para compartirlo, cámbialo
  por uno en la nube: `dvc remote modify almacen url <url>`.
