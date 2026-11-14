# Warehouse Rush · Proyecto final de MLOps

Un robot de un centro de distribución de paquetería aprende a completar pedidos: recoger paquetes (normales,
pesados o frágiles), entregarlos en el muelle de su ruta, cargar su batería a tiempo y cumplir el plazo de los envíos
Express. El proyecto integra ambiente, agentes (Q-learning tabular y redes neuronales), visualización 3D,
experimentos con criterios fijados antes de entrenar, registro de modelos, pipeline orquestado, despliegue,
monitoreo y rollback.

- **Web (Vercel):** https://profinalvisu2.vercel.app (pestañas Simulación, Experimentos, Modelos y Monitoreo)
- **Experimentos (MLflow en DagsHub):** https://dagshub.com/c851458/pro_final_visu2.mlflow · datos DVC: https://dagshub.com/c851458/pro_final_visu2
- **Modelos (Hugging Face):** [`c851458/warehouse-rush-qlearning`](https://huggingface.co/c851458/warehouse-rush-qlearning)
- **Ficha del ambiente y niveles:** [`docs/ficha_ambiente.md`](docs/ficha_ambiente.md) · **Incidentes:** [`docs/incidentes.md`](docs/incidentes.md)

## Modelos en producción

| Nivel | Ambiente | Modelo recomendado | Tipo | Éxito | OTIF | Rescates/pedido | Utilidad/pedido |
|---|---|---|---|---|---|---|---|
| N1 | v1.0.0 | `n1-n1-base` | tabla Q | 100 % | – | 0,00 | – |
| N2 | v1.0.0 | `n2-pipeline-n2` | tabla Q | 98,6 % | – | 0,17 | – |
| N3 | v1.0.0 | `n3-contraste-n3-A-s2` | tabla Q | 99,2 % | – | 0,09 | – |
| v2-n1 | v2.0.0 | `v2-n1-v2-n1-base` | tabla Q | 100 % | 100 % | 0,00 | $18.814 |
| v2-n2 | v2.0.0 | `v2-n2-imitacion-v2-n2-A-s0` | red neuronal | 100 % | 99,2 % | 0,00 | $47.163 |
| v2-n3 | v2.0.0 | `v2-n3-imitacion-v2-n3-A-s2` | red neuronal | 100 % | 95,2 % | 0,00 | $46.008 |

Métricas sobre la suite de evaluación de cada nivel (500 pedidos que no se usaron para entrenar). La pestaña
**Monitoreo** las vuelve a medir sobre la URL pública.

**Contexto de negocio (ambiente v2.0.0):** recompensa en **pesos colombianos** (utilidad de la operación) y servicio
**Express / Estándar** con plazo (SLA). Los niveles v1 se conservan con sus modelos históricos. Modelo de costos y
fuentes en la [ficha del ambiente](docs/ficha_ambiente.md#versión-200--contexto-de-negocio-warehouserush-v2-niveles-v2-n1-v2-n2-v2-n3).

## Experimentos y decisiones

Cada contraste fija su pregunta, sus brazos, las semillas y los requisitos en
[`configs/criterios.yaml`](configs/criterios.yaml) **antes de entrenar** (la fecha del commit lo demuestra). Se mide
con 3 semillas, se decide con la regla de selección y se evalúa una sola vez en un **test reservado** con semilla
secreta. Requisitos: éxito ≥ 95 %, rescates ≤ 0,30 y frágiles rotos ≤ 0,05 por pedido.

| Contraste | Pregunta | Resultado |
|---|---|---|
| [`contraste-n3`](docs/resultados/contraste-n3.md) | Tras el incidente de N3, ¿reentrenar o continuar desde N2? | Gana reentrenar desde cero: 99,1 % ± 0,1 |
| [`medicion-v2-n2`](docs/resultados/medicion-v2-n2.md), [`-n3`](docs/resultados/medicion-v2-n3.md) | ¿Q-learning es estable en el negocio? | No: 93,8 % ± 7,2 y 75,7 % ± 20,1 |
| [`ppo-v2-n2`](docs/resultados/ppo-v2-n2.md), [`-n3`](docs/resultados/ppo-v2-n3.md) | ¿PPO desde cero lo resuelve? | No: 63,5 % ± 55,2 y 0 %; 4 de 6 redes aprendieron a quedarse quietas |
| [`imitacion-v2-n2`](docs/resultados/imitacion-v2-n2.md), [`-n3`](docs/resultados/imitacion-v2-n3.md) | ¿Imitar al planificador de reglas (DAgger), y afinar con PPO? | Sí: 99,8–100 % en las 6 semillas; promovido |

Q-learning falla en el negocio porque la tabla no observa el plazo del SLA ni la batería exacta. PPO desde cero
falla porque, con la recompensa en pesos, quedarse quieto es lo más barato al inicio. La red pre-entrenada imitando
al planificador nunca cae en esa trampa. Afinarla con PPO mejoró la utilidad en v2-n2 ($47.326–47.368 por pedido frente a
$47.163–47.204 de la imitación sola), pero empeoró el OTIF en v2-n3 (89–93 % frente a 94–95 %). Según la regla de selección, en un empate de éxito gana el de menor presupuesto.

## Arquitectura

```
                 ┌──────────────────────────── Vercel (un proyecto) ───────────────────────────┐
Docente ──HTTPS──▶  web/ (Vite + TypeScript + three.js)  ──/api──▶  api/index.py (FastAPI)       │
                 │  simulación · experimentos · modelos ·         simulación, evaluación y      │
                 │  monitoreo                                      sonda de monitoreo            │
                 └──────────────────────────────────────────────────────┬──────────────────────┘
                                                                        │ descarga los pesos y verifica el SHA-256
  PC del equipo: DVC / experimentar.py / entrenar_ppo.py ──registra──▶ MLflow (DagsHub)          │
                 │  Prefect: flujo.py (publicar) · vigilar.py (rollback)  │                      ▼
                 └──publicar_modelo.py──▶ Hugging Face Hub (pesos) + manifests/ (git) ──────────▶ catálogo
```

- **Python es la autoridad.** La API ejecuta el mismo ambiente y el mismo código de evaluación que el pipeline
  (`warehouse_rush/`). La web solo dibuja los fotogramas que recibe, y la velocidad de animación no cambia las reglas.
- **La API no guarda estado:** el control manual reenvía la lista de acciones y la API la reproduce desde el inicio.
- **El catálogo lo sirve git:** `manifests/catalogo.json` (modelos, recomendado por nivel e historial para rollback)
  y `manifests/runs.json` (exportado de MLflow). Cada despliegue de Vercel queda atado a un commit.
- **Dos tipos de modelo, un mismo contrato de servicio:** tabla Q (`q_table.npy`) o red neuronal exportada a NumPy
  (`politica.npz`). La API elige el cargador según `tipo_modelo` del manifiesto y no necesita PyTorch.

| Qué | Dónde | Herramienta |
|---|---|---|
| Código, configuración, manifiestos | este repositorio | git (commits y tags) |
| Niveles (ambiente + distribución de pedidos) | `configs/niveles.yaml` | git |
| Criterios y contrastes pre-registrados | `configs/criterios.yaml` | git |
| Dataset de pedidos de cada nivel y trayectorias evaluadas | `data/` | DVC |
| Pipeline `generar_datos` → `entrenar` → `evaluar` | `dvc.yaml`, `pipeline/`, `params.yaml` | DVC |
| Experimentos (hipótesis, parámetros, curvas, métricas, resultados por caso, modelo) | MLflow | DagsHub |
| Modelos publicados (pesos inmutables por commit) | `modelos/<model_id>/q_table.npy` o `politica.npz` | Hugging Face Hub |
| Manifiesto de cada modelo (SHA-256, contrato, ambiente, run, suite, semilla, origen) | `manifests/modelos/` | git |
| Versiones de modelos con su estado y alias `recomendado` | `warehouse-rush-<nivel>` | MLflow Model Registry (DagsHub) |
| Decisiones de los contrastes y evidencia de los flujos | `docs/resultados/`, `docs/ejecuciones/` | git |

## Estructura

```
warehouse_rush/      ambientes (env_v1.py, env_v2.py), niveles, políticas, simulación y evaluación
  ppo.py             características para redes, envoltorio de entrenamiento e inferencia NumPy
  modelos.py         carga una tabla Q (.npy) o una red (.npz) según el archivo
configs/             niveles.yaml (N1-N3, v2-n1-v2-n3) y criterios.yaml (requisitos y contrastes)
pipeline/            etapas de DVC, MLflow, exportación del catálogo y:
  entrenar_ppo.py    redes: PPO, imitación (DAgger) o imitación + PPO, con checkpoints
  supervisor_ppo.py  ejecuta los contrastes con reintentos (lanzar_ppo.sh lo desacopla de la terminal)
  etapas_ppo.py      encadena entrenar → criterios → publicar → evidencia → verificar (lanzar_etapas_ppo.sh)
  comparar_finalistas.py  aplica los criterios a un contraste y evalúa el test reservado
  flujo.py           flujo de Prefect para publicar y promover
  vigilar.py         flujo de Prefect de vigilancia: sonda de la URL y rollback
  simulacro_rollback.py  simulacro de extremo a extremo
api/index.py         API FastAPI (Vercel Function), incluido /api/monitoreo
web/                 frontend Vite + TypeScript + three.js
manifests/           catálogo y manifiestos de modelos (los sirve la API)
tests/               ambiente, datos, políticas, entrenamiento y API
docs/                ficha del ambiente, incidentes, resultados de los contrastes y evidencia de los flujos
experimentar.py      experimentos de Q-learning registrados en MLflow
publicar_modelo.py   publicar en Hugging Face, promover, rechazar y hacer rollback
play3d.py, train3d.py  visor local en vivo (entrenamiento y control manual)
```

## Instalación

Probado con Python 3.10.12 en local (Vercel usa Python 3.12) y Node 24.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # versiones exactas de todo: requirements-lock.txt
cd web && npm ci && cd ..                         # versiones del frontend: web/package-lock.json
cp .env.example .env                              # credenciales de DagsHub (no se sube a git)
```

`requirements.txt` contiene solo lo que necesita la API (es lo que instala Vercel). `requirements-dev.txt` agrega
MLflow, DVC, pygame, Hugging Face, pytest, Prefect y, para entrenar redes, PyTorch (CPU) y Stable-Baselines3.

## Ejecutar en local

```bash
.venv/bin/uvicorn api.index:app --port 8800      # API
cd web && npm run dev                            # web en http://localhost:5173 (redirige /api a la API)
.venv/bin/python -m pytest                       # pruebas
```

Visor local en vivo, sin la web desplegada: `./train3d.py` (ver el entrenamiento) y `./play3d.py` (control manual).

## Pipeline, experimentos y modelos

```bash
dvc repro                       # datasets de todos los niveles + entrenar y evaluar el nivel de params.yaml
dvc push                        # sube el dataset al almacenamiento de DVC (DagsHub)

./experimentar.py --baselines   # referencias aleatoria y de regla en cada nivel
./experimentar.py -e alpha-alto agente.alpha=0.2 --hipotesis "..."   # un experimento con hipótesis
./experimentar.py --contraste medicion-v2-n2                         # brazos x semillas de un contraste

./publicar_modelo.py --run pipeline-n2 --tag v1.1   # sube los pesos y escribe el manifiesto (estado: candidato)
./publicar_modelo.py --recomendar <model_id>        # promueve si cumple configs/criterios.yaml
./publicar_modelo.py --rollback v2-n2               # vuelve al recomendado anterior del nivel
.venv/bin/python pipeline/comparar_finalistas.py <contraste> --test-reservado   # decide un contraste
.venv/bin/python pipeline/exportar_catalogo.py      # exporta los runs de MLflow a manifests/runs.json
```

Cada modelo publicado es una versión del **Model Registry de MLflow** (`warehouse-rush-<nivel>`), con tags de
estado, SHA-256 y URI de Hugging Face. El alias `recomendado` de cada nivel es el modelo que sirve la web.
`./publicar_modelo.py --sincronizar-registro` alinea el registro con el catálogo.

Publicar no promueve: el recomendado de cada nivel se cambia aparte, solo si cumple los requisitos y no empeora al
actual, y el historial de recomendados permite el rollback. Publicar dos veces el mismo artefacto no crea otra
versión, porque se identifica por su SHA-256.

### Redes neuronales: entrenamiento tolerante a fallos

```bash
pipeline/lanzar_ppo.sh imitacion-v2-n2 imitacion-v2-n3          # entrena los brazos x semillas en segundo plano
pipeline/lanzar_etapas_ppo.sh imitacion-v2-n2 imitacion-v2-n3   # y encadena criterios, publicación y verificación
cat runs/ppo/estado.json; tail -f runs/ppo/supervisor.log runs/ppo/etapas.log
```

`--algoritmo` de `entrenar_ppo.py` admite `ppo`, `imitacion` (DAgger con la política de reglas como maestro) e
`imitacion+ppo`. Cada trabajo vive en `runs/ppo/<trabajo>/` (no versionado) y el sistema cubre estos fallos:

| Fallo | Respuesta |
|---|---|
| El proceso muere (error, falta de memoria, kill) | se relanza y continúa desde su último checkpoint |
| El proceso se cuelga | sin latido en 20 min: se mata y se reanuda |
| Poca memoria | no lanza más trabajos y baja el paralelismo |
| MLflow no responde | el resultado queda en `resultado/` y solo se reintenta el registro, sin duplicar runs |
| El supervisor se cae | se relanza, adopta los procesos vivos y salta lo registrado |
| Error de código (mismo error dos veces en menos de 3 min) | no se reintenta: se detiene y se informa ([INC-3](docs/incidentes.md#inc-3--el-brazo-b-de-imitación-no-arranca-y-se-reintenta-toda-la-noche)) |
| Servidor de Prefect caído al publicar | el encadenador lo arranca |

Si el equipo se apaga, basta con volver a ejecutar los dos lanzadores: continúan desde los checkpoints.

## Orquestación con Prefect

[`pipeline/flujo.py`](pipeline/flujo.py) encadena la secuencia mínima del enunciado como tareas de Prefect:
**1** validar el perfil, el esquema, las rutas y la compatibilidad · **2** pruebas · **3** entrenar con DVC o cargar
un checkpoint compatible · **4** evaluar la suite y comprobar que coincide con MLflow · **5** registrar el modelo con
su hash · **6** aplicar los criterios · **7** exportar el catálogo (con `--publicar`: commit y push) · **8** verificar
la versión servida en Vercel, con rollback si falla después de una promoción. Cada ejecución deja su estado en
Prefect y su evidencia en `docs/ejecuciones/`.

```bash
# servidor de Prefect (entorno aparte: Prefect 3.8.8 no es compatible con FastAPI 0.162 de la API)
python3 -m venv .venv-prefect && .venv-prefect/bin/pip install -r requirements-prefect-server.txt
.venv-prefect/bin/prefect server start                 # historial y UI en http://127.0.0.1:4200

.venv/bin/python pipeline/flujo.py --modo checkpoint --run pipeline-n2            # sin publicar
.venv/bin/python pipeline/flujo.py --modo pipeline --promover --publicar          # ciclo completo
.venv/bin/python pipeline/flujo.py --modo checkpoint --run <run> --nivel n3 --promover --publicar
```

## Monitoreo y rollback

**`GET /api/monitoreo?casos=50`** ejecuta, en la API desplegada, el modelo recomendado de cada nivel sobre los
primeros pedidos de su suite de evaluación y marca cada nivel como:
- **ok:** cumple los requisitos de promoción;
- **alerta:** no los cumple, o su éxito cae más de 10 puntos frente a lo registrado al publicarlo (deriva);
- **error:** error de inferencia, por ejemplo un modelo inaccesible o con un SHA-256 distinto;
- **sin modelo:** el nivel no tiene un recomendado.

También informa las peticiones, latencias (p50 y p95) y errores recientes de la instancia. Esos datos viven en
memoria y se reinician en cada arranque en frío de Vercel. La pestaña **Monitoreo** de la web muestra todo esto junto
con el historial de promociones y rollback de cada nivel.

```bash
.venv/bin/python pipeline/vigilar.py                    # sondea la URL y revierte los niveles en alerta
.venv/bin/python pipeline/vigilar.py --solo-observar    # solo informa
.venv/bin/python pipeline/simulacro_rollback.py         # simulacro: fuerza un modelo rechazado y lo revierte
```

La vigilancia revierte al recomendado anterior, publica, espera el despliegue y vuelve a sondear hasta que el nivel
queda en "ok". El simulacro del 14 de noviembre forzó a producción una red con 0,41 rescates por pedido; la vigilancia
la detectó y la revirtió en 5,2 minutos, sin intervención manual
([informe](docs/resultados/simulacro-rollback.md)).

## Despliegue en Vercel

El proyecto de Vercel está conectado a GitHub: **cada push a `main` despliega a producción** y cada despliegue
queda atado a su commit (la API lo informa en `/api/salud`). Es un solo proyecto con la **raíz del repositorio** como
Root Directory: `vercel.json` compila `web/` (salida `web/dist`) y publica `api/index.py` como función de Python en
`/api/*`.

| Variable (Vercel) | Uso |
|---|---|
| `HF_TOKEN` | Solo si el repositorio de Hugging Face es privado: token de **solo lectura** para descargar los pesos. |
| `CORS_ORIGINS` | Opcional; por defecto `*` (la API es pública y de solo lectura). |

La web consume la API en la misma URL. Para usar otra API, se compila con `VITE_API_BASE_URL` (ver `web/.env.example`).
