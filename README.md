# Warehouse Rush · Proyecto final de MLOps

Un robot de bodega aprende con **Q-learning tabular** a completar pedidos: recoger paquetes (normales, pesados o
frágiles), entregarlos en la zona de su color y cargar su batería a tiempo. El proyecto integra ambiente, agente,
visualización 3D, experimentos, modelos, pipeline y despliegue.

- **Ficha del ambiente y niveles N1/N2/N3:** [`docs/ficha_ambiente.md`](docs/ficha_ambiente.md)
- **Web (Vercel):** https://profinalvisu2.vercel.app
- **Experimentos (MLflow en DagsHub):** https://dagshub.com/c851458/pro_final_visu2.mlflow · datos DVC: https://dagshub.com/c851458/pro_final_visu2
- **Modelos (Hugging Face):** [`c851458/warehouse-rush-qlearning`](https://huggingface.co/c851458/warehouse-rush-qlearning)

## Arquitectura

```
                 ┌──────────────────────────── Vercel (un proyecto) ───────────────────────────┐
Docente ──HTTPS──▶  web/ (Vite + TypeScript + three.js)  ──/api──▶  api/index.py (FastAPI)       │
                 │  catálogo · comparación · escena 3D            simulación + evaluación       │
                 └──────────────────────────────────────────────────────┬──────────────────────┘
                                                                        │ descarga los pesos y verifica el SHA-256
  PC del equipo: pipeline DVC / experimentar.py ──registra──▶ MLflow (DagsHub)                   │
                 │                                                       │                      ▼
                 └──publicar_modelo.py──▶ Hugging Face Hub (pesos) + manifests/ (git) ──────────▶ catálogo
```

- **Python es la autoridad.** La API ejecuta el mismo ambiente y el mismo código de evaluación que el pipeline
  (`warehouse_rush/`). La web solo dibuja los fotogramas que recibe, y la velocidad de animación no cambia las reglas.
- **La API no guarda estado:** el control manual reenvía la lista de acciones y la API la reproduce desde el inicio.
- **El catálogo lo sirve git:** `manifests/catalogo.json` (modelos y recomendado) y `manifests/runs.json` (exportado de
  MLflow). Cada despliegue de Vercel queda atado a un commit.

| Qué | Dónde | Herramienta |
|---|---|---|
| Código, configuración, manifiestos | este repositorio | git (commits y tags) |
| Niveles (ambiente + distribución de pedidos) | `configs/niveles.yaml` | git |
| Dataset de pedidos de cada nivel y trayectorias evaluadas | `data/` | DVC |
| Pipeline `generar_datos` → `entrenar` → `evaluar` | `dvc.yaml`, `pipeline/`, `params.yaml` | DVC |
| Experimentos (hipótesis, parámetros, curvas, métricas, resultados por caso, modelo) | MLflow | DagsHub |
| Modelos publicados (pesos inmutables por commit) | `modelos/<model_id>/q_table.npy` | Hugging Face Hub |
| Manifiesto de cada modelo (SHA-256, contrato, ambiente, run, suite, semilla, origen) | `manifests/modelos/` | git |

## Estructura

```
warehouse_rush/      ambiente (env_v1.py), niveles, políticas (aleatoria, regla, Q), simulación y evaluación
configs/niveles.yaml N1, N2 y N3: ambiente y distribución de pedidos
pipeline/            etapas de DVC, entrenamiento, registro en MLflow y exportación del catálogo
api/index.py         API FastAPI (Vercel Function)
web/                 frontend Vite + TypeScript + three.js
manifests/           catálogo y manifiestos de modelos (los sirve la API)
tests/               pruebas de reglas, observaciones, terminaciones, datos, políticas y API
docs/                ficha del ambiente y decisiones
experimentar.py      experimentos registrados en MLflow
publicar_modelo.py   publicar en Hugging Face, promover y hacer rollback
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

`requirements.txt` contiene solo lo que necesita la API (es lo que instala Vercel), y `requirements-dev.txt`
agrega MLflow, DVC, pygame, Hugging Face y pytest.

## Ejecutar en local

```bash
.venv/bin/uvicorn api.index:app --port 8800      # API
cd web && npm run dev                            # web en http://localhost:5173 (redirige /api a la API)
.venv/bin/python -m pytest                       # pruebas
```

Visor local en vivo, sin la web desplegada: `./train3d.py` (ver el entrenamiento) y `./play3d.py` (control manual).

## Pipeline, experimentos y modelos

```bash
dvc repro                       # datasets de N1-N3 + entrenar y evaluar el nivel de params.yaml (run "pipeline-n2")
dvc push                        # sube el dataset al almacenamiento de DVC

./experimentar.py --baselines   # referencias aleatoria y de regla en cada nivel
./experimentar.py -e alpha-alto agente.alpha=0.2 --hipotesis "..."   # un experimento con hipótesis
./experimentar.py -e n3-recuperacion nivel=n3 --desde pipeline-n2   # continúa desde otro modelo

./publicar_modelo.py --run pipeline-n2 --tag v1.1   # sube los pesos y escribe el manifiesto (estado: candidato)
./publicar_modelo.py --recomendar <model_id>        # promueve (lo que sirve la web por defecto)
./publicar_modelo.py --rollback                     # vuelve al recomendado anterior
.venv/bin/python pipeline/exportar_catalogo.py      # exporta los runs de MLflow a manifests/runs.json
```

Publicar no promueve: el recomendado se cambia aparte y el historial de recomendados permite el rollback.
Publicar dos veces el mismo artefacto no crea otra versión, porque se identifica por su SHA-256.

## Despliegue en Vercel

Un solo proyecto con la **raíz del repositorio** como Root Directory: `vercel.json` compila `web/` (salida
`web/dist`) y publica `api/index.py` como función de Python en `/api/*`.

| Variable (Vercel) | Uso |
|---|---|
| `HF_TOKEN` | Solo si el repositorio de Hugging Face es privado: token de **solo lectura** para descargar los pesos. |
| `CORS_ORIGINS` | Opcional; por defecto `*` (la API es pública y de solo lectura). |

La web consume la API en la misma URL. Para usar otra API, se compila con `VITE_API_BASE_URL` (ver `web/.env.example`).
