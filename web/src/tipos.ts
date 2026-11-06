// Tipos de los datos que devuelve la API de Python (la autoridad de la simulación).

export interface Paquete {
  shelf: number;
  zone: number;
  type: number;
  status: 'pendiente' | 'cargando' | 'entregado';
  broken: number;
}

export interface Fotograma {
  pos: [number, number];
  battery: number;
  carrying: boolean;
  current: number;
  order: Paquete[];
  steps: number;
  total: number;
  event: string;
  success: boolean;
  rescues: number;
  action: number | null;
  reward: number;
  terminated: boolean;
  truncated: boolean;
}

export interface Mapa {
  layout: string[];
  shelf_cells: number[][];
  zone_cells: number[][][];
  charger: number[];
  start: number[];
  bloqueos: number[][];
}

export interface Ambiente {
  nivel: string;
  nombre: string;
  environment_version: string;
  familia_modelo: string;
  evaluation_suite_id: string;
}

export interface Nivel extends Ambiente {
  decision: string;
  entorno: { bateria: boolean; bateria_max: number; max_pasos: number; bloqueos: number[][] };
  datos: Record<string, unknown>;
  casos_evaluacion: number;
  mapa: Mapa;
}

export interface Niveles {
  niveles: Nivel[];
  acciones: string[];
  tipos: string[];
  zonas: string[];
}

export interface ModeloInfo {
  model_id: string;
  model_sha256: string;
  nivel_entrenamiento: string;
  environment_version: string;
  run_name: string;
  hf_commit: string;
  git_commit: string;
}

export interface Resultado {
  exito: boolean;
  causa: string;
  pasos: number;
  recompensa: number;
  entregados: number;
  paquetes: number;
  rescates: number;
  rotos: number;
  choques: number;
  cargas: number;
  latencia_media_ms: number;
  caso?: number;
}

export interface Episodio {
  modo: string;
  ambiente: Ambiente;
  modelo: ModeloInfo | null;
  caso: number;
  resultado?: Resultado;
  fotogramas: Fotograma[];
}

export interface Metricas {
  tasa_exito: number;
  pasos_medios: number;
  pasos_medios_exitosos: number;
  recompensa_media: number;
  rescates_medios: number;
  fragiles_rotos_medios: number;
  cargas_medias: number;
  choques_medios: number;
  paquetes_entregados_medios: number;
  latencia_media_ms: number;
  pedidos_evaluados: number;
  fallos_por_causa: Record<string, number>;
}

export interface Evaluacion {
  modo: string;
  ambiente: Ambiente;
  modelo: ModeloInfo | null;
  politica: string;
  metricas: Metricas;
  por_caso: Resultado[];
  suite_completa: boolean;
  tiempo_servidor_s: number;
}

export interface ModeloCatalogo {
  model_id: string;
  run_id: string;
  run_name: string;
  level_id: string;
  familia_modelo: string;
  environment_version: string;
  model_sha256: string;
  hf_commit: string;
  hf_tag: string | null;
  git_commit: string;
  publicado: string;
  hipotesis: string;
  tasa_exito: number;
  estado: string;
  motivo?: string;
}

export interface Run {
  run_id: string;
  nombre: string;
  estado: string;
  inicio: number;
  duracion_s: number | null;
  tipo: string;
  origen: string;
  nivel: string;
  environment_version: string;
  familia_modelo: string;
  hipotesis: string;
  cambios: string;
  semilla: string;
  episodios: string;
  git_commit: string;
  evaluation_suite_id: string;
  estado_modelo: string;
  parent_run_id: string | null;
  model_id: string | null;
  error: string | null;
  metricas: Record<string, number>;
  tiempo_entrenamiento_s: number | null;
  curvas: Record<string, [number, number][]>;
  mlflow_url: string;
}

export interface Catalogo {
  catalogo: { recomendado: string | null; historial_recomendados?: string[]; modelos: ModeloCatalogo[]; actualizado?: string };
  runs: { tracking?: string; runs: Run[] };
}

export interface Salud {
  estado: string;
  api_version: string;
  environment_version: string;
  git_commit: string;
  despliegue: string;
  modelo_recomendado: string | null;
}
