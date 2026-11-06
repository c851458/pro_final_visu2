// Cliente de la API. VITE_API_BASE_URL vacío = mismo dominio (Vercel sirve web y API juntas).
import type { Catalogo, Episodio, Evaluacion, Niveles, Salud } from './tipos';

const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, body === undefined ? undefined : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError('No se pudo conectar con la API de simulación.', 0);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail ?? data);
    } catch { /* respuesta sin JSON */ }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  salud: () => request<Salud>('/api/salud'),
  niveles: () => request<Niveles>('/api/niveles'),
  catalogo: () => request<Catalogo>('/api/catalogo'),
  episodio: (body: { nivel: string; politica: string; model_id?: string | null; caso: number; acciones?: number[] }) =>
    request<Episodio>('/api/episodio', body),
  evaluar: (body: { nivel: string; politica: string; model_id?: string | null; casos: number }) =>
    request<Evaluacion>('/api/evaluar', body),
};
