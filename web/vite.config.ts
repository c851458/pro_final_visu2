import { defineConfig } from 'vite';

// En desarrollo, /api se redirige a la API local (uvicorn api.index:app --port 8800).
const proxy = { '/api': process.env.VITE_DEV_API ?? 'http://localhost:8800' };

export default defineConfig({
  server: { proxy },
  preview: { proxy },
  build: { chunkSizeWarningLimit: 900 }, // three.js ocupa la mayor parte del paquete
});
