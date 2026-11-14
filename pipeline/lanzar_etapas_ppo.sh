#!/usr/bin/env bash
# Lanza, desacoplado de la terminal, el encadenador de etapas posteriores al entrenamiento PPO
# (espera al supervisor -> criterios -> publicar con Prefect si hay ganador -> subir evidencia -> verificar Vercel).
# Si termina con error lo relanza (las etapas hechas se saltan). Idempotente: no lanza un segundo encadenador.
#
# Uso: pipeline/lanzar_etapas_ppo.sh ppo-v2-n2 ppo-v2-n3
#      Estado: cat runs/ppo/etapas.json    Registro: tail -f runs/ppo/etapas.log
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/ppo
PIDFILE=runs/ppo/etapas.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Ya hay un encadenador activo (pid $(cat "$PIDFILE")); ver runs/ppo/etapas.json"
  exit 0
fi

setsid nohup bash -c '
  echo $$ > '"$PIDFILE"'
  for intento in 1 2 3 4; do
    .venv/bin/python -u pipeline/etapas_ppo.py "$@" && break
    echo "$(date "+%F %T") encadenador termino con error; se relanza en 300 s (relanzamiento $intento)" >> runs/ppo/etapas.log
    sleep 300
  done
  rm -f '"$PIDFILE"'
' _ "$@" > runs/ppo/lanzador-etapas.log 2>&1 < /dev/null &
sleep 1
echo "Encadenador lanzado (pid $(cat "$PIDFILE" 2>/dev/null)). Estado: runs/ppo/etapas.json"
