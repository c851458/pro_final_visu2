#!/usr/bin/env bash
# Lanza el supervisor PPO desacoplado de la terminal (sobrevive a cerrar VS Code o la sesion) y lo relanza si se
# cae. Es idempotente: si ya hay un supervisor vivo no lanza otro; relanzarlo tras un reinicio de la maquina
# continua desde los checkpoints.
#
# Uso: pipeline/lanzar_ppo.sh ppo-v2-n2 ppo-v2-n3 [opciones del supervisor]
#      Estado: cat runs/ppo/estado.json    Registro: tail -f runs/ppo/supervisor.log
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/ppo
PIDFILE=runs/ppo/supervisor.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Ya hay un supervisor activo (pid $(cat "$PIDFILE")); ver runs/ppo/estado.json"
  exit 0
fi

PY=.venv/bin/python
setsid nohup bash -c '
  echo $$ > '"$PIDFILE"'
  for intento in $(seq 1 30); do
    '"$PY"' -u pipeline/supervisor_ppo.py "$@"
    code=$?
    case $code in
      0) echo "$(date "+%F %T") supervisor terminado" >> runs/ppo/supervisor.log; break ;;
      2|4) echo "$(date "+%F %T") supervisor termino con codigo $code (requiere revision)" >> runs/ppo/supervisor.log; break ;;
      *) echo "$(date "+%F %T") supervisor se cayo (codigo $code); se relanza en 60 s (relanzamiento $intento)" >> runs/ppo/supervisor.log
         sleep 60 ;;
    esac
  done
  rm -f '"$PIDFILE"'
' _ "$@" > runs/ppo/lanzador.log 2>&1 < /dev/null &
sleep 1
echo "Supervisor lanzado (pid $(cat "$PIDFILE" 2>/dev/null)). Estado: runs/ppo/estado.json"
