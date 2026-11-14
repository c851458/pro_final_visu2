"""Supervisor de entrenamientos PPO: ejecuta los brazos x semillas de uno o varios contrastes sin detenerse.

Planes de contingencia:
1. Proceso que muere (error, OOM, kill): se relanza y continua desde su ultimo checkpoint (hasta --reintentos veces,
   con espera creciente entre intentos).
2. Proceso colgado: si su archivo latido no cambia en --colgado minutos se mata y se relanza desde el checkpoint.
3. Poca memoria: no lanza un trabajo nuevo si la memoria disponible baja de --memoria-min MB; si un trabajo muere
   por falta de memoria, el paralelismo baja en uno.
4. MLflow/DagsHub caido: el resultado queda en runs/ppo/<trabajo>/resultado/ y se reintenta solo el registro
   (sin reentrenar), cada vez con mas espera.
5. El propio supervisor se cae o se reinicia la maquina: todo el estado se deduce de los archivos de cada trabajo,
   asi que relanzarlo continua donde iba (no repite lo registrado, adopta los procesos que siguen vivos y reanuda
   el resto desde su checkpoint). lanzar_ppo.sh lo relanza automaticamente si termina con error.
6. Al terminar todos los trabajos: aplica los criterios pre-registrados a cada contraste (comparar_finalistas
   con el test reservado).

Estado legible en runs/ppo/estado.json; registro en runs/ppo/supervisor.log; salida de cada trabajo en
runs/ppo/<trabajo>/salida.log.

Uso:
    python pipeline/supervisor_ppo.py ppo-v2-n2 ppo-v2-n3 [--paralelo 6]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs", "ppo")
TRAIN = os.path.join(ROOT, "pipeline", "entrenar_ppo.py")
COMPARE = os.path.join(ROOT, "pipeline", "comparar_finalistas.py")
PYTHON = sys.executable


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(RUNS_DIR, "supervisor.log"), "a") as f:
        f.write(line + "\n")


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 10**6


def alive(pid):
    """El proceso existe y no es un zombi."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def sha256_json(data):
    sys.path.insert(0, ROOT)
    from warehouse_rush import niveles
    return niveles.sha256_json(data)


class Job:
    def __init__(self, contrast, criteria, arm_id, arm, seed):
        self.contrast = contrast
        self.name = f"{contrast}-{arm_id}-s{seed}"
        self.dir = os.path.join(RUNS_DIR, self.name)
        os.makedirs(self.dir, exist_ok=True)
        c = criteria["contrastes"][contrast]
        self.cmd = [PYTHON, "-u", TRAIN, "--nivel", c["nivel"], "--pasos", str(arm["pasos"]), "--semilla", str(seed),
                    "--nombre", self.name, "--hipotesis", " ".join(arm["hipotesis"].split()),
                    "--tag", f"grupo={contrast}", "--tag", f"brazo={arm_id}", "--tag", f"brazo_nombre={arm['nombre']}",
                    "--tag", "tipo=finalista", "--tag", f"criterios_sha256={sha256_json(criteria)}"]
        self.proc = None       # Popen si lo lanzo este supervisor
        self.pid = None        # pid (propio o adoptado)
        self.mode = None       # "entrenar" o "registrar"
        self.attempts = 0
        self.register_attempts = 0
        self.not_before = 0.0  # espera antes del siguiente intento
        self.last_error = ""
        meta = self._read("supervisor.json") or {}
        self.attempts = meta.get("intentos", 0)
        self.register_attempts = meta.get("intentos_registro", 0)
        if meta.get("pid") and alive(meta["pid"]) and self._cmdline_matches(meta["pid"]):
            self.pid, self.mode = meta["pid"], meta.get("modo", "entrenar")

    def _read(self, fname):
        try:
            with open(os.path.join(self.dir, fname)) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _cmdline_matches(self, pid):
        try:
            with open(f"/proc/{pid}/cmdline") as f:
                cmdline = f.read()
            return self.name in cmdline or self.dir in cmdline
        except FileNotFoundError:
            return False

    def save_meta(self):
        meta = {"pid": self.pid, "modo": self.mode, "intentos": self.attempts,
                "intentos_registro": self.register_attempts, "ultimo_error": self.last_error}
        tmp = os.path.join(self.dir, "supervisor.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(self.dir, "supervisor.json"))

    @property
    def registered(self):
        return os.path.exists(os.path.join(self.dir, "registrado.json"))

    @property
    def trained(self):
        return os.path.exists(os.path.join(self.dir, "resultado", "completo"))

    @property
    def running(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return self.pid is not None and alive(self.pid)

    def progress(self):
        state = self._read("estado.json") or {}
        return state.get("pasos", 0)

    def heartbeat_age(self):
        try:
            return time.time() - os.path.getmtime(os.path.join(self.dir, "latido"))
        except FileNotFoundError:
            return 0.0

    def launch(self, mode):
        self.mode = mode
        cmd = self.cmd if mode == "entrenar" else [PYTHON, "-u", TRAIN, "--registrar", self.dir]
        out = open(os.path.join(self.dir, "salida.log"), "a")
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {mode} =====\n")
        out.flush()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.pid = self.proc.pid
        # el latido arranca ahora (si no, un latido viejo de un intento anterior lo daria por colgado)
        with open(os.path.join(self.dir, "latido"), "w") as f:
            f.write(str(time.time()))
        self.save_meta()

    def kill(self):
        if self.pid and alive(self.pid):
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                os.kill(self.pid, signal.SIGKILL)

    def exit_code(self):
        return self.proc.returncode if self.proc is not None else None

    def status(self):
        if self.registered:
            return "registrado"
        if self.running:
            return "registrando" if self.mode == "registrar" else "entrenando"
        if self.trained:
            return "entrenado, pendiente de registro"
        return "pendiente" if self.attempts == 0 else "esperando reintento"


def write_status(jobs, parallel, phase):
    data = {"actualizado": time.strftime("%Y-%m-%d %H:%M:%S"), "fase": phase, "paralelo": parallel,
            "memoria_disponible_mb": mem_available_mb(),
            "trabajos": {j.name: {"estado": j.status(), "pasos": j.progress(), "intentos": j.attempts,
                                  "intentos_registro": j.register_attempts, "pid": j.pid if j.running else None,
                                  "ultimo_error": j.last_error} for j in jobs}}
    tmp = os.path.join(RUNS_DIR, "estado.json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(RUNS_DIR, "estado.json"))


def tail(path, n=5):
    try:
        with open(path, errors="replace") as f:
            return " | ".join(line.strip() for line in f.readlines()[-n:])[-400:]
    except FileNotFoundError:
        return ""


def supervise(jobs, args):
    parallel = args.paralelo
    while True:
        now = time.time()
        # 1) revisar los que estaban corriendo
        for j in jobs:
            if j.pid is None or j.running:
                if j.running and j.mode == "entrenar" and j.heartbeat_age() > args.colgado * 60:
                    log(f"{j.name}: sin latido hace {j.heartbeat_age() / 60:.0f} min -> se mata y se reanuda")
                    j.kill()
                    j.last_error = "colgado (sin latido)"
                continue
            code = j.exit_code()
            if j.registered:
                log(f"{j.name}: registrado ({j.progress():,} pasos)")
            elif j.trained:
                j.register_attempts += 1 if j.mode == "registrar" else 0
                j.last_error = f"registro fallido (codigo {code}): {tail(os.path.join(j.dir, 'salida.log'), 2)}"
                j.not_before = now + min(60 * 2 ** j.register_attempts, 1800)
                log(f"{j.name}: entrenado pero sin registrar; reintento del registro en "
                    f"{(j.not_before - now) / 60:.0f} min")
            else:
                j.last_error = j.last_error or f"codigo {code}: {tail(os.path.join(j.dir, 'salida.log'))}"
                oom = code in (-9, 137) and mem_available_mb() < args.memoria_min * 2
                if oom and parallel > 1:
                    parallel -= 1
                    log(f"{j.name}: posible falta de memoria; paralelismo baja a {parallel}")
                j.not_before = now + 30 * 2 ** min(j.attempts, 5)
                log(f"{j.name}: fallo el intento {j.attempts} ({j.last_error}); se reanuda desde el checkpoint")
            j.proc, j.pid = None, None
            j.save_meta()
        # 2) lanzar trabajos (uno por ciclo para que la medicion de memoria refleje el anterior)
        pending_register = [j for j in jobs if not j.running and j.trained and not j.registered
                            and now >= j.not_before]
        for j in pending_register:
            log(f"{j.name}: registrando en MLflow (intento {j.register_attempts + 1})")
            j.launch("registrar")
        n_running = sum(j.running and j.mode == "entrenar" for j in jobs)
        to_train = [j for j in jobs if not j.running and not j.trained and now >= j.not_before]
        failed = [j for j in to_train if j.attempts > args.reintentos]
        for j in failed:
            to_train.remove(j)
        if to_train and n_running < parallel:
            if mem_available_mb() < args.memoria_min:
                log(f"memoria disponible {mem_available_mb()} MB < {args.memoria_min} MB: se espera para lanzar")
            else:
                j = to_train[0]
                j.attempts += 1
                j.last_error = ""
                log(f"{j.name}: lanzando entrenamiento (intento {j.attempts}, desde {j.progress():,} pasos)")
                j.launch("entrenar")
        # 3) fin
        open_jobs = [j for j in jobs if not j.registered and j not in failed]
        write_status(jobs, parallel, "entrenando" if open_jobs else "terminado")
        if not open_jobs:
            return [j for j in jobs if not j.registered]
        time.sleep(args.ciclo)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("contrastes", nargs="+")
    parser.add_argument("--paralelo", type=int, default=6, help="entrenamientos simultaneos (maximo)")
    parser.add_argument("--reintentos", type=int, default=5, help="relanzamientos por trabajo tras el primero")
    parser.add_argument("--colgado", type=float, default=20, help="minutos sin latido para dar un trabajo por colgado")
    parser.add_argument("--memoria-min", type=int, default=900, help="MB disponibles minimos para lanzar otro")
    parser.add_argument("--ciclo", type=float, default=15, help="segundos entre revisiones")
    parser.add_argument("--sin-comparar", action="store_true", help="no aplicar los criterios al terminar")
    parser.add_argument("--criterios", default=os.path.join(ROOT, "configs", "criterios.yaml"))
    args = parser.parse_args()
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(args.criterios) as f:
        criteria = yaml.safe_load(f)
    jobs = [Job(name, criteria, arm_id, arm, seed)
            for name in args.contrastes
            for arm_id, arm in criteria["contrastes"][name]["brazos"].items()
            for seed in criteria["contrastes"][name]["semillas"]]
    adopted = [j.name for j in jobs if j.running]
    log(f"supervisor iniciado (pid {os.getpid()}): {len(jobs)} trabajos, paralelo {args.paralelo}"
        + (f"; adopta procesos vivos: {', '.join(adopted)}" if adopted else ""))
    failed = supervise(jobs, args)
    if failed:
        log(f"trabajos que agotaron los reintentos: {', '.join(j.name for j in failed)}")
        sys.exit(2)
    log("todos los trabajos registrados")
    if args.sin_comparar:
        return
    for name in args.contrastes:
        marker = os.path.join(RUNS_DIR, f"comparado-{name}")
        if os.path.exists(marker):
            continue
        for attempt in range(1, 6):
            log(f"{name}: aplicando criterios con el test reservado (intento {attempt})")
            code = subprocess.call([PYTHON, "-u", COMPARE, name, "--test-reservado"], cwd=ROOT,
                                   stdout=open(os.path.join(RUNS_DIR, f"comparar-{name}.log"), "a"),
                                   stderr=subprocess.STDOUT)
            if code == 0:
                open(marker, "w").close()
                log(f"{name}: decision registrada (docs/resultados/{name}.md)")
                break
            time.sleep(60 * attempt)
        else:
            log(f"{name}: comparar_finalistas fallo 5 veces; ver runs/ppo/comparar-{name}.log")
            sys.exit(4)
    log("supervisor terminado")


if __name__ == "__main__":
    main()
