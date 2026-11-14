"""Encadena las etapas posteriores al entrenamiento PPO, una tras otra, sin intervencion manual.

Etapas (cada una se marca como hecha en runs/ppo/etapas.json; relanzar el script salta lo ya hecho):
1. entrenar    espera al supervisor (entrenamiento + registro + criterios). Si termina sin completar, lo relanza
               con los reintentos reiniciados (hasta 3 veces).
2. criterios   comprueba que cada contraste tenga su decision (docs/resultados/<contraste>.json); si falta, ejecuta
               comparar_finalistas con el test reservado (con reintentos).
3. publicar    para cada contraste con ganador: flujo Prefect en modo checkpoint con --promover --publicar (valida,
               pruebas, evalua, registra, aplica criterios, commit + push, verifica la URL y hace rollback si falla).
               Si el servidor Prefect no responde, lo arranca. Reintenta el flujo hasta 3 veces.
4. evidencia   commit + push de los informes de decision que no haya subido el flujo (contrastes sin ganador).
5. verificar   espera a que Vercel sirva el ultimo commit de main (/api/salud).

Cada paso reintenta con espera creciente; un fallo definitivo queda escrito en runs/ppo/etapas.json y en
runs/ppo/etapas.log, y lanzar_etapas_ppo.sh relanza este script si termina con error.

Uso: python pipeline/etapas_ppo.py ppo-v2-n2 ppo-v2-n3
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs", "ppo")
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
PREFECT = os.path.join(ROOT, ".venv-prefect", "bin", "prefect")
URL = "https://profinalvisu2.vercel.app"
STATE = os.path.join(RUNS_DIR, "etapas.json")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(RUNS_DIR, "etapas.log"), "a") as f:
        f.write(line + "\n")


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    state["actualizado"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{STATE}.tmp", "w") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)
    os.replace(f"{STATE}.tmp", STATE)


def run(cmd, log_name, timeout=7200, env=None):
    with open(os.path.join(RUNS_DIR, log_name), "a") as out:
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)} =====\n")
        out.flush()
        try:
            return subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=timeout,
                                  env={**os.environ, **(env or {})}).returncode
        except subprocess.TimeoutExpired:
            out.write("TIEMPO AGOTADO\n")
            return 124


def retry(what, fn, attempts=4, wait=60):
    """Ejecuta fn() hasta que devuelva True, con espera creciente; devuelve si lo logro."""
    for attempt in range(1, attempts + 1):
        try:
            if fn():
                return True
        except Exception as e:  # noqa: BLE001  (cualquier error se reintenta y queda en el registro)
            log(f"{what}: error {type(e).__name__}: {e}")
        if attempt < attempts:
            log(f"{what}: intento {attempt} fallido; reintento en {wait * attempt} s")
            time.sleep(wait * attempt)
    log(f"{what}: FALLO tras {attempts} intentos")
    return False


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def supervisor_alive():
    path = os.path.join(RUNS_DIR, "supervisor.pid")
    try:
        os.kill(int(open(path).read().strip()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def all_registered(contrasts):
    import yaml
    with open(os.path.join(ROOT, "configs", "criterios.yaml")) as f:
        criteria = yaml.safe_load(f)["contrastes"]
    names = [f"{c}-{arm}-s{seed}" for c in contrasts for arm in criteria[c]["brazos"] for seed in criteria[c]["semillas"]]
    return all(os.path.exists(os.path.join(RUNS_DIR, n, "registrado.json")) for n in names), names


# ---------------------------------------------------------------- 1. entrenar

def stage_train(contrasts):
    for relaunch in range(4):
        while supervisor_alive():
            time.sleep(30)
        done, names = all_registered(contrasts)
        if done:
            log("1-entrenar: todos los trabajos entrenados y registrados")
            return True
        deterministic = [n for n in names if (json.load(open(os.path.join(RUNS_DIR, n, "supervisor.json")))
                                              if os.path.exists(os.path.join(RUNS_DIR, n, "supervisor.json"))
                                              else {}).get("error_deterministico")]
        if deterministic:
            log(f"1-entrenar: errores de codigo en {', '.join(deterministic)}; relanzar no los arregla")
            return False
        if relaunch == 3:
            break
        log(f"1-entrenar: el supervisor termino con trabajos sin registrar; se relanza ({relaunch + 1}/3) "
            "con los reintentos reiniciados")
        for n in names:  # reiniciar el contador de intentos (los checkpoints se conservan)
            meta = os.path.join(RUNS_DIR, n, "supervisor.json")
            if os.path.exists(meta):
                data = json.load(open(meta))
                data.update(intentos=0, intentos_registro=0)
                json.dump(data, open(meta, "w"))
        subprocess.run([os.path.join(ROOT, "pipeline", "lanzar_ppo.sh"), *contrasts], cwd=ROOT)
        time.sleep(20)
    log("1-entrenar: FALLO, quedan trabajos sin registrar tras 3 relanzamientos")
    return False


# ---------------------------------------------------------------- 2. criterios

def decision(contrast):
    try:
        with open(os.path.join(ROOT, "docs", "resultados", f"{contrast}.json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def stage_criteria(contrast):
    if decision(contrast) and os.path.exists(os.path.join(RUNS_DIR, f"comparado-{contrast}")):
        return True

    def attempt():
        code = run([PYTHON, "-u", "pipeline/comparar_finalistas.py", contrast, "--test-reservado"],
                   f"comparar-{contrast}.log")
        if code == 0 and decision(contrast):
            open(os.path.join(RUNS_DIR, f"comparado-{contrast}"), "w").close()
            return True
        return False

    return retry(f"2-criterios {contrast}", attempt)


# ---------------------------------------------------------------- 3. publicar

def prefect_up():
    try:
        with urllib.request.urlopen("http://127.0.0.1:4200/api/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def ensure_prefect():
    if prefect_up():
        return True
    log("3-publicar: el servidor Prefect no responde; se arranca")
    subprocess.Popen([PREFECT, "server", "start", "--host", "127.0.0.1", "--port", "4200"], cwd=ROOT,
                     stdout=open(os.path.join(RUNS_DIR, "prefect-server.log"), "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(40):
        time.sleep(3)
        if prefect_up():
            return True
    return False


def stage_publish(contrast, level):
    d = decision(contrast)
    if not d.get("ganador"):
        log(f"3-publicar {contrast}: sin ganador ({d.get('motivo')}); se mantiene el recomendado actual de {level}")
        return True
    run_id = d["seleccionado"]["run_id"]

    def attempt():
        if not ensure_prefect():
            raise RuntimeError("no se pudo arrancar el servidor Prefect")
        code = run([PYTHON, "-u", "pipeline/flujo.py", "--nivel", level, "--modo", "checkpoint", "--run", run_id,
                    "--promover", "--publicar"], f"flujo-{contrast}.log",
                   env={"PREFECT_API_URL": "http://127.0.0.1:4200/api"})
        return code == 0

    log(f"3-publicar {contrast}: ganador {d['ganador']}, se publica el run {run_id} en {level} con el flujo Prefect")
    return retry(f"3-publicar {contrast}", attempt, attempts=3, wait=120)


# ---------------------------------------------------------------- 4. evidencia

def stage_evidence(contrasts):
    files = [os.path.join("docs", "resultados", f"{c}.{ext}") for c in contrasts for ext in ("md", "json")]
    files = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
    git("add", *files)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        log("4-evidencia: los informes ya estaban subidos")
        return True
    git("commit", "-q", "-m", f"Resultados del PPO con red neuronal: decision de {', '.join(contrasts)}")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")

    def push():
        ok = subprocess.run(["git", "push", "-q", "origin", f"{branch}:main"], cwd=ROOT).returncode == 0
        subprocess.run(["git", "push", "-q", "origin", branch], cwd=ROOT)
        return ok

    if retry("4-evidencia push", push):
        log(f"4-evidencia: informes subidos a main ({git('rev-parse', '--short', 'HEAD')})")
        return True
    return False


# ---------------------------------------------------------------- 5. verificar

def stage_verify():
    expected = git("rev-parse", "HEAD")

    def served():
        with urllib.request.urlopen(f"{URL}/api/salud", timeout=30) as r:
            health = json.load(r)
        if health.get("git_commit") == expected:
            log(f"5-verificar: Vercel sirve {expected[:7]}; recomendados {health.get('modelos_recomendados')}")
            return True
        log(f"5-verificar: Vercel aun sirve {str(health.get('git_commit'))[:7]}, se espera {expected[:7]}")
        return False

    return retry("5-verificar", served, attempts=20, wait=30)


def main():
    contrasts = sys.argv[1:] or ["ppo-v2-n2", "ppo-v2-n3"]
    os.makedirs(RUNS_DIR, exist_ok=True)
    import yaml
    with open(os.path.join(ROOT, "configs", "criterios.yaml")) as f:
        criteria = yaml.safe_load(f)["contrastes"]
    levels = {c: criteria[c]["nivel"] for c in contrasts}
    state = load_state()
    state.setdefault("etapas", {})
    log(f"etapas iniciadas (pid {os.getpid()}) para {', '.join(contrasts)}")

    plan = [("1-entrenar", lambda: stage_train(contrasts))]
    plan += [(f"2-criterios {c}", (lambda c=c: stage_criteria(c))) for c in contrasts]
    plan += [(f"3-publicar {c}", (lambda c=c: stage_publish(c, levels[c]))) for c in contrasts]
    plan += [("4-evidencia", lambda: stage_evidence(contrasts)), ("5-verificar", stage_verify)]
    for name, fn in plan:
        if state["etapas"].get(name) == "hecho":
            continue
        state["etapas"][name] = "en curso"
        save_state(state)
        log(f"{name}: inicia")
        if not fn():
            state["etapas"][name] = "fallido"
            save_state(state)
            log(f"{name}: FALLIDO; se detienen las etapas siguientes (ver runs/ppo/*.log)")
            sys.exit(1)
        state["etapas"][name] = "hecho"
        save_state(state)
    state["resultado"] = {c: (decision(c) or {}).get("motivo") for c in contrasts}
    save_state(state)
    log("todas las etapas terminadas")


if __name__ == "__main__":
    main()
