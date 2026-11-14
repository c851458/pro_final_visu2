"""Entrena una politica de red neuronal (PPO, imitacion o imitacion + PPO) y la registra en MLflow.

Algoritmos (--algoritmo):
- ppo            PPO desde cero (Stable-Baselines3).
- imitacion      la red imita al planificador de reglas con DAgger: en cada iteracion juega pedidos de
                 entrenamiento, el planificador etiqueta con su accion cada estado visitado (tambien los errores
                 de la red) y la red se ajusta a todas las etiquetas acumuladas. Tambien ajusta el critico (valor).
- imitacion+ppo  imitacion y despues afinamiento con PPO sobre la recompensa de negocio (--pasos de PPO).

La red recibe las caracteristicas de warehouse_rush/ppo.py (incluyen bateria exacta y plazo restante del SLA).
Se registran las mismas curvas, metricas, resultados por caso y criterios que en Q-learning, y el modelo se exporta
a NumPy (politica.npz) para que la API lo sirva sin PyTorch.

Tolerancia a fallos (cada trabajo vive en runs/ppo/<nombre>/):
- checkpoint.zip y curva.json se guardan en cada evaluacion de la curva; si el proceso muere, al relanzarlo con el
  mismo nombre continua desde el ultimo checkpoint (tag reanudaciones = cuantas veces se reanudo).
- latido: se actualiza durante el entrenamiento; el supervisor lo usa para detectar procesos colgados.
- resultado/: paquete completo (politica, metricas, casos, curva, configuracion, tags) que se guarda ANTES de
  registrar en MLflow. Si MLflow falla, el entrenamiento no se pierde: `--registrar DIR` lo registra despues.
- registrado.json: se escribe al registrar; un trabajo con este archivo no se vuelve a entrenar ni a registrar.

Codigos de salida: 0 registrado, 3 entrenado pero sin registrar (reintentar con --registrar), otro: fallo.

Uso:
    python pipeline/entrenar_ppo.py --nivel v2-n2 --pasos 3000000 --semilla 0 --nombre ppo-v2-n2-s0 --hipotesis "..."
    python pipeline/entrenar_ppo.py --registrar runs/ppo/ppo-v2-n2-A-s0
    python pipeline/supervisor_ppo.py ppo-v2-n2 ppo-v2-n3     # contrastes completos, con reintentos
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

import tracking  # noqa: E402  (antes que mlflow: configura su entorno)
import mlflow  # noqa: E402
from common import resolve_params, load_params  # noqa: E402
from warehouse_rush import niveles  # noqa: E402
from common import data_path, load_orders  # noqa: E402
from warehouse_rush.politicas import RulePolicy  # noqa: E402
from warehouse_rush.ppo import FEATURE_NAMES, FeatureObservation, MLPPolicy, export_sb3, features  # noqa: E402
from warehouse_rush.simulacion import evaluate  # noqa: E402

ROOT = tracking.ROOT
RUNS_DIR = os.path.join(ROOT, "runs", "ppo")
PPO_DEFAULTS = {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 512, "n_envs": 8, "ent_coef": 0.01,
                "gamma": 0.99, "clip_range": 0.2, "net_arch": [128, 128], "reward_scale": 1e-3, "evaluar_cada": 250_000,
                "casos_curva": 50}
IMITATION_DEFAULTS = {"iteraciones_dagger": 8, "episodios_por_iteracion": 1500, "epocas": 4, "lr": 1e-3,
                      "lote": 1024, "peso_valor": 0.5}
ALGORITHMS = ("ppo", "imitacion", "imitacion+ppo")
EXIT_NOT_REGISTERED = 3


def _write_json(path, data):
    """Escritura atomica: un corte a mitad nunca deja un JSON a medias."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, path)


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _touch(path):
    with open(path, "w") as f:
        f.write(str(time.time()))


def job_dir(name):
    return os.path.join(RUNS_DIR, name)


def imitate(model, cfg, params, imitation, seed, heartbeat, curve, curve_cases):
    """DAgger con el planificador de reglas como maestro. Ajusta actor (entropia cruzada) y critico (retornos).
    Devuelve (pasos de ambiente usados, muestras, precision final sobre las etiquetas)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    orders = load_orders(os.path.join(ROOT, data_path(cfg["nivel"], "entrenamiento")))
    env, expert, policy = niveles.make_env(cfg), RulePolicy(), model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=imitation["lr"])
    X, Y, G = [], [], []
    env_steps = 0
    accuracy = 0.0
    for it in range(imitation["iteraciones_dagger"]):
        learner = None if it == 0 else MLPPolicy(export_sb3(model))  # iteracion 0: solo demostraciones
        for ep in range(imitation["episodios_por_iteracion"]):
            order = orders[int(rng.integers(len(orders)))]
            obs, _ = env.reset(seed=int(rng.integers(2**31)), options={"pedido": order})
            xs, ys, rewards, done = [], [], [], False
            while not done:
                x = features(env.unwrapped)
                label = expert.act(env, obs)
                xs.append(x)
                ys.append(label)
                action = label if learner is None else int(np.argmax(learner.logits(x)))
                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward * params["reward_scale"])
                done = terminated or truncated
            ret, returns = 0.0, []
            for r in reversed(rewards):
                ret = r + params["gamma"] * ret
                returns.append(ret)
            X.append(np.stack(xs))  # un arreglo por episodio: sin el costo de memoria de un objeto por paso
            Y.append(np.array(ys, dtype=np.int64))
            G.append(np.array(returns[::-1], dtype=np.float32))
            env_steps += len(rewards)
            if ep % 100 == 0:
                _touch(heartbeat)
        x_t = torch.from_numpy(np.concatenate(X))
        y_t = torch.from_numpy(np.concatenate(Y))
        g_t = torch.from_numpy(np.concatenate(G))
        policy.set_training_mode(True)
        for _ in range(imitation["epocas"]):
            perm = torch.randperm(len(x_t))
            for i in range(0, len(x_t), imitation["lote"]):
                idx = perm[i:i + imitation["lote"]]
                logits = policy.get_distribution(x_t[idx]).distribution.logits
                values = policy.predict_values(x_t[idx]).squeeze(-1)
                loss = F.cross_entropy(logits, y_t[idx]) + imitation["peso_valor"] * F.mse_loss(values, g_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            _touch(heartbeat)
        policy.set_training_mode(False)
        with torch.no_grad():
            sample = torch.randperm(len(x_t))[:50_000]
            pred = policy.get_distribution(x_t[sample]).distribution.logits.argmax(-1)
            accuracy = float((pred == y_t[sample]).float().mean())
        metrics, _ = evaluate(niveles.make_env(cfg), MLPPolicy(export_sb3(model)), curve_cases)
        curve.append({"episodio": env_steps, "tasa_exito": metrics["tasa_exito"],
                      "recompensa_media": metrics["recompensa_media"], "rescates_medios": metrics["rescates_medios"],
                      "pasos_medios": metrics["pasos_medios"], "precision_imitacion": accuracy,
                      "eval_tasa_exito": metrics["tasa_exito"], "eval_recompensa_media": metrics["recompensa_media"],
                      "eval_pasos_medios": metrics["pasos_medios"]})
        print(f"  DAgger iteracion {it + 1}: {len(x_t):,} etiquetas, precision {accuracy:.3f}, "
              f"exito eval {metrics['tasa_exito']:.2f} rescates {metrics['rescates_medios']:.2f}", flush=True)
    return env_steps, int(len(x_t)), accuracy


def train(level, steps, seed, name, hypothesis, extra_tags, ppo_params, algorithm="ppo"):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    torch.set_num_threads(1)
    work = job_dir(name)
    os.makedirs(work, exist_ok=True)
    checkpoint, curve_path, state_path, heartbeat = (os.path.join(work, f) for f in
                                                     ("checkpoint.zip", "curva.json", "estado.json", "latido"))
    _touch(heartbeat)
    params = {**PPO_DEFAULTS, **(ppo_params or {})}
    cfg = resolve_params(dict(load_params(os.path.join(ROOT, "params.yaml")), nivel=level))
    cfg.pop("agente")
    cfg.update(algoritmo=algorithm, ppo=params, entrenamiento={"semilla": seed, "pasos": steps})
    imitating = algorithm.startswith("imitacion")
    if imitating:
        cfg["imitacion"] = dict(IMITATION_DEFAULTS, maestro="politica de reglas (BFS)")
    env = niveles.make_env(cfg)
    cases, suite_id = niveles.evaluation_suite(cfg)
    curve_cases = cases[:params["casos_curva"]]
    state = _read_json(state_path, {"segundos": 0.0, "reanudaciones": 0, "code_version": tracking.code_version()})
    curve = _read_json(curve_path, []) if os.path.exists(checkpoint) else []

    venv = make_vec_env(lambda: FeatureObservation(niveles.make_env(cfg), reward_scale=params["reward_scale"]),
                        n_envs=params["n_envs"], seed=seed, monitor_kwargs={"info_keywords": ("success", "rescues")})
    if os.path.exists(checkpoint):
        model = PPO.load(checkpoint, env=venv, device="cpu")
        state["reanudaciones"] += 1
        print(f"{name}: reanuda desde el checkpoint en {model.num_timesteps:,} pasos "
              f"(reanudacion {state['reanudaciones']})", flush=True)
    else:
        model = PPO("MlpPolicy", venv, learning_rate=params["learning_rate"], n_steps=params["n_steps"],
                    batch_size=params["batch_size"], ent_coef=params["ent_coef"], gamma=params["gamma"],
                    clip_range=params["clip_range"], policy_kwargs={"net_arch": params["net_arch"]}, seed=seed,
                    verbose=0, device="cpu")
    session_start = time.time()
    if imitating and "imitacion" not in state:
        t0 = time.time()
        env_steps, samples, accuracy = imitate(model, cfg, params, cfg["imitacion"], seed, heartbeat, curve,
                                               curve_cases)
        state["imitacion"] = {"pasos": env_steps, "muestras": samples, "precision": accuracy,
                              "segundos": time.time() - t0}
    offset = state.get("imitacion", {}).get("pasos", 0)  # eje de la curva: pasos de ambiente totales
    base_seconds = state["segundos"]

    def save_progress():
        model.save(f"{checkpoint}.tmp.zip")
        os.replace(f"{checkpoint}.tmp.zip", checkpoint)
        _write_json(curve_path, curve)
        _write_json(state_path, {**state, "segundos": base_seconds + time.time() - session_start,
                                 "pasos": model.num_timesteps})

    class Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = (model.num_timesteps // params["evaluar_cada"] + 1) * params["evaluar_cada"]
            self.next_beat = 0

        def _on_step(self):
            if self.num_timesteps >= self.next_beat:
                self.next_beat = self.num_timesteps + 20_000
                _touch(heartbeat)
            if self.num_timesteps >= self.next:
                self.next += params["evaluar_cada"]
                info = list(self.model.ep_info_buffer)
                metrics, _ = evaluate(niveles.make_env(cfg), MLPPolicy(export_sb3(self.model)), curve_cases)
                curve.append({
                    "episodio": offset + self.num_timesteps,  # en PPO el eje es de pasos de ambiente
                    "tasa_exito": float(np.mean([e.get("success", 0) for e in info])) if info else 0.0,
                    "recompensa_media": float(np.mean([e["r"] for e in info]) / params["reward_scale"]) if info else 0.0,
                    "rescates_medios": float(np.mean([e.get("rescues", 0) for e in info])) if info else 0.0,
                    "pasos_medios": float(np.mean([e["l"] for e in info])) if info else 0.0,
                    "eval_tasa_exito": metrics["tasa_exito"], "eval_recompensa_media": metrics["recompensa_media"],
                    "eval_pasos_medios": metrics["pasos_medios"]})
                save_progress()
                print(f"  {name} {self.num_timesteps:>9,} pasos: exito eval {metrics['tasa_exito']:.2f} "
                      f"rescates {metrics['rescates_medios']:.2f}", flush=True)
            return True

    if imitating:
        save_progress()  # checkpoint tras la imitacion: un corte durante el afinamiento no la repite
    remaining = steps - model.num_timesteps
    if remaining > 0:
        model.learn(remaining, callback=Curve(), reset_num_timesteps=False)
    save_progress()
    elapsed = base_seconds + time.time() - session_start
    _touch(heartbeat)

    weights = export_sb3(model)
    metrics, per_case = evaluate(niveles.make_env(cfg), MLPPolicy(weights), cases)
    if imitating:
        imit = state["imitacion"]
        cfg["entrenamiento"].update(pasos=imit["pasos"] + steps, pasos_imitacion=imit["pasos"], pasos_ppo=steps)
        metrics_extra = {"imitacion_precision": imit["precision"], "imitacion_muestras": imit["muestras"]}
    else:
        metrics_extra = {}
    origin = {"ppo": "desde cero", "imitacion": "imitacion de la politica de reglas (DAgger)",
              "imitacion+ppo": "imitacion de la politica de reglas (DAgger) + afinamiento PPO"}[algorithm]
    tags = {"algoritmo": algorithm, "familia_modelo": niveles.ppo_family(env), "hipotesis": hypothesis or "",
            "estado_modelo": "checkpoint final disponible", "origen_modelo": origin,
            "caracteristicas": ",".join(FEATURE_NAMES), "trabajo": name,
            "reanudaciones": str(state["reanudaciones"]), **state["code_version"], **(extra_tags or {})}
    package = os.path.join(work, "resultado")
    os.makedirs(package, exist_ok=True)
    np.savez(os.path.join(package, "politica.npz"), **weights)
    for fname, data in (("curva.json", curve), ("cfg.json", cfg), ("tags.json", tags),
                        ("evaluacion.json", {"metricas": metrics, "casos": per_case, "suite_id": suite_id,
                                             "tiempo_entrenamiento_s": elapsed, "extra": metrics_extra})):
        _write_json(os.path.join(package, fname), data)
    _touch(os.path.join(package, "completo"))  # marca de paquete completo (ultimo archivo en escribirse)
    print(f"{name}: exito {metrics['tasa_exito'] * 100:.1f}% OTIF {metrics.get('otif', 0) * 100:.1f}% "
          f"express {metrics.get('cumplimiento_express', 0) * 100:.1f}% rescates {metrics['rescates_medios']:.2f} "
          f"utilidad ${metrics.get('utilidad_media_cop', 0):,.0f} ({elapsed / 60:.1f} min)", flush=True)
    return register(work)


def _existing_run(name):
    """Run ya registrado de este trabajo (por si se registro pero se corto antes de escribir registrado.json)."""
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], output_format="list", max_results=5,
                              filter_string=f"tags.trabajo = '{name}' and attributes.status = 'FINISHED'")
    return runs[0] if runs else None


def register(work):
    """Registra en MLflow el paquete resultado/ de un trabajo. Devuelve el codigo de salida."""
    package = os.path.join(work, "resultado")
    done = os.path.join(work, "registrado.json")
    if os.path.exists(done):
        print(f"{work}: ya registrado ({_read_json(done)['run_id']})")
        return 0
    if not os.path.exists(os.path.join(package, "completo")):
        print(f"{work}: el paquete resultado/ no esta completo", file=sys.stderr)
        return 1
    cfg, tags, curve, ev = (_read_json(os.path.join(package, f)) for f in
                            ("cfg.json", "tags.json", "curva.json", "evaluacion.json"))
    name = tags["trabajo"]
    try:
        tracking.setup()
        existing = _existing_run(name)
        if existing:
            run_id = existing.info.run_id
        else:
            with tempfile.TemporaryDirectory() as tmp, tracking.start_run(
                    name, cfg, origin="pipeline/entrenar_ppo.py", extra_tags=tags) as run:
                tracking.log_curves(curve)
                mlflow.log_metric("tiempo_entrenamiento_s", ev["tiempo_entrenamiento_s"])
                if ev.get("extra"):
                    mlflow.log_metrics(ev["extra"])
                tracking.log_evaluation(ev["metricas"], ev["casos"], ev["suite_id"])
                mlflow.log_artifact(os.path.join(package, "politica.npz"), "modelo")
                run_id = run.info.run_id
    except Exception as e:  # MLflow/DagsHub caido o sin red: el paquete queda para registrarlo despues
        print(f"{name}: no se pudo registrar en MLflow ({type(e).__name__}: {e}); "
              f"reintentar con --registrar {work}", file=sys.stderr, flush=True)
        return EXIT_NOT_REGISTERED
    _write_json(done, {"run_id": run_id, "registrado": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"{name}: registrado -> {run_id}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nivel", default="v2-n2")
    parser.add_argument("--pasos", type=int, default=3_000_000)
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--nombre")
    parser.add_argument("--hipotesis")
    parser.add_argument("--tag", action="append", default=[], help="tag extra clave=valor")
    parser.add_argument("--registrar", metavar="DIR", help="registra en MLflow un trabajo ya entrenado")
    parser.add_argument("--algoritmo", choices=ALGORITHMS, default="ppo")
    parser.add_argument("--ppo-param", action="append", default=[], help="hiperparametro PPO clave=valor")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.registrar:
        sys.exit(register(args.registrar))
    import yaml

    tags = dict(t.split("=", 1) for t in args.tag)
    ppo_params = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in args.ppo_param)}
    sys.exit(train(args.nivel, args.pasos, args.semilla, args.nombre or f"ppo-{args.nivel}-s{args.semilla}",
                   args.hipotesis, tags, ppo_params, args.algoritmo))


if __name__ == "__main__":
    main()
