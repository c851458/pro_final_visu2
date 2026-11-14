"""Pruebas de la API de simulacion e inferencia (la que se despliega en Vercel)."""

import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import index as api_module
from warehouse_rush import niveles


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API con un catalogo de prueba: un modelo compatible con N2/N3 servido desde un archivo local."""
    env = niveles.make_env(niveles.resolve("n2"))
    q = np.zeros(tuple(env.observation_space.nvec) + (env.action_space.n,), dtype=np.float32)
    weights = tmp_path / "q_table.npy"
    np.save(weights, q)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifests = tmp_path / "manifests"
    (manifests / "modelos").mkdir(parents=True)
    manifest = {"model_id": "prueba", "familia_modelo": niveles.family(env), "model_sha256": sha,
                "model_download_url": weights.as_uri(), "level_id": "n2", "environment_version": "1.0.0",
                "run_name": "prueba", "hf_commit": "x", "git_commit": "y"}
    (manifests / "modelos" / "prueba.json").write_text(json.dumps(manifest))
    (manifests / "catalogo.json").write_text(json.dumps({"recomendados": {"n2": "prueba"}, "modelos": [manifest]}))
    monkeypatch.setattr(api_module, "MANIFESTS", str(manifests))
    monkeypatch.setattr(api_module, "CACHE_DIR", str(tmp_path / "cache"))
    api_module.load_model.cache_clear()
    yield TestClient(api_module.app)
    api_module.load_model.cache_clear()


def test_salud_y_niveles(client):
    assert client.get("/api/salud").json()["modelos_recomendados"] == {"n2": "prueba"}
    levels = client.get("/api/niveles").json()["niveles"]
    assert [lv["nivel"] for lv in levels] == ["n1", "n2", "n3", "v2-n1", "v2-n2", "v2-n3"]
    assert {lv["nivel"]: lv["environment_version"] for lv in levels}["v2-n2"] == "2.0.0"
    assert levels[2]["mapa"]["bloqueos"] == [[7, 4], [7, 7]]


def test_episodio_manual_reproduce_las_acciones(client):
    data = client.post("/api/episodio", json={"nivel": "n2", "politica": "manual", "acciones": [0, 0, 2]}).json()
    assert data["modo"] == "manual" and len(data["fotogramas"]) == 4
    assert data["fotogramas"][-1]["steps"] == 3


@pytest.mark.parametrize("policy", ["aleatoria", "regla"])
def test_episodio_con_referencias(client, policy):
    data = client.post("/api/episodio", json={"nivel": "n1", "politica": policy, "caso": 3}).json()
    assert data["modo"] == policy and data["resultado"]["pasos"] == len(data["fotogramas"]) - 1


def test_evaluar_modelo_verifica_sha_y_compatibilidad(client):
    data = client.post("/api/evaluar", json={"nivel": "n2", "politica": "modelo", "casos": 3}).json()
    assert data["modelo"]["model_id"] == "prueba" and data["metricas"]["pedidos_evaluados"] == 3
    # N3 es de la misma familia: se puede evaluar el modelo de N2, pero no es su recomendado
    same_family = client.post("/api/evaluar", json={"nivel": "n3", "politica": "modelo", "model_id": "prueba", "casos": 3})
    assert same_family.status_code == 200
    assert client.post("/api/evaluar", json={"nivel": "n3", "politica": "modelo", "casos": 3}).status_code == 404
    incompatible = client.post("/api/evaluar", json={"nivel": "n1", "politica": "modelo", "model_id": "prueba", "casos": 3})
    assert incompatible.status_code == 409


def test_errores_claros(client):
    assert client.post("/api/episodio", json={"nivel": "n9"}).status_code == 404
    assert client.post("/api/episodio", json={"nivel": "n1", "caso": 9999}).status_code == 400
    assert client.get("/api/modelos/no-existe").status_code == 404


def test_modelo_ppo_se_sirve_en_numpy_solo_en_niveles_compatibles(tmp_path, monkeypatch):
    """Una red PPO exportada (.npz) se evalua sin PyTorch en niveles v2 y se rechaza en v1."""
    import numpy as np
    from warehouse_rush.ppo import N_FEATURES

    rng = np.random.default_rng(0)
    weights = tmp_path / "politica.npz"
    np.savez(weights, n_layers=np.array(2), W0=rng.normal(size=(N_FEATURES, 8)).astype(np.float32),
             b0=np.zeros(8, np.float32), W1=rng.normal(size=(8, 8)).astype(np.float32), b1=np.zeros(8, np.float32),
             W_out=rng.normal(size=(8, 7)).astype(np.float32), b_out=np.zeros(7, np.float32))
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifests = tmp_path / "manifests"
    (manifests / "modelos").mkdir(parents=True)
    env = niveles.make_env(niveles.resolve("v2-n2"))
    manifest = {"model_id": "ppo", "tipo_modelo": "ppo-mlp", "familia_modelo": niveles.ppo_family(env),
                "model_sha256": sha, "model_download_url": weights.as_uri(), "level_id": "v2-n2",
                "environment_version": "2.0.0", "run_name": "ppo", "hf_commit": "x", "git_commit": "y"}
    (manifests / "modelos" / "ppo.json").write_text(json.dumps(manifest))
    (manifests / "catalogo.json").write_text(json.dumps({"recomendados": {}, "modelos": [manifest]}))
    monkeypatch.setattr(api_module, "MANIFESTS", str(manifests))
    monkeypatch.setattr(api_module, "CACHE_DIR", str(tmp_path / "cache"))
    api_module.load_model.cache_clear()
    client = TestClient(api_module.app)
    ok = client.post("/api/evaluar", json={"nivel": "v2-n3", "politica": "modelo", "model_id": "ppo", "casos": 2})
    assert ok.status_code == 200 and ok.json()["modelo"]["tipo_modelo"] == "ppo-mlp"
    assert client.post("/api/evaluar", json={"nivel": "n2", "politica": "modelo", "model_id": "ppo", "casos": 2}).status_code == 409
    api_module.load_model.cache_clear()
