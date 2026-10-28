"""Descarga de modelos publicados en Hugging Face Hub.

Formato de referencia: "hf:usuario/repositorio" o "hf:usuario/repositorio@revision"
(la revision puede ser una rama, un tag o un commit).
"""

MODEL_FILE = "q_table.npy"


def resolve_model(spec):
    """Devuelve la ruta local del modelo: descarga de Hugging Face si `spec` empieza con "hf:"."""
    if not spec.startswith("hf:"):
        return spec
    from huggingface_hub import hf_hub_download

    repo_id, _, revision = spec[3:].partition("@")
    return hf_hub_download(repo_id, MODEL_FILE, revision=revision or None)
