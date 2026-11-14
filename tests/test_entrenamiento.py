"""Regresiones del entrenamiento de redes: hiperparametros de la linea de comandos y comandos del supervisor."""

import sys

import pytest

sys.path.insert(0, "pipeline")
from common import parse_value  # noqa: E402


@pytest.mark.parametrize("text,expected", [("5e-05", 5e-05), ("0.1", 0.1), ("0.0", 0.0), ("8", 8),
                                           ("[128, 128]", [128, 128]), ("true", True)])
def test_parse_value_tipos(text, expected):
    value = parse_value(text)
    assert value == expected and type(value) is type(expected)


def test_hiperparametros_de_los_brazos_son_numericos():
    """Regresion: el brazo B de imitacion paso learning_rate='5e-05' como texto y PPO no arrancaba."""
    import yaml

    with open("configs/criterios.yaml") as f:
        criteria = yaml.safe_load(f)
    for name, contrast in criteria["contrastes"].items():
        for arm_id, arm in contrast["brazos"].items():
            for key, value in (arm.get("ppo") or {}).items():
                # el supervisor los pasa como texto clave=valor; entrenar_ppo.py los lee con parse_value
                assert isinstance(parse_value(str(value)), (int, float)), f"{name}/{arm_id}: {key}={value!r}"
