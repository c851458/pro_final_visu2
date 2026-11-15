import warnings

from gymnasium.envs.registration import register

from warehouse_rush.env import WarehouseRushEnv
from warehouse_rush.env_v1 import WarehouseRushV1Env
from warehouse_rush.env_v2 import WarehouseRushV2Env
from warehouse_rush.env_v3 import WarehouseRushV3Env
from warehouse_rush.agent import QLearningAgent

register(id="WarehouseRush-v0", entry_point="warehouse_rush.env:WarehouseRushEnv")
register(id="WarehouseRush-v1", entry_point="warehouse_rush.env_v1:WarehouseRushV1Env")
register(id="WarehouseRush-v2", entry_point="warehouse_rush.env_v2:WarehouseRushV2Env")
register(id="WarehouseRush-v3", entry_point="warehouse_rush.env_v3:WarehouseRushV3Env")

# v1 y v2 se conservan a proposito: los modelos historicos se evaluan con sus reglas originales
warnings.filterwarnings("ignore", message=r".*WarehouseRush-v[12] is out of date.*")

__all__ = ["WarehouseRushEnv", "WarehouseRushV1Env", "WarehouseRushV2Env", "WarehouseRushV3Env", "QLearningAgent"]
