import warnings

from gymnasium.envs.registration import register

from warehouse_rush.env import WarehouseRushEnv
from warehouse_rush.env_v1 import WarehouseRushV1Env
from warehouse_rush.env_v2 import WarehouseRushV2Env
from warehouse_rush.agent import QLearningAgent

register(id="WarehouseRush-v0", entry_point="warehouse_rush.env:WarehouseRushEnv")
register(id="WarehouseRush-v1", entry_point="warehouse_rush.env_v1:WarehouseRushV1Env")
register(id="WarehouseRush-v2", entry_point="warehouse_rush.env_v2:WarehouseRushV2Env")

# v1 se conserva a proposito: los modelos historicos se evaluan con sus reglas originales
warnings.filterwarnings("ignore", message=r".*WarehouseRush-v1 is out of date.*")

__all__ = ["WarehouseRushEnv", "WarehouseRushV1Env", "WarehouseRushV2Env", "QLearningAgent"]
