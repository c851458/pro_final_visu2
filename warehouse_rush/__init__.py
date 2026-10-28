from gymnasium.envs.registration import register

from warehouse_rush.env import WarehouseRushEnv
from warehouse_rush.env_v1 import WarehouseRushV1Env
from warehouse_rush.agent import QLearningAgent

register(id="WarehouseRush-v0", entry_point="warehouse_rush.env:WarehouseRushEnv")
register(id="WarehouseRush-v1", entry_point="warehouse_rush.env_v1:WarehouseRushV1Env")

__all__ = ["WarehouseRushEnv", "WarehouseRushV1Env", "QLearningAgent"]
