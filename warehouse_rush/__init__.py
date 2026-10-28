from gymnasium.envs.registration import register

from warehouse_rush.env import WarehouseRushEnv
from warehouse_rush.agent import QLearningAgent

register(id="WarehouseRush-v0", entry_point="warehouse_rush.env:WarehouseRushEnv")

__all__ = ["WarehouseRushEnv", "QLearningAgent"]
