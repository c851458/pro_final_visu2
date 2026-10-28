"""Agente Q-learning tabular para Warehouse Rush."""

import pickle

import numpy as np


class QLearningAgent:
    def __init__(self, obs_dims, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, seed=None):
        self.q = np.zeros((*obs_dims, n_actions), dtype=np.float32)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[tuple(obs)]))

    def update(self, obs, action, reward, next_obs, terminated):
        target = reward if terminated else reward + self.gamma * np.max(self.q[tuple(next_obs)])
        idx = (*obs, action)
        self.q[idx] += self.alpha * (target - self.q[idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q = pickle.load(f)
