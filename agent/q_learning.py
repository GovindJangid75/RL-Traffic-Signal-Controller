"""
Q-Learning Agent — Tabular RL for Traffic Signal Control.

Algorithm  : Q-learning (off-policy TD control)
Policy     : ε-greedy with linear decay
Q-update   : Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional


class QLearningAgent:
    """
    Tabular Q-Learning agent for the traffic signal control task.

    Parameters
    ----------
    state_space_size  : int   — total discrete states (256 for 4 lanes × 4 bins)
    action_space_size : int   — number of actions (2)
    learning_rate     : float — α, step size for Q-update
    discount_factor   : float — γ, importance of future rewards
    epsilon_start     : float — initial exploration rate
    epsilon_end       : float — minimum exploration rate
    epsilon_decay     : float — per-episode multiplicative decay
    """

    def __init__(
        self,
        state_space_size : int   = 256,
        action_space_size: int   = 2,
        learning_rate    : float = 0.1,
        discount_factor  : float = 0.95,
        epsilon_start    : float = 1.0,
        epsilon_end      : float = 0.05,
        epsilon_decay    : float = 0.995,
    ):
        self.state_space_size  = state_space_size
        self.action_space_size = action_space_size
        self.alpha             = learning_rate
        self.gamma             = discount_factor
        self.epsilon           = epsilon_start
        self.epsilon_end       = epsilon_end
        self.epsilon_decay     = epsilon_decay

        # ── Q-table: rows = states, columns = actions ─────────────────────
        # Initialised with small random values to break symmetry
        self.q_table: np.ndarray = np.random.uniform(
            low=-0.01, high=0.01,
            size=(state_space_size, action_space_size)
        )

        # ── Training history ──────────────────────────────────────────────
        self.episode_rewards  : List[float] = []
        self.episode_epsilons : List[float] = []
        self.episode_cleared  : List[int]   = []
        self.episode_emissions: List[float] = []

    # ── Policy ────────────────────────────────────────────────────────────────

    def choose_action(self, state_index: int, greedy: bool = False) -> int:
        """
        Select an action using ε-greedy policy.

        Parameters
        ----------
        state_index : int  — flat index into Q-table
        greedy      : bool — if True, always exploit (used during evaluation)

        Returns
        -------
        action : int (0 or 1)
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)   # explore
        return int(np.argmax(self.q_table[state_index]))            # exploit

    # ── Q-Update ──────────────────────────────────────────────────────────────

    def update(
        self,
        state_index     : int,
        action          : int,
        reward          : float,
        next_state_index: int,
        done            : bool,
    ) -> float:
        """
        Apply the Bellman update to the Q-table.

        Returns
        -------
        td_error : float — temporal difference error magnitude
        """
        current_q = self.q_table[state_index, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_index])

        td_error = target - current_q
        self.q_table[state_index, action] += self.alpha * td_error
        return abs(td_error)

    # ── Epsilon Decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Reduce epsilon after each episode (explore less over time)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_q_table(self, path: str = "q_table.npy"):
        """Persist the Q-table to disk."""
        np.save(path, self.q_table)

    def load_q_table(self, path: str = "q_table.npy"):
        """Load a previously trained Q-table from disk."""
        self.q_table = np.load(path)

    # ── Statistics ────────────────────────────────────────────────────────────

    def record_episode(self, total_reward: float, total_cleared: int,
                       total_emissions: float = 0.0):
        """Store per-episode metrics for later analysis."""
        self.episode_rewards.append(total_reward)
        self.episode_epsilons.append(self.epsilon)
        self.episode_cleared.append(total_cleared)
        self.episode_emissions.append(total_emissions)

    def get_training_stats(self) -> Dict:
        """Return a dictionary of training statistics."""
        if not self.episode_rewards:
            return {}
        rewards = np.array(self.episode_rewards)
        window  = min(50, len(rewards))
        return {
            "total_episodes"      : len(rewards),
            "mean_reward"         : float(np.mean(rewards)),
            "best_reward"         : float(np.max(rewards)),
            "recent_mean_reward"  : float(np.mean(rewards[-window:])),
            "current_epsilon"     : round(self.epsilon, 4),
            "total_cleared"       : int(sum(self.episode_cleared)),
            "recent_mean_cleared" : float(np.mean(self.episode_cleared[-window:])),
            "total_emissions"     : float(sum(self.episode_emissions)),
            "recent_mean_emit"    : float(np.mean(self.episode_emissions[-window:])),
        }

    def get_policy_summary(self) -> np.ndarray:
        """Return greedy policy: best action for every state."""
        return np.argmax(self.q_table, axis=1)