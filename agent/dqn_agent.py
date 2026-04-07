"""
Deep Q-Network (DQN) Agent
===========================
Replaces Q-table with a neural network.
Features:
  - Experience Replay (deque buffer)
  - Target Network (soft update every N steps)
  - Huber loss for stability
  - Falls back gracefully if torch not available
"""

import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Neural Network ────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """3-layer MLP: state_dim → 128 → 64 → action_dim"""
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
            )

        def forward(self, x):
            return self.net(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer."""
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for traffic signal control.

    State representation: raw vehicle counts [N, S, E, W] (continuous)
    Actions: 0 = NS green, 1 = EW green
    """

    def __init__(
        self,
        state_dim        : int   = 4,
        action_dim       : int   = 2,
        lr               : float = 1e-3,
        gamma            : float = 0.95,
        epsilon_start    : float = 1.0,
        epsilon_end      : float = 0.05,
        epsilon_decay    : float = 0.995,
        batch_size       : int   = 64,
        buffer_capacity  : int   = 10_000,
        target_update_freq: int  = 200,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not installed. Run: pip install torch --break-system-packages"
            )

        self.state_dim          = state_dim
        self.action_dim         = action_dim
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.agent_type         = "DQN"
        self._step              = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online and target networks
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.HuberLoss()
        self.buffer    = ReplayBuffer(buffer_capacity)

        # Training history
        self.episode_rewards  : List[float] = []
        self.episode_cleared  : List[int]   = []
        self.episode_losses   : List[float] = []
        self.episode_emissions: List[float] = []

    def _state_to_tensor(self, state) -> "torch.Tensor":
        """Convert raw vehicle counts to normalised float tensor."""
        arr = np.array(state, dtype=np.float32) / 20.0   # normalise by max vehicles
        return torch.FloatTensor(arr).unsqueeze(0).to(self.device)

    def choose_action(self, state, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_vals = self.online_net(self._state_to_tensor(state))
        return int(q_vals.argmax().item())

    def store(self, state, action, reward, next_state, done):
        """Push transition to replay buffer."""
        s  = np.array(state,      dtype=np.float32) / 20.0
        ns = np.array(next_state, dtype=np.float32) / 20.0
        self.buffer.push(s, action, reward, ns, done)

    def learn(self) -> float:
        """Sample a mini-batch and perform one gradient update. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q
        current_q = self.online_net(states).gather(1, actions)

        # Target Q (using target network)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "dqn_model.pt"):
        torch.save({
            "online_net" : self.online_net.state_dict(),
            "target_net" : self.target_net.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "epsilon"    : self.epsilon,
            "step"       : self._step,
        }, path)

    def load(self, path: str = "dqn_model.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self._step   = ckpt["step"]

    def record_episode(self, total_reward: float, total_cleared: int,
                       total_emissions: float = 0.0, mean_loss: float = 0.0):
        self.episode_rewards.append(total_reward)
        self.episode_cleared.append(total_cleared)
        self.episode_emissions.append(total_emissions)
        self.episode_losses.append(mean_loss)

    def get_training_stats(self) -> Dict:
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
            "mean_loss"           : float(np.mean(self.episode_losses[-window:])) if self.episode_losses else 0.0,
            "agent_type"          : self.agent_type,
            "device"              : str(self.device),
        }

    # Compatibility shim so app.py can call agent.choose_action(state_idx)
    def __call__(self, state, greedy=False):
        return self.choose_action(state, greedy)