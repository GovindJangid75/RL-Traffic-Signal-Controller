"""
Train Script v2
===============
Supports both Q-Learning and DQN agents.
Usage:
    python train.py --agent qlearn --difficulty Hard --episodes 1000
    python train.py --agent dqn    --difficulty Hard --episodes 500
"""

import argparse
import numpy as np
import sys
import os
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(__file__))

from env.traffic_env import TrafficEnv
from agent.q_learning import QLearningAgent


def train(
    difficulty        : str = "Medium",
    n_episodes        : int = 1000,
    max_steps         : int = 100,
    agent_type        : str = "qlearn",
    save_path         : str = "q_table.npy",
    verbose           : bool = True,
    progress_callback : Optional[Callable] = None,
    enable_weather    : bool = True,
    enable_time       : bool = True,
    lr                : float = 0.1,
    gamma             : float = 0.95,
    eps_end           : float = 0.05,
):
    env = TrafficEnv(
        difficulty=difficulty,
        max_steps=max_steps,
        enable_weather=enable_weather,
        enable_time_patterns=enable_time,
    )

    if agent_type == "dqn":
        try:
            from agent.dqn_agent import DQNAgent
            agent = DQNAgent(
                state_dim=4,
                action_dim=2,
                lr=lr,
                gamma=gamma,
                epsilon_end=eps_end,
            )
            use_dqn = True
        except (ImportError, RuntimeError) as e:
            print(f"DQN unavailable ({e}), falling back to Q-Learning.")
            agent_type = "qlearn"
            use_dqn    = False
    else:
        use_dqn = False

    if not use_dqn:
        agent = QLearningAgent(
            state_space_size  = env.state_space_size,
            action_space_size = env.action_space_size,
            learning_rate     = lr,
            discount_factor   = gamma,
            epsilon_end       = eps_end,
        )

    if verbose:
        print(f"\n{'='*56}")
        print(f"  🚦 Traffic RL v2 — {agent_type.upper()} | {difficulty} | {n_episodes} ep")
        print(f"  Weather: {enable_weather}  Time patterns: {enable_time}")
        print(f"{'='*56}\n")

    best_reward = -np.inf

    for episode in range(1, n_episodes + 1):
        state      = env.reset()
        ep_reward  = 0.0
        ep_cleared = 0
        ep_emit    = 0.0
        ep_losses  = []

        for _ in range(max_steps):
            if use_dqn:
                action = agent.choose_action(env.vehicles.copy())
            else:
                action = agent.choose_action(TrafficEnv.state_to_index(state))

            next_state, reward, done, info = env.step(action)
            ep_reward  += reward
            ep_cleared += info["cleared"]
            ep_emit    += info["emissions"]

            if use_dqn:
                agent.store(env.vehicles.copy(), action, reward,
                            env.vehicles.copy(), done)
                loss = agent.learn()
                if loss > 0:
                    ep_losses.append(loss)
            else:
                sidx  = TrafficEnv.state_to_index(state)
                nsidx = TrafficEnv.state_to_index(next_state)
                agent.update(sidx, action, reward, nsidx, done)

            state = next_state
            if done:
                break

        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        if use_dqn:
            agent.record_episode(ep_reward, ep_cleared, ep_emit, mean_loss)
        else:
            agent.record_episode(ep_reward, ep_cleared, ep_emit)

        agent.decay_epsilon()

        if ep_reward > best_reward:
            best_reward = ep_reward

        if progress_callback:
            progress_callback(episode, n_episodes, agent.get_training_stats())

        if verbose and episode % 100 == 0:
            stats = agent.get_training_stats()
            print(
                f"  Ep {episode:>5}/{n_episodes} | "
                f"ε={stats['current_epsilon']:.3f} | "
                f"Avg R: {stats['recent_mean_reward']:>8.2f} | "
                f"Cleared: {stats['recent_mean_cleared']:>6.1f}"
            )

    # Save
    if use_dqn:
        agent.save(save_path.replace(".npy", ".pt"))
    else:
        agent.save_q_table(save_path)

    if verbose:
        stats = agent.get_training_stats()
        print(f"\n{'='*56}")
        print(f"  ✅ Done! Best: {best_reward:.2f} | Mean: {stats['mean_reward']:.2f}")
        print(f"{'='*56}\n")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",      default="qlearn", choices=["qlearn", "dqn"])
    parser.add_argument("--difficulty", default="Medium", choices=["Easy","Medium","Hard"])
    parser.add_argument("--episodes",   type=int, default=1000)
    parser.add_argument("--steps",      type=int, default=100)
    parser.add_argument("--save",       default="q_table.npy")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--no-time",    action="store_true")
    args = parser.parse_args()

    train(
        difficulty     = args.difficulty,
        n_episodes     = args.episodes,
        max_steps      = args.steps,
        agent_type     = args.agent,
        save_path      = args.save,
        enable_weather = not args.no_weather,
        enable_time    = not args.no_time,
    )