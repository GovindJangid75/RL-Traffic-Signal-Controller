"""
FastAPI Backend v2
==================
REST API for the Traffic RL system.

Endpoints:
  POST /train           — start training job (background)
  GET  /train/status    — training progress
  POST /sim/reset       — reset simulation
  POST /sim/step        — advance one step
  POST /sim/run/{n}     — run N steps
  GET  /sim/state       — current state
  GET  /stats           — agent training stats
  GET  /health          — health check

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import numpy as np
import threading
import time

from env.traffic_env import TrafficEnv
from agent.q_learning import QLearningAgent
from train import train as run_training

app = FastAPI(
    title="🚦 Traffic RL API",
    description="AI-powered traffic signal control — REST interface",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "agent"          : None,
    "env"            : None,
    "sim_state"      : None,
    "sim_step"       : 0,
    "sim_done"       : False,
    "sim_rewards"    : [],
    "sim_cleared"    : [],
    "sim_actions"    : [],
    "sim_history"    : [],
    "training_running": False,
    "training_progress": 0,
    "training_total"   : 0,
    "training_stats"   : {},
    "last_info"        : {},
}
_lock = threading.Lock()


# ── Auto-load pre-trained Q-table at startup ───────────────────────────────────
def _try_load_pretrained():
    """If q_table.npy exists next to this file (or in the project root),
    load it so /reset works immediately without callers having to /train first."""
    root = os.path.dirname(os.path.dirname(__file__))
    q_path = os.path.join(root, "q_table.npy")
    if not os.path.exists(q_path):
        return
    try:
        # Create a default env just to get the space sizes
        default_env = TrafficEnv()
        agent = QLearningAgent(
            state_space_size  = default_env.state_space_size,
            action_space_size = default_env.action_space_size,
        )
        agent.load_q_table(q_path)
        agent.epsilon = agent.epsilon_end  # fully greedy for inference
        with _lock:
            _state["agent"] = agent
            _state["env"]   = default_env
        print(f"[startup] Loaded pre-trained Q-table from {q_path}")
    except Exception as exc:
        print(f"[startup] Could not load q_table.npy: {exc}")


_try_load_pretrained()


# ── Request/Response models ───────────────────────────────────────────────────
class TrainRequest(BaseModel):
    difficulty     : str   = "Medium"
    n_episodes     : int   = 500
    max_steps      : int   = 100
    agent_type     : str   = "qlearn"
    enable_weather : bool  = True
    enable_time    : bool  = True
    lr             : float = 0.1
    gamma          : float = 0.95
    eps_end        : float = 0.05

class SimResetRequest(BaseModel):
    difficulty     : str  = "Medium"
    max_steps      : int  = 100
    enable_weather : bool = True
    enable_time    : bool = True


# ── Background training ───────────────────────────────────────────────────────
def _train_background(req: TrainRequest):
    def progress_cb(ep, total, stats):
        with _lock:
            _state["training_progress"] = ep
            _state["training_total"]    = total
            _state["training_stats"]    = stats

    with _lock:
        _state["training_running"] = True

    agent = run_training(
        difficulty        = req.difficulty,
        n_episodes        = req.n_episodes,
        max_steps         = req.max_steps,
        agent_type        = req.agent_type,
        enable_weather    = req.enable_weather,
        enable_time       = req.enable_time,
        lr                = req.lr,
        gamma             = req.gamma,
        eps_end           = req.eps_end,
        verbose           = False,
        progress_callback = progress_cb,
    )

    with _lock:
        _state["agent"]            = agent
        _state["training_running"] = False
        _state["training_stats"]   = agent.get_training_stats()
        # Auto-init env after training
        _state["env"] = TrafficEnv(
            difficulty=req.difficulty,
            max_steps=req.max_steps,
            enable_weather=req.enable_weather,
            enable_time_patterns=req.enable_time,
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    with _lock:
        if _state["training_running"]:
            raise HTTPException(409, "Training already in progress")
    background_tasks.add_task(_train_background, req)
    return {"message": "Training started", "config": req.dict()}


@app.get("/train/status")
def training_status():
    with _lock:
        return {
            "running"   : _state["training_running"],
            "progress"  : _state["training_progress"],
            "total"     : _state["training_total"],
            "pct"       : round(100 * _state["training_progress"] /
                                max(_state["training_total"], 1), 1),
            "stats"     : _state["training_stats"],
        }


@app.post("/sim/reset")
def sim_reset(req: SimResetRequest):
    with _lock:
        if _state["agent"] is None:
            raise HTTPException(400, "No trained agent. Call /train first.")
        _state["env"] = TrafficEnv(
            difficulty=req.difficulty,
            max_steps=req.max_steps,
            enable_weather=req.enable_weather,
            enable_time_patterns=req.enable_time,
        )
        state = _state["env"].reset()
        _state.update({
            "sim_state"  : state,
            "sim_step"   : 0,
            "sim_done"   : False,
            "sim_rewards": [],
            "sim_cleared": [],
            "sim_actions": [],
            "sim_history": [],
            "last_info"  : {},
        })
    return {"message": "Simulation reset", "state": list(state)}


@app.post("/sim/step")
def sim_step():
    with _lock:
        if _state["env"] is None or _state["sim_state"] is None:
            raise HTTPException(400, "Call /sim/reset first.")
        if _state["sim_done"]:
            raise HTTPException(400, "Episode done. Call /sim/reset.")

        agent = _state["agent"]
        env   = _state["env"]
        state = _state["sim_state"]

        # Choose action
        if hasattr(agent, 'store'):   # DQN
            action = agent.choose_action(env.vehicles.copy(), greedy=True)
        else:                          # Q-Learning
            action = agent.choose_action(TrafficEnv.state_to_index(state), greedy=True)

        next_state, reward, done, info = env.step(action)

        _state["sim_state"] = next_state
        _state["sim_step"] += 1
        _state["sim_done"]  = done
        _state["sim_rewards"].append(reward)
        _state["sim_cleared"].append(info["cleared"])
        _state["sim_actions"].append(action)
        _state["last_info"]  = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                 for k, v in info.items()}

        step_rec = {
            "step"    : _state["sim_step"],
            "action"  : action,
            "reward"  : round(reward, 3),
            "cleared" : info["cleared"],
            "waiting" : info["waiting"],
            "weather" : info["weather"],
            "hour"    : info["hour"],
            "vehicles": info["vehicles"].tolist(),
        }
        _state["sim_history"].append(step_rec)

    return {
        "step"          : step_rec["step"],
        "action"        : step_rec["action"],
        "reward"        : step_rec["reward"],
        "cleared"       : step_rec["cleared"],
        "waiting"       : step_rec["waiting"],
        "weather"       : step_rec["weather"],
        "hour"          : step_rec["hour"],
        "vehicles"      : step_rec["vehicles"],
        "total_reward"  : round(sum(_state["sim_rewards"]), 2),
        "total_cleared" : sum(_state["sim_cleared"]),
        "done"          : done,
    }


@app.post("/sim/run/{n_steps}")
def sim_run_n(n_steps: int):
    results = []
    for _ in range(min(n_steps, 200)):
        with _lock:
            if _state["sim_done"]:
                break
        results.append(sim_step())
    return {"steps_run": len(results), "results": results}


@app.get("/sim/state")
def sim_state():
    with _lock:
        if _state["env"] is None:
            raise HTTPException(400, "No simulation running.")
        env = _state["env"]
        return {
            "step"         : _state["sim_step"],
            "done"         : _state["sim_done"],
            "vehicles"     : env.vehicles.tolist(),
            "action"       : env.current_action,
            "weather"      : env.current_weather,
            "hour"         : env.current_hour,
            "total_reward" : round(sum(_state["sim_rewards"]), 2),
            "total_cleared": sum(_state["sim_cleared"]),
            "history"      : _state["sim_history"][-20:],
        }


@app.get("/stats")
def get_stats():
    with _lock:
        agent = _state["agent"]
        if agent is None:
            return {"message": "No trained agent yet."}
        return agent.get_training_stats()


@app.get("/sim/history")
def sim_history():
    with _lock:
        return {
            "history"   : _state["sim_history"],
            "rewards"   : _state["sim_rewards"],
            "cleared"   : _state["sim_cleared"],
            "actions"   : _state["sim_actions"],
        }


# ── OpenEnv Compatible Endpoints ─────────────────────────────

@app.post("/reset")
def openenv_reset():
    """OpenEnv /reset — works even before /train by auto-initialising a default agent."""
    with _lock:
        # Initialise agent from disk if not already loaded
        if _state["agent"] is None:
            root   = os.path.dirname(os.path.dirname(__file__))
            q_path = os.path.join(root, "q_table.npy")
            default_env = TrafficEnv()
            agent = QLearningAgent(
                state_space_size  = default_env.state_space_size,
                action_space_size = default_env.action_space_size,
            )
            if os.path.exists(q_path):
                try:
                    agent.load_q_table(q_path)
                    agent.epsilon = agent.epsilon_end  # fully greedy
                except Exception as e:
                    print(f"[reset] q_table.npy load failed ({e}), using fresh agent")
            _state["agent"] = agent

        # (Re-)create the environment
        _state["env"] = TrafficEnv()
        state = _state["env"].reset()
        _state.update({
            "sim_state"  : state,
            "sim_step"   : 0,
            "sim_done"   : False,
            "sim_rewards": [],
            "sim_cleared": [],
            "sim_actions": [],
            "sim_history": [],
            "last_info"  : {},
        })
    return {"message": "Simulation reset", "state": list(state)}


@app.post("/step")
def openenv_step():
    return sim_step()


@app.get("/state")
def openenv_state():
    return sim_state()