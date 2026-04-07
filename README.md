# 🚦 AI Traffic Control System — v2

> **Full-stack Reinforcement Learning** traffic signal controller with DQN, real traffic patterns, weather simulation, FastAPI backend, and animated Streamlit dashboard.

---

## 🆕 What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| RL Algorithm | Q-Learning only | Q-Learning + **DQN** |
| State space | 256 discrete | Continuous (DQN) |
| Vehicle types | Cars only | Car, bike, bus, truck |
| Weather | ❌ | ☀️ 🌧️ 🌫️ ⛈️ |
| Time patterns | ❌ | 24-hour rush hour curves |
| Emergency vehicles | ❌ | ✅ Random events |
| Training chart | Post-training only | **Live real-time** |
| Intersection | Static matplotlib | **Animated JS canvas** |
| Comparison | ❌ | **AI vs Fixed-Timer** |
| Backend API | ❌ | **FastAPI REST** |
| Deployment | Single container | **Docker Compose** |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│            Streamlit Dashboard (app.py)                │
│  Tab1: Train │ Tab2: Simulate │ Tab3: Compare │ Tab4: Inspect │
└───────────────────┬────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       │                         │
┌──────▼──────┐         ┌────────▼────────┐
│  Q-Learning  │         │   DQN Agent     │
│  (q_table)  │         │  (neural net)   │
└──────┬──────┘         └────────┬────────┘
       └────────────┬────────────┘
                    │ action
         ┌──────────▼──────────┐
         │  Traffic Env v2     │
         │  Vehicle types      │
         │  Weather effects    │
         │  Time-of-day        │
         │  Emergency events   │
         └─────────────────────┘

FastAPI (api/server.py) — REST interface to same agents/env
```

---

## 📁 Project Structure

```
traffic-rl-v2/
├── env/
│    ├── __init__.py
│    └── traffic_env.py       # Enhanced env — weather, vehicles, time
├── agent/
│    ├── __init__.py
│    ├── q_learning.py        # Tabular Q-Learning
│    └── dqn_agent.py         # Deep Q-Network (PyTorch)
├── api/
│    ├── __init__.py
│    └── server.py            # FastAPI REST backend
├── app.py                    # Streamlit dashboard (4 tabs)
├── train.py                  # CLI training — qlearn or dqn
├── requirements.txt
├── Dockerfile                # Multi-stage (app + api targets)
├── docker-compose.yml        # App + API services
└── README.md
```

---

## 🧠 RL Details

### Q-Learning
```
State  : (n_bin, s_bin, e_bin, w_bin) → 256 discrete states
Action : 0 = NS Green, 1 = EW Green
Reward : +2×cleared_weight − 0.5×waiting − 0.05×emissions + 5×emergency_clear
Update : Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]
Policy : ε-greedy with exponential decay
```

### DQN
```
State  : raw [N, S, E, W] vehicle counts (continuous, normalised)
Network: Linear(4→128) → ReLU → Linear(128→64) → ReLU → Linear(64→2)
Loss   : Huber loss
Optim  : Adam (lr=1e-3)
Buffer : Replay buffer (10k transitions)
Target : Hard update every 200 steps
```

### Vehicle Types
| Type  | Clear Rate | Weight | Emission | Prob |
|-------|-----------|--------|----------|------|
| Car   | 65%       | 1.0×   | 1.0      | 55%  |
| Bike  | 85%       | 0.3×   | 0.2      | 20%  |
| Bus   | 40%       | 3.0×   | 4.0      | 10%  |
| Truck | 35%       | 2.5×   | 5.0      | 15%  |

### Weather
| Condition | Clear Multiplier | Arrival Multiplier |
|-----------|-----------------|-------------------|
| ☀️ Clear  | 1.00            | 1.00              |
| 🌧️ Rain  | 0.70            | 1.20              |
| 🌫️ Fog  | 0.60            | 0.90              |
| ⛈️ Storm | 0.40            | 1.50              |

---

## 🚀 Run Instructions

### Option A — Docker Compose (Recommended)
```bash
cd traffic-rl-v2

# Build both services
docker-compose build

# Start everything
docker-compose up

# Dashboard → http://localhost:8501
# API docs  → http://localhost:8000/docs
```

### Option B — Local Python
```bash
pip install -r requirements.txt
# For DQN support:
pip install torch

# Run dashboard
streamlit run app.py

# Or train from CLI
python train.py --agent qlearn --difficulty Hard --episodes 1000
python train.py --agent dqn    --difficulty Medium --episodes 500

# Or run API
uvicorn api.server:app --reload --port 8000
```

---

## 🌐 API Reference

```
POST /train            — Start training job (background)
GET  /train/status     — Live progress & stats
POST /sim/reset        — Reset simulation episode
POST /sim/step         — Advance 1 step
POST /sim/run/{n}      — Run N steps
GET  /sim/state        — Current state snapshot
GET  /sim/history      — Full step history
GET  /stats            — Agent training stats
GET  /health           — Health check
```

Auto-generated docs: **http://localhost:8000/docs**

---

## 📊 Difficulty Levels

| Level  | Arrival λ | Max Vehicles | Notes                    |
|--------|-----------|--------------|--------------------------|
| Easy   | 0.3/step  | 10           | Light traffic            |
| Medium | 0.6/step  | 15           | Realistic city traffic   |
| Hard   | 1.0/step  | 20           | Rush hour + weather      |

---

## 🐳 Docker Notes

```bash
# View running services
docker-compose ps

# Logs
docker-compose logs app
docker-compose logs api

# Stop
docker-compose down
```

---

## 🔧 DQN Installation

PyTorch is optional. Install separately:
```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

Built with ❤️ — Traffic RL v2