---
title: Traffic AI
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: RL-based Traffic Control Environment using OpenEnv
license: mit
---

# 🚦 AI Traffic Control System — v2

> **Full-stack Reinforcement Learning** traffic signal controller with Q-Learning, real traffic patterns, weather simulation, and a FastAPI REST backend compatible with OpenEnv.

---

## 🌐 OpenEnv API Endpoints

```
POST /reset   — Reset simulation episode (called first by checker)
POST /step    — Advance 1 simulation step
GET  /state   — Get current environment state
```

## 🔧 Full API Reference

```
POST /train           — Start training job (background)
GET  /train/status    — Live training progress & stats
POST /sim/reset       — Reset simulation episode
POST /sim/step        — Advance 1 step
POST /sim/run/{n}     — Run N steps
GET  /sim/state       — Current state snapshot
GET  /sim/history     — Full step history
GET  /stats           — Agent training stats
GET  /health          — Health check
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

---

## 📊 Difficulty Levels

| Level  | Arrival λ | Max Vehicles | Notes                  |
| ------ | --------- | ------------ | ---------------------- |
| Easy   | 0.3/step  | 10           | Light traffic          |
| Medium | 0.6/step  | 15           | Realistic city traffic |
| Hard   | 1.0/step  | 20           | Rush hour + weather    |

---

Built with ❤️ — Traffic RL v2
