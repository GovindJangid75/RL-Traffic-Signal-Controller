"""
Streamlit Dashboard v2 — AI Traffic Control System
====================================================
New features:
  - Agent selector: Q-Learning vs DQN
  - Live training chart (real-time updates while training)
  - Animated intersection (JS canvas animation)
  - Comparison mode: AI vs Fixed-timer side-by-side
  - Vehicle type breakdown
  - Weather & time-of-day display
  - Emission tracking
  - Dark industrial aesthetic
"""

import streamlit as st
import numpy as np
import time
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from env.traffic_env import TrafficEnv, WEATHER_EFFECTS, TIME_PATTERN
from agent.q_learning import QLearningAgent
from train import train as run_training

# ════════════════════════════════════════════════════════
#  Page Config
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🚦 Traffic RL v2",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
#  CSS — Dark Industrial Theme
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
  }
  .main { background: #0a0a0f; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #f0c040 0%, #e05820 50%, #c0392b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
  }
  .hero-sub {
    color: #555;
    font-size: 0.78rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0c040;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid #222;
    padding-bottom: 6px;
    margin: 1.4rem 0 0.8rem 0;
  }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }
  .badge-green  { background: #0d2e1a; color: #2ecc71; border: 1px solid #1a5c34; }
  .badge-blue   { background: #0d1e2e; color: #3498db; border: 1px solid #1a3c5c; }
  .badge-red    { background: #2e0d0d; color: #e74c3c; border: 1px solid #5c1a1a; }
  .badge-amber  { background: #2e220d; color: #f0c040; border: 1px solid #5c440d; }
  .badge-gray   { background: #1a1a1a; color: #888; border: 1px solid #333; }

  .kpi-box {
    background: #111118;
    border: 1px solid #222;
    border-top: 2px solid #f0c040;
    border-radius: 6px;
    padding: 14px 16px;
    text-align: center;
  }
  .kpi-val  { font-size: 1.8rem; font-weight: 600; color: #f0c040; font-family: 'Syne', sans-serif; }
  .kpi-lbl  { font-size: 0.68rem; color: #555; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }

  .stButton > button {
    background: #1a1a1a;
    color: #f0c040;
    border: 1px solid #333;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    font-weight: 600;
    width: 100%;
    transition: all 0.15s;
  }
  .stButton > button:hover {
    background: #f0c040;
    color: #0a0a0f;
    border-color: #f0c040;
  }
  .weather-panel {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #888;
  }
  .log-box {
    background: #08080e;
    border: 1px solid #1a1a2a;
    border-radius: 4px;
    padding: 10px;
    font-size: 0.72rem;
    color: #3a9e6a;
    font-family: 'JetBrains Mono', monospace;
    max-height: 180px;
    overflow-y: auto;
  }
  /* Hide streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Session State
# ════════════════════════════════════════════════════════
def _init():
    defaults = {
        "agent"           : None,
        "env"             : None,
        "trained"         : False,
        "agent_type"      : "Q-Learning",
        "difficulty"      : "Medium",
        "n_episodes"      : 500,
        "max_steps"       : 100,
        "sim_state"       : None,
        "sim_step"        : 0,
        "sim_rewards"     : [],
        "sim_cleared"     : [],
        "sim_actions"     : [],
        "sim_emissions"   : [],
        "sim_weathers"    : [],
        "sim_hours"       : [],
        "sim_vehicles_hist": [],
        "sim_done"        : False,
        "last_info"       : {},
        "comparison_ai_rewards"   : [],
        "comparison_fixed_rewards": [],
        "comparison_ai_cleared"   : [],
        "comparison_fixed_cleared": [],
        "training_log"    : [],
        "live_rewards"    : [],
        "live_cleared"    : [],
        "enable_weather"  : True,
        "enable_time"     : True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ════════════════════════════════════════════════════════
#  Helpers — Drawing
# ════════════════════════════════════════════════════════
def draw_intersection_animated(vehicles, action, info: Dict = {}) -> str:
    """Return HTML/JS animated canvas intersection."""
    n, s, e, w = [int(v) for v in vehicles]
    ns_green = "true" if action == 0 else "false"
    ew_green = "true" if action == 1 else "false"
    weather  = info.get("weather", "clear")
    hour     = info.get("hour", 12)
    emg      = info.get("emergency", False)
    emg_lane = info.get("emergency_lane", -1)

    weather_label = WEATHER_EFFECTS.get(weather, {}).get("label", "☀️ Clear")

    # Sky tint by hour
    if 6 <= hour < 9 or 17 <= hour < 20:
        sky = "#1a0e05"   # dawn/dusk
    elif 9 <= hour < 17:
        sky = "#080c18"   # day
    else:
        sky = "#050508"   # night

    return f"""
<div style="position:relative;background:{sky};border:1px solid #222;border-radius:8px;overflow:hidden;">
  <canvas id="tc" width="520" height="520" style="display:block;width:100%;"></canvas>
  <div style="position:absolute;top:8px;right:10px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#888;">
    {weather_label} &nbsp;|&nbsp; {hour:02d}:00
    {"&nbsp;|&nbsp;<span style='color:#e74c3c;font-weight:700;'>🚨 EMERGENCY</span>" if emg else ""}
  </div>
</div>
<script>
(function(){{
  const cv = document.getElementById('tc');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W=520, H=520, CX=260, CY=260, RW=80, LL=180;

  const N_CARS={n}, S_CARS={s}, E_CARS={e}, W_CARS={w};
  const NS_GREEN={ns_green}, EW_GREEN={ew_green};
  const EMG={str(emg).lower()}, EMG_LANE={emg_lane};
  const MAX_V=20;

  let tick=0;
  const cars = [];

  // Spawn car objects
  function spawnCars(lane, count) {{
    for(let i=0;i<Math.min(count,8);i++) {{
      const c = {{lane, idx:i, t: Math.random(), speed:0}};
      cars.push(c);
    }}
  }}
  spawnCars(0, N_CARS);
  spawnCars(1, S_CARS);
  spawnCars(2, E_CARS);
  spawnCars(3, W_CARS);

  function carPos(c) {{
    const base = c.idx * 28 + 20;
    switch(c.lane) {{
      case 0: return [CX+18, CY - LL - base + c.t*4];   // North
      case 1: return [CX-18, CY + LL + base - c.t*4];   // South
      case 2: return [CX + LL + base - c.t*4, CY+18];   // East
      case 3: return [CX - LL - base + c.t*4, CY-18];   // West
    }}
  }}

  function drawRoad() {{
    ctx.fillStyle = '#0e0e0e';
    ctx.fillRect(0,0,W,H);

    // Road surface
    ctx.fillStyle='#1c1c1c';
    ctx.fillRect(CX-RW/2, 0, RW, H);
    ctx.fillRect(0, CY-RW/2, W, RW);

    // Intersection box
    ctx.fillStyle='#242424';
    ctx.fillRect(CX-RW/2, CY-RW/2, RW, RW);

    // Lane markings
    ctx.strokeStyle='#333'; ctx.lineWidth=1.5; ctx.setLineDash([12,8]);
    ctx.beginPath(); ctx.moveTo(CX,0); ctx.lineTo(CX, CY-RW/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX,CY+RW/2); ctx.lineTo(CX, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,CY); ctx.lineTo(CX-RW/2, CY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX+RW/2,CY); ctx.lineTo(W, CY); ctx.stroke();
    ctx.setLineDash([]);

    // Stop lines
    ctx.strokeStyle='#fff'; ctx.lineWidth=2.5;
    ctx.beginPath(); ctx.moveTo(CX-RW/2,CY-RW/2-2); ctx.lineTo(CX+RW/2,CY-RW/2-2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX-RW/2,CY+RW/2+2); ctx.lineTo(CX+RW/2,CY+RW/2+2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX-RW/2-2,CY-RW/2); ctx.lineTo(CX-RW/2-2,CY+RW/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX+RW/2+2,CY-RW/2); ctx.lineTo(CX+RW/2+2,CY+RW/2); ctx.stroke();
  }}

  function drawSignals() {{
    const ns = NS_GREEN ? '#2ecc71' : '#e74c3c';
    const ew = EW_GREEN ? '#2ecc71' : '#e74c3c';

    const pulse = NS_GREEN ? 0.7 + 0.3*Math.sin(tick*0.15) : 1.0;
    const pulsew = EW_GREEN ? 0.7 + 0.3*Math.sin(tick*0.15) : 1.0;

    // NS lights
    ctx.globalAlpha = pulse;
    ctx.fillStyle = ns;
    ctx.beginPath(); ctx.arc(CX-RW/2-12, CY-RW/2-12, 7, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CX+RW/2+12, CY+RW/2+12, 7, 0, Math.PI*2); ctx.fill();

    // EW lights
    ctx.globalAlpha = pulsew;
    ctx.fillStyle = ew;
    ctx.beginPath(); ctx.arc(CX+RW/2+12, CY-RW/2-12, 7, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CX-RW/2-12, CY+RW/2+12, 7, 0, Math.PI*2); ctx.fill();
    ctx.globalAlpha = 1.0;
  }}

  function drawCars() {{
    cars.forEach(c => {{
      const [cx,cy] = carPos(c);
      const isMoving = (c.lane<2 && NS_GREEN) || (c.lane>=2 && EW_GREEN);
      const isEmg = EMG && EMG_LANE===c.lane && c.idx===0;

      ctx.save();
      if (c.lane===0||c.lane===1) {{
        // Vertical car
        const color = isEmg ? '#e74c3c' : (isMoving ? '#f0c040' : '#2c4a7c');
        ctx.fillStyle = color;
        ctx.shadowColor = color; ctx.shadowBlur = isMoving ? 6 : 2;
        ctx.fillRect(cx-7, cy-12, 14, 22);
        ctx.fillStyle='#0a0a0f';
        ctx.fillRect(cx-5, cy-10, 10, 6);
        ctx.fillRect(cx-5, cy+4, 10, 5);
        // Wheels
        ctx.fillStyle = '#111';
        ctx.fillRect(cx-9, cy-9, 3, 5);
        ctx.fillRect(cx+6, cy-9, 3, 5);
        ctx.fillRect(cx-9, cy+5, 3, 5);
        ctx.fillRect(cx+6, cy+5, 3, 5);
      }} else {{
        // Horizontal car
        const color = isEmg ? '#e74c3c' : (isMoving ? '#f0c040' : '#2c4a7c');
        ctx.fillStyle = color;
        ctx.shadowColor = color; ctx.shadowBlur = isMoving ? 6 : 2;
        ctx.fillRect(cx-12, cy-7, 22, 14);
        ctx.fillStyle='#0a0a0f';
        ctx.fillRect(cx-10, cy-5, 6, 10);
        ctx.fillRect(cx+4, cy-5, 6, 10);
        ctx.fillStyle = '#111';
        ctx.fillRect(cx-9, cy-9, 5, 3);
        ctx.fillRect(cx+5, cy-9, 5, 3);
        ctx.fillRect(cx-9, cy+6, 5, 3);
        ctx.fillRect(cx+5, cy+6, 5, 3);
      }}
      ctx.restore();
    }});
  }}

  function drawLabels() {{
    ctx.font = '600 13px JetBrains Mono, monospace';
    ctx.textAlign = 'center';

    const barW = 56, barH = 7;
    function bar(x, y, count) {{
      const pct = Math.min(count/MAX_V, 1);
      const color = pct>0.7 ? '#e74c3c' : pct>0.4 ? '#f0a020' : '#2ecc71';
      ctx.fillStyle='#1a1a1a'; ctx.fillRect(x-barW/2, y, barW, barH);
      ctx.fillStyle=color; ctx.fillRect(x-barW/2, y, barW*pct, barH);
      ctx.strokeStyle='#333'; ctx.lineWidth=0.5;
      ctx.strokeRect(x-barW/2, y, barW, barH);
    }}

    [[CX,'N',N_CARS,20],[CX,'S',S_CARS,H-40],[18,'W',W_CARS,CY],[W-20,'E',E_CARS,CY]].forEach(([x,lbl,cnt,y])=>{{
      ctx.fillStyle='#f0c040';
      ctx.fillText(lbl, x, y);
      ctx.fillStyle='#999';
      ctx.fillText(cnt, x, y+16);
    }});

    bar(CX, 42, N_CARS);
    bar(CX, H-52, S_CARS);
  }}

  function frame() {{
    tick++;
    drawRoad();
    drawSignals();
    drawCars();
    drawLabels();
    requestAnimationFrame(frame);
  }}
  frame();
}})();
</script>
"""


def kpi(val, label):
    return f"""<div class="kpi-box">
  <div class="kpi-val">{val}</div>
  <div class="kpi-lbl">{label}</div>
</div>"""


def plot_live_training(rewards, cleared, title="Live Training") -> plt.Figure:
    fig = plt.figure(figsize=(9, 3), facecolor="#0a0a0f")
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    data = [(rewards, "Episode Reward", "#f0c040"),
            (cleared, "Cars Cleared",  "#e05820")]

    for ax, (d, label, color) in zip(axes, data):
        ax.set_facecolor("#111118")
        ax.plot(d, color="#2a2a3a", lw=0.6, alpha=0.5)
        if len(d) >= 20:
            wn = max(10, len(d)//20)
            sm = np.convolve(d, np.ones(wn)/wn, mode='valid')
            ax.plot(range(wn-1, len(d)), sm, color=color, lw=2.0)
        ax.set_title(label, color="#888", fontsize=9, pad=6,
                     fontfamily="monospace")
        ax.tick_params(colors="#333")
        for sp in ax.spines.values():
            sp.set_edgecolor("#222")

    plt.tight_layout(pad=0.5)
    return fig


def plot_sim_metrics(rewards, cleared, actions, emissions) -> plt.Figure:
    fig = plt.figure(figsize=(12, 2.8), facecolor="#0a0a0f")
    gs  = GridSpec(1, 4, figure=fig, wspace=0.4)
    datasets = [
        (rewards,   "Reward/Step",   "#f0c040"),
        (cleared,   "Cleared",       "#2ecc71"),
        (actions,   "Signal (0=NS)", "#3498db"),
        (emissions, "Emissions",     "#e74c3c"),
    ]
    for i, (ax_idx, (d, label, color)) in enumerate(zip(range(4), datasets)):
        ax = fig.add_subplot(gs[0, ax_idx])
        ax.set_facecolor("#111118")
        ax.plot(d, color=color, lw=1.4)
        ax.fill_between(range(len(d)), d, alpha=0.08, color=color)
        ax.set_title(label, color="#888", fontsize=8, pad=4, fontfamily="monospace")
        ax.tick_params(colors="#333", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1a1a")
    plt.tight_layout(pad=0.3)
    return fig


def plot_comparison(ai_r, fixed_r, ai_c, fixed_c) -> plt.Figure:
    fig = plt.figure(figsize=(10, 3.5), facecolor="#0a0a0f")
    gs  = GridSpec(1, 2, figure=fig, wspace=0.4)
    pairs = [
        (ai_r, fixed_r, "Cumulative Reward"),
        (ai_c, fixed_c, "Total Cars Cleared"),
    ]
    for ax_idx, (ai_d, fx_d, label) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, ax_idx])
        ax.set_facecolor("#111118")
        cum_ai = np.cumsum(ai_d)
        cum_fx = np.cumsum(fx_d)
        ax.plot(cum_ai, color="#f0c040", lw=2.0, label="AI Agent")
        ax.plot(cum_fx, color="#e74c3c", lw=2.0, label="Fixed Timer", ls="--")
        ax.set_title(label, color="#888", fontsize=9, pad=6, fontfamily="monospace")
        ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#aaa", fontsize=8)
        ax.tick_params(colors="#333")
        for sp in ax.spines.values():
            sp.set_edgecolor("#222")
    plt.tight_layout(pad=0.5)
    return fig


def plot_time_heatmap() -> plt.Figure:
    """Show traffic multiplier by hour."""
    hours  = list(TIME_PATTERN.keys())
    values = [TIME_PATTERN[h] for h in hours]
    fig, ax = plt.subplots(figsize=(10, 1.8), facecolor="#0a0a0f")
    ax.set_facecolor("#111118")
    colors = ["#e74c3c" if v > 1.3 else "#f0c040" if v > 0.8 else "#2c4a7c" for v in values]
    bars = ax.bar(hours, values, color=colors, width=0.85, edgecolor="#0a0a0f", lw=0.5)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)], color="#555", fontsize=8)
    ax.set_title("Traffic Intensity by Hour", color="#888", fontsize=9, pad=6, fontfamily="monospace")
    ax.tick_params(colors="#333")
    ax.axhline(1.0, color="#333", lw=0.8, ls="--")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a1a")
    plt.tight_layout(pad=0.2)
    return fig


# ════════════════════════════════════════════════════════
#  Fixed-timer baseline for comparison
# ════════════════════════════════════════════════════════
def run_fixed_timer_episode(difficulty, max_steps, fixed_interval=5):
    """Run one episode with fixed timer, return rewards & cleared."""
    env = TrafficEnv(
        difficulty=difficulty, max_steps=max_steps,
        enable_weather=True, enable_time_patterns=True
    )
    state = env.reset()
    rewards, cleared = [], []
    for step in range(max_steps):
        action = (step // fixed_interval) % 2
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        cleared.append(info["cleared"])
        if done:
            break
    return rewards, cleared


# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ CONFIG")

    agent_type = st.selectbox("Agent", ["Q-Learning", "DQN"],
        index=["Q-Learning","DQN"].index(st.session_state["agent_type"]))
    st.session_state["agent_type"] = agent_type

    difficulty = st.selectbox("Difficulty", ["Easy","Medium","Hard"],
        index=["Easy","Medium","Hard"].index(st.session_state["difficulty"]))
    st.session_state["difficulty"] = difficulty

    n_episodes = st.slider("Episodes", 100, 2000,
        value=st.session_state["n_episodes"], step=100)
    st.session_state["n_episodes"] = n_episodes

    max_steps = st.slider("Steps/Episode", 50, 300,
        value=st.session_state["max_steps"], step=50)
    st.session_state["max_steps"] = max_steps

    st.markdown("---")
    st.markdown("### 🌍 Environment")
    enable_weather = st.toggle("Weather Effects", value=st.session_state["enable_weather"])
    enable_time    = st.toggle("Time-of-Day Patterns", value=st.session_state["enable_time"])
    st.session_state["enable_weather"] = enable_weather
    st.session_state["enable_time"]    = enable_time

    st.markdown("---")
    st.markdown("### 🧠 Hyperparameters")
    lr      = st.slider("Learning Rate", 0.001, 0.5, 0.1, 0.001)
    gamma   = st.slider("Discount γ",   0.80, 0.99, 0.95, 0.01)
    eps_end = st.slider("ε minimum",    0.01, 0.20, 0.05, 0.01)

    st.markdown("---")
    st.markdown("### 📚 Algorithm")
    if agent_type == "Q-Learning":
        st.info("**Tabular Q-Learning**\nQ(s,a) ← Q + α[r + γ·maxQ' − Q]\n256 states × 2 actions")
    else:
        st.info("**Deep Q-Network**\nNeural net replaces Q-table\nExperience replay + target net\nHuber loss, Adam optimizer")


# ════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-title">🚦 TRAFFIC RL v2</div>'
    '<div class="hero-sub">AI Traffic Signal Control · Reinforcement Learning · Real-Time Simulation</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════
#  TAB LAYOUT
# ════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🎓 Train", "🎬 Simulate", "⚖️ Compare", "🔬 Inspect"])


# ── TAB 1: TRAINING ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Train Agent</div>', unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        train_btn = st.button(f"🚀 Train {agent_type}", type="primary")
    with col_btn2:
        load_qlearn = st.button("📂 Load Q-Table (.npy)")
    with col_btn3:
        load_dqn = st.button("📂 Load DQN (.pt)")

    # Training time estimate
    est_secs = n_episodes * max_steps * (0.0002 if agent_type == "Q-Learning" else 0.001)
    st.markdown(
        f'<span class="badge badge-gray">Estimated time: ~{est_secs:.0f}s</span>',
        unsafe_allow_html=True
    )

    # ── Time heatmap always visible
    if enable_time:
        st.markdown('<div class="section-header">Traffic Pattern (24h)</div>', unsafe_allow_html=True)
        fig_heat = plot_time_heatmap()
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

    # ── Train ─────────────────────────────────────────────────────────────
    if train_btn:
        st.markdown('<div class="section-header">Live Training Progress</div>', unsafe_allow_html=True)

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        chart_placeholder = st.empty()
        log_placeholder   = st.empty()

        st.session_state["live_rewards"] = []
        st.session_state["live_cleared"] = []

        def progress_callback(ep, total, stats):
            pct = ep / total
            progress_bar.progress(pct)
            status_text.markdown(
                f'<span class="badge badge-amber">EP {ep}/{total}</span> &nbsp;'
                f'<span class="badge badge-gray">ε={stats.get("current_epsilon", 0):.3f}</span> &nbsp;'
                f'<span class="badge badge-green">Avg R={stats.get("recent_mean_reward", 0):.1f}</span>',
                unsafe_allow_html=True
            )
            if ep % 20 == 0:
                r = stats.get("recent_mean_reward", 0)
                c = stats.get("recent_mean_cleared", 0)
                st.session_state["live_rewards"].append(r)
                st.session_state["live_cleared"].append(c)
                if len(st.session_state["live_rewards"]) >= 3:
                    fig_live = plot_live_training(
                        st.session_state["live_rewards"],
                        st.session_state["live_cleared"],
                    )
                    chart_placeholder.pyplot(fig_live, use_container_width=True)
                    plt.close(fig_live)

        with st.spinner("Training in progress…"):
            agent = run_training(
                difficulty        = difficulty,
                n_episodes        = n_episodes,
                max_steps         = max_steps,
                agent_type        = "qlearn" if agent_type == "Q-Learning" else "dqn",
                enable_weather    = enable_weather,
                enable_time       = enable_time,
                lr                = lr,
                gamma             = gamma,
                eps_end           = eps_end,
                verbose           = False,
                progress_callback = progress_callback,
            )

        progress_bar.progress(1.0)
        st.session_state["agent"]   = agent
        st.session_state["trained"] = True
        st.session_state["env"]     = TrafficEnv(
            difficulty=difficulty, max_steps=max_steps,
            enable_weather=enable_weather, enable_time_patterns=enable_time
        )

        stats = agent.get_training_stats()
        st.success(f"✅ Training complete! Best Reward: {stats['best_reward']:.1f}")

        c1, c2, c3, c4 = st.columns(4)
        for col, (v, l) in zip(
            [c1,c2,c3,c4],
            [(stats["total_episodes"],"Episodes"),
             (f"{stats['mean_reward']:.1f}","Mean Reward"),
             (f"{stats['best_reward']:.1f}","Best Reward"),
             (stats["current_epsilon"],"Final ε")]
        ):
            col.markdown(kpi(v, l), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
        fig_train = plot_live_training(
            agent.episode_rewards,
            agent.episode_cleared,
            "Final Training Curves"
        )
        st.pyplot(fig_train, use_container_width=True)
        plt.close(fig_train)

    if load_qlearn and os.path.exists("q_table.npy"):
        agent = QLearningAgent()
        agent.load_q_table("q_table.npy")
        agent.epsilon = 0.0
        st.session_state.update({
            "agent": agent, "trained": True,
            "env": TrafficEnv(difficulty=difficulty, max_steps=max_steps,
                              enable_weather=enable_weather, enable_time_patterns=enable_time)
        })
        st.success("✅ Q-Table loaded!")

    if load_dqn and os.path.exists("dqn_model.pt"):
        try:
            from agent.dqn_agent import DQNAgent
            agent = DQNAgent()
            agent.load("dqn_model.pt")
            agent.epsilon = 0.0
            st.session_state.update({
                "agent": agent, "trained": True, "agent_type": "DQN",
                "env": TrafficEnv(difficulty=difficulty, max_steps=max_steps,
                                  enable_weather=enable_weather, enable_time_patterns=enable_time)
            })
            st.success("✅ DQN model loaded!")
        except Exception as e:
            st.error(f"DQN load failed: {e}")


# ── TAB 2: SIMULATION ─────────────────────────────────────────────────────────
with tab2:
    if not st.session_state["trained"]:
        st.warning("⚠️ Train or load an agent first (Tab 1).")
    else:
        st.markdown('<div class="section-header">Live Simulation</div>', unsafe_allow_html=True)

        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        with bc1: start_btn  = st.button("▶ Step")
        with bc2: auto_btn   = st.button("⚡ Auto 20")
        with bc3: full_btn   = st.button("🏁 Full Ep")
        with bc4: reset_btn  = st.button("↺ Reset")
        with bc5: auto50_btn = st.button("⚡ Auto 50")

        agent = st.session_state["agent"]
        env   = st.session_state["env"]

        def _reset_sim():
            st.session_state.update({
                "sim_state"       : env.reset(),
                "sim_step"        : 0,
                "sim_rewards"     : [],
                "sim_cleared"     : [],
                "sim_actions"     : [],
                "sim_emissions"   : [],
                "sim_weathers"    : [],
                "sim_hours"       : [],
                "sim_vehicles_hist": [],
                "sim_done"        : False,
                "last_info"       : {},
            })

        def _do_step():
            if st.session_state["sim_done"]:
                return
            state = st.session_state["sim_state"]

            if hasattr(agent, 'store'):   # DQN
                action = agent.choose_action(env.vehicles.copy(), greedy=True)
            else:
                action = agent.choose_action(TrafficEnv.state_to_index(state), greedy=True)

            next_state, reward, done, info = env.step(action)
            st.session_state["sim_state"]  = next_state
            st.session_state["sim_step"]  += 1
            st.session_state["sim_done"]   = done
            st.session_state["sim_rewards"].append(reward)
            st.session_state["sim_cleared"].append(info["cleared"])
            st.session_state["sim_actions"].append(action)
            st.session_state["sim_emissions"].append(info["emissions"])
            st.session_state["sim_weathers"].append(info["weather"])
            st.session_state["sim_hours"].append(info["hour"])
            st.session_state["sim_vehicles_hist"].append(env.vehicles.copy())
            st.session_state["last_info"] = info

        if st.session_state["sim_state"] is None:
            _reset_sim()

        if reset_btn: _reset_sim()
        if start_btn: _do_step()
        if auto_btn:
            for _ in range(20):
                _do_step()
                if st.session_state["sim_done"]: break
        if auto50_btn:
            for _ in range(50):
                _do_step()
                if st.session_state["sim_done"]: break
        if full_btn:
            while not st.session_state["sim_done"]:
                _do_step()

        # ── Render ────────────────────────────────────────────────────────
        left_col, right_col = st.columns([1.1, 1.0])

        with left_col:
            html_canvas = draw_intersection_animated(
                env.vehicles, env.current_action,
                st.session_state.get("last_info", {})
            )
            st.components.v1.html(html_canvas, height=560)

        with right_col:
            info = st.session_state.get("last_info", {})
            step = st.session_state["sim_step"]
            rewards  = st.session_state["sim_rewards"]
            cleared  = st.session_state["sim_cleared"]
            weathers = st.session_state["sim_weathers"]
            hours    = st.session_state["sim_hours"]

            # Signal badge
            if env.current_action == 0:
                st.markdown('<span class="badge badge-green">🟢 NORTH-SOUTH GREEN</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-blue">🟢 EAST-WEST GREEN</span>',
                            unsafe_allow_html=True)

            # Weather + emergency
            w_lbl = WEATHER_EFFECTS.get(info.get("weather","clear"), {}).get("label","")
            if info.get("emergency"):
                lane_names = ["North","South","East","West"]
                emg_lane = info.get("emergency_lane", 0)
                st.markdown(f'<span class="badge badge-red">🚨 EMERGENCY — {lane_names[emg_lane]}</span>',
                            unsafe_allow_html=True)
            st.markdown(f'<span class="badge badge-gray">{w_lbl} &nbsp; Hour: {info.get("hour",0):02d}:00</span>',
                        unsafe_allow_html=True)

            st.markdown("")

            # Lane bars
            max_v = env.config["max_vehicles"]
            lane_names = ["⬆ North","⬇ South","➡ East","⬅ West"]
            for ln, cnt in zip(lane_names, env.vehicles):
                pct = min(cnt / max_v, 1.0)
                col_bar = "#e74c3c" if pct > 0.7 else "#f0a020" if pct > 0.4 else "#2ecc71"
                st.markdown(f"**{ln}** — {cnt} vehicles")
                st.progress(pct)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.markdown(kpi(step, "Step"), unsafe_allow_html=True)
            c2.markdown(kpi(f"{sum(rewards):.0f}" if rewards else "0", "Total Reward"), unsafe_allow_html=True)
            c3.markdown(kpi(sum(cleared) if cleared else 0, "Cleared"), unsafe_allow_html=True)

            if rewards:
                st.markdown(
                    f"Last step → R: `{rewards[-1]:.2f}` | Cleared: `{cleared[-1]}` | "
                    f"Emissions: `{st.session_state['sim_emissions'][-1]:.1f}`"
                )

        # ── Charts ────────────────────────────────────────────────────────
        if len(rewards) >= 2:
            st.markdown('<div class="section-header">Simulation Metrics</div>', unsafe_allow_html=True)
            fig_sim = plot_sim_metrics(
                rewards, cleared,
                st.session_state["sim_actions"],
                st.session_state["sim_emissions"]
            )
            st.pyplot(fig_sim, use_container_width=True)
            plt.close(fig_sim)

        if st.session_state["sim_done"]:
            st.success(
                f"🏁 Episode complete! Steps: {step} | "
                f"Reward: **{sum(rewards):.1f}** | "
                f"Cleared: **{sum(cleared)}** | "
                f"Emissions: **{sum(st.session_state['sim_emissions']):.0f}**"
            )
            st.balloons()


# ── TAB 3: COMPARISON ────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">AI Agent vs Fixed Timer</div>', unsafe_allow_html=True)
    st.markdown("Run the same episode with both agents and compare performance.")

    if not st.session_state["trained"]:
        st.warning("⚠️ Train an agent first.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fixed_interval = st.slider("Fixed Timer Interval (steps)", 1, 20, 5)
        with col_c2:
            n_compare_eps = st.slider("Episodes to compare", 1, 20, 5)

        if st.button("⚖️ Run Comparison", type="primary"):
            with st.spinner("Running comparison episodes…"):
                all_ai_r, all_ai_c   = [], []
                all_fx_r, all_fx_c   = [], []

                env_ai = TrafficEnv(
                    difficulty=difficulty,
                    max_steps=max_steps,
                    enable_weather=enable_weather,
                    enable_time_patterns=enable_time,
                )
                agent_sim = st.session_state["agent"]

                for _ in range(n_compare_eps):
                    # AI episode
                    state = env_ai.reset()
                    ep_r, ep_c = [], []
                    for _ in range(max_steps):
                        if hasattr(agent_sim, 'store'):
                            action = agent_sim.choose_action(env_ai.vehicles.copy(), greedy=True)
                        else:
                            action = agent_sim.choose_action(TrafficEnv.state_to_index(state), greedy=True)
                        state, reward, done, info = env_ai.step(action)
                        ep_r.append(reward)
                        ep_c.append(info["cleared"])
                        if done: break
                    all_ai_r.extend(ep_r)
                    all_ai_c.extend(ep_c)

                    # Fixed timer episode
                    fx_r, fx_c = run_fixed_timer_episode(difficulty, max_steps, fixed_interval)
                    all_fx_r.extend(fx_r)
                    all_fx_c.extend(fx_c)

            st.session_state["comparison_ai_rewards"]    = all_ai_r
            st.session_state["comparison_fixed_rewards"] = all_fx_r
            st.session_state["comparison_ai_cleared"]    = all_ai_c
            st.session_state["comparison_fixed_cleared"] = all_fx_c

        ai_r  = st.session_state["comparison_ai_rewards"]
        fx_r  = st.session_state["comparison_fixed_rewards"]
        ai_c  = st.session_state["comparison_ai_cleared"]
        fx_c  = st.session_state["comparison_fixed_cleared"]

        if ai_r and fx_r:
            # KPI summary
            k1, k2, k3, k4 = st.columns(4)
            ai_sum  = sum(ai_r)
            fx_sum  = sum(fx_r)
            ai_clr  = sum(ai_c)
            fx_clr  = sum(fx_c)
            delta_r = (ai_sum - fx_sum) / max(abs(fx_sum), 1) * 100
            delta_c = (ai_clr - fx_clr) / max(fx_clr, 1) * 100

            k1.markdown(kpi(f"{ai_sum:.0f}", "AI Total Reward"), unsafe_allow_html=True)
            k2.markdown(kpi(f"{fx_sum:.0f}", "Fixed Total Reward"), unsafe_allow_html=True)
            k3.markdown(kpi(f"{ai_clr}", "AI Cleared"), unsafe_allow_html=True)
            k4.markdown(kpi(f"{fx_clr}", "Fixed Cleared"), unsafe_allow_html=True)

            sign_r = "+" if delta_r >= 0 else ""
            sign_c = "+" if delta_c >= 0 else ""
            badge_r = "badge-green" if delta_r >= 0 else "badge-red"
            badge_c = "badge-green" if delta_c >= 0 else "badge-red"
            st.markdown(
                f'<span class="badge {badge_r}">Reward: {sign_r}{delta_r:.1f}%</span> &nbsp;'
                f'<span class="badge {badge_c}">Cleared: {sign_c}{delta_c:.1f}%</span>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-header">Cumulative Performance</div>', unsafe_allow_html=True)
            fig_comp = plot_comparison(ai_r, fx_r, ai_c, fx_c)
            st.pyplot(fig_comp, use_container_width=True)
            plt.close(fig_comp)


# ── TAB 4: INSPECT ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Q-Table / Model Inspector</div>', unsafe_allow_html=True)

    if not st.session_state["trained"]:
        st.info("Train or load an agent first.")
    else:
        agent = st.session_state["agent"]
        stats = agent.get_training_stats()

        # Agent info
        atype = stats.get("agent_type", "Q-Learning")
        st.markdown(f'<span class="badge badge-amber">{atype}</span>', unsafe_allow_html=True)
        if stats.get("device"):
            st.markdown(f'<span class="badge badge-gray">Device: {stats["device"]}</span>',
                        unsafe_allow_html=True)

        if hasattr(agent, 'q_table'):
            with st.expander("🗃️ Q-Table Heatmap (first 64 states)"):
                fig_qt, ax_qt = plt.subplots(figsize=(10, 3.5))
                fig_qt.patch.set_facecolor("#0a0a0f")
                ax_qt.set_facecolor("#111118")
                subset = agent.q_table[:64, :]
                im = ax_qt.imshow(subset.T, aspect="auto", cmap="RdYlGn",
                                  interpolation="nearest")
                ax_qt.set_title("Q-Values (states 0–63 × actions)", color="#888", pad=8, fontsize=10)
                ax_qt.set_xlabel("State Index", color="#555")
                ax_qt.set_ylabel("Action", color="#555")
                ax_qt.set_yticks([0, 1])
                ax_qt.set_yticklabels(["0 (NS)", "1 (EW)"], color="#888")
                ax_qt.tick_params(colors="#333")
                plt.colorbar(im, ax=ax_qt)
                plt.tight_layout()
                st.pyplot(fig_qt, use_container_width=True)
                plt.close(fig_qt)

            with st.expander("📋 Policy Summary"):
                policy  = agent.get_policy_summary()
                ns_pct  = 100 * np.mean(policy == 0)
                ew_pct  = 100 * np.mean(policy == 1)
                p1, p2  = st.columns(2)
                p1.markdown(kpi(f"{ns_pct:.1f}%", "States → NS Green"), unsafe_allow_html=True)
                p2.markdown(kpi(f"{ew_pct:.1f}%", "States → EW Green"), unsafe_allow_html=True)

        elif hasattr(agent, 'online_net'):
            with st.expander("🧠 DQN Network Summary"):
                st.code(str(agent.online_net))
                if agent.episode_losses:
                    fig_loss, ax_loss = plt.subplots(figsize=(8, 2.5))
                    fig_loss.patch.set_facecolor("#0a0a0f")
                    ax_loss.set_facecolor("#111118")
                    ax_loss.plot(agent.episode_losses, color="#e74c3c", lw=1.2)
                    ax_loss.set_title("Training Loss", color="#888", fontsize=9)
                    ax_loss.tick_params(colors="#333")
                    for sp in ax_loss.spines.values(): sp.set_edgecolor("#222")
                    plt.tight_layout()
                    st.pyplot(fig_loss, use_container_width=True)
                    plt.close(fig_loss)

# ════════════════════════════════════════════════════════
#  Footer
# ════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<center style='color:#333;font-size:0.72rem;letter-spacing:1px;'>"
    "TRAFFIC RL v2 &nbsp;·&nbsp; Q-LEARNING + DQN &nbsp;·&nbsp; "
    "WEATHER · TIME PATTERNS · VEHICLE TYPES · FastAPI"
    "</center>",
    unsafe_allow_html=True,
)