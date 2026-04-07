"""
Enhanced Traffic Environment v2
================================
New in v2:
  - Vehicle types: car, truck, bus, bike (different clear rates & weights)
  - Weather effects: clear, rain, fog, storm
  - Real time-of-day traffic patterns (morning rush, midday, evening rush, night)
  - Emergency vehicle events (random priority override)
  - Emission/fuel cost in reward
  - Multi-metric info dict for rich dashboard display
"""

import numpy as np
import random
from typing import Tuple, Dict, List

# ── Difficulty presets ────────────────────────────────────────────────────────
DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    "Easy":   {"base_arrival": 0.3, "max_vehicles": 10},
    "Medium": {"base_arrival": 0.6, "max_vehicles": 15},
    "Hard":   {"base_arrival": 1.0, "max_vehicles": 20},
}

# ── Vehicle types ─────────────────────────────────────────────────────────────
VEHICLE_TYPES = {
    "car":   {"clear_rate": 0.65, "weight": 1.0, "emission": 1.0, "prob": 0.55},
    "bike":  {"clear_rate": 0.85, "weight": 0.3, "emission": 0.2, "prob": 0.20},
    "bus":   {"clear_rate": 0.40, "weight": 3.0, "emission": 4.0, "prob": 0.10},
    "truck": {"clear_rate": 0.35, "weight": 2.5, "emission": 5.0, "prob": 0.15},
}
VEHICLE_NAMES = list(VEHICLE_TYPES.keys())
VEHICLE_PROBS = [VEHICLE_TYPES[v]["prob"] for v in VEHICLE_NAMES]

# ── Weather effects ───────────────────────────────────────────────────────────
WEATHER_EFFECTS = {
    "clear": {"clear_multiplier": 1.00, "arrival_multiplier": 1.00, "label": "☀️ Clear"},
    "rain":  {"clear_multiplier": 0.70, "arrival_multiplier": 1.20, "label": "🌧️ Rain"},
    "fog":   {"clear_multiplier": 0.60, "arrival_multiplier": 0.90, "label": "🌫️ Fog"},
    "storm": {"clear_multiplier": 0.40, "arrival_multiplier": 1.50, "label": "⛈️ Storm"},
}

# ── Time-of-day patterns (hour 0-23 → arrival multiplier) ────────────────────
# Peaks: morning rush 8-10, evening rush 17-19
TIME_PATTERN = {
    0: 0.15, 1: 0.10, 2: 0.08, 3: 0.08, 4: 0.10, 5: 0.20,
    6: 0.45, 7: 0.80, 8: 1.50, 9: 1.60, 10: 1.10, 11: 0.90,
    12: 1.00, 13: 0.95, 14: 0.85, 15: 0.90, 16: 1.20, 17: 1.55,
    18: 1.65, 19: 1.30, 20: 0.90, 21: 0.65, 22: 0.40, 23: 0.25,
}

# ── State discretisation ──────────────────────────────────────────────────────
BINS = [0, 3, 7, 11]
N_BINS = 4
N_LANES = 4


def discretize(value: int) -> int:
    return int(np.digitize(value, BINS) - 1)


class TrafficEnv:
    """
    Enhanced 4-way intersection environment.

    Observation: (n_bin, s_bin, e_bin, w_bin) → 256 states
    Actions    : 0 = NS green, 1 = EW green
    Reward     : cleared_weight * 2 - waiting_weight * 0.5 - emission_cost * 0.1
    """

    SIGNAL_LABELS = {0: "North-South GREEN", 1: "East-West GREEN"}
    LANE_NAMES    = ["North", "South", "East", "West"]

    def __init__(
        self,
        difficulty: str = "Medium",
        max_steps: int = 100,
        enable_weather: bool = True,
        enable_time_patterns: bool = True,
        start_hour: int = 8,
    ):
        assert difficulty in DIFFICULTY_CONFIGS
        self.difficulty           = difficulty
        self.max_steps            = max_steps
        self.enable_weather       = enable_weather
        self.enable_time_patterns = enable_time_patterns
        self.config               = DIFFICULTY_CONFIGS[difficulty]
        self._step_count          = 0
        self.current_hour         = start_hour

        # Per-lane vehicle counts [N, S, E, W]
        self.vehicles      = np.zeros(4, dtype=int)
        # Per-lane weighted vehicle load (accounts for vehicle type)
        self.vehicle_load  = np.zeros(4, dtype=float)

        self.current_action    = 0
        self.current_weather   = "clear"
        self.emergency_active  = False
        self.emergency_lane    = -1

        # Episode stats
        self.total_cleared   = 0
        self.total_waited    = 0
        self.total_emissions = 0.0

        # Per-step history for dashboard
        self.step_history: List[Dict] = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self) -> Tuple[int, ...]:
        self._step_count     = 0
        self.total_cleared   = 0
        self.total_waited    = 0
        self.total_emissions = 0.0
        self.current_action  = 0
        self.step_history    = []
        self.emergency_active = False

        # Randomise starting conditions
        if self.enable_weather:
            self.current_weather = random.choices(
                list(WEATHER_EFFECTS.keys()),
                weights=[0.55, 0.25, 0.12, 0.08]
            )[0]

        arr = self.config["base_arrival"] * 2
        self.vehicles = np.array([
            np.random.poisson(arr) for _ in range(4)
        ], dtype=int)
        self.vehicle_load = self.vehicles.astype(float)
        self._clip_vehicles()
        return self.get_state()

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict]:
        assert action in (0, 1)
        self.current_action  = action
        self._step_count    += 1
        self.current_hour    = (self.current_hour + 1) % 24

        # ── Weather change (5% chance per step) ───────────────────────────
        if self.enable_weather and random.random() < 0.05:
            self.current_weather = random.choices(
                list(WEATHER_EFFECTS.keys()),
                weights=[0.55, 0.25, 0.12, 0.08]
            )[0]

        weather = WEATHER_EFFECTS[self.current_weather]

        # ── Emergency vehicle (3% chance) ─────────────────────────────────
        self.emergency_active = random.random() < 0.03
        if self.emergency_active:
            self.emergency_lane = random.randint(0, 3)

        # ── Arrivals ───────────────────────────────────────────────────────
        time_mult = TIME_PATTERN[self.current_hour] if self.enable_time_patterns else 1.0
        base_arr  = self.config["base_arrival"]

        arrivals       = np.zeros(4, dtype=int)
        arrival_load   = np.zeros(4, dtype=float)
        arrival_emit   = 0.0

        for lane in range(4):
            rate     = base_arr * time_mult * weather["arrival_multiplier"]
            n_arrive = np.random.poisson(rate)
            arrivals[lane] = n_arrive
            for _ in range(n_arrive):
                vtype = random.choices(VEHICLE_NAMES, VEHICLE_PROBS)[0]
                arrival_load[lane] += VEHICLE_TYPES[vtype]["weight"]
                arrival_emit       += VEHICLE_TYPES[vtype]["emission"]

        self.vehicles     = np.clip(self.vehicles + arrivals, 0, self.config["max_vehicles"])
        self.vehicle_load = np.clip(self.vehicle_load + arrival_load, 0, self.config["max_vehicles"] * 3)

        # ── Clearing ───────────────────────────────────────────────────────
        green_lanes  = [0, 1] if action == 0 else [2, 3]
        cleared      = 0
        cleared_load = 0.0
        emit_step    = 0.0

        for lane in green_lanes:
            # Emergency override: clear that lane fully if emergency
            if self.emergency_active and self.emergency_lane == lane:
                actually_cleared = self.vehicles[lane]
            else:
                base_rate = 0.55 * weather["clear_multiplier"]
                can_clear = max(0, int(self.vehicles[lane] * base_rate
                                       + np.random.normal(0, 0.5)))
                actually_cleared = min(self.vehicles[lane], can_clear)

            self.vehicles[lane]     = max(0, self.vehicles[lane] - actually_cleared)
            load_cleared             = min(self.vehicle_load[lane],
                                          actually_cleared * 1.2)
            self.vehicle_load[lane] = max(0, self.vehicle_load[lane] - load_cleared)
            cleared      += actually_cleared
            cleared_load += load_cleared

            # Idling emission for red lanes
        for lane in range(4):
            if lane not in green_lanes:
                emit_step += self.vehicles[lane] * 0.3

        self._clip_vehicles()

        waiting = int(np.sum(self.vehicles))
        self.total_cleared   += cleared
        self.total_waited    += waiting
        self.total_emissions += emit_step

        # ── Reward ─────────────────────────────────────────────────────────
        reward = (
            cleared_load * 2.0
            - waiting * 0.5
            - emit_step * 0.05
            + (5.0 if self.emergency_active and cleared > 0 else 0)
        )

        done = self._step_count >= self.max_steps

        info = {
            "vehicles"       : self.vehicles.copy(),
            "vehicle_load"   : self.vehicle_load.copy(),
            "cleared"        : cleared,
            "waiting"        : waiting,
            "weather"        : self.current_weather,
            "weather_label"  : weather["label"],
            "hour"           : self.current_hour,
            "emergency"      : self.emergency_active,
            "emergency_lane" : self.emergency_lane if self.emergency_active else -1,
            "emissions"      : emit_step,
            "step"           : self._step_count,
            "total_cleared"  : self.total_cleared,
            "total_emissions": self.total_emissions,
            "time_multiplier": time_mult,
        }

        self.step_history.append({
            "step"    : self._step_count,
            "reward"  : reward,
            "cleared" : cleared,
            "waiting" : waiting,
            "action"  : action,
            "weather" : self.current_weather,
            "hour"    : self.current_hour,
        })

        return self.get_state(), reward, done, info

    def get_state(self) -> Tuple[int, ...]:
        return tuple(discretize(int(v)) for v in self.vehicles)

    def render(self) -> str:
        n, s, e, w = self.vehicles
        sig = self.SIGNAL_LABELS[self.current_action]
        w_label = WEATHER_EFFECTS[self.current_weather]["label"]
        return (
            f"\n{'─'*42}\n"
            f"  Signal  : {sig}\n"
            f"  Weather : {w_label}  Hour: {self.current_hour:02d}:00\n"
            f"{'─'*42}\n"
            f"           North [{n:>2}]\n"
            f"  West [{w:>2}] ──┼── [{e:>2}] East\n"
            f"           South [{s:>2}]\n"
            f"{'─'*42}\n"
            f"  Step: {self._step_count}/{self.max_steps}  "
            f"Cleared: {self.total_cleared}\n"
            f"{'─'*42}\n"
        )

    def _clip_vehicles(self):
        mx = self.config["max_vehicles"]
        self.vehicles     = np.clip(self.vehicles, 0, mx)
        self.vehicle_load = np.clip(self.vehicle_load, 0, mx * 3)

    @property
    def state_space_size(self) -> int:
        return N_BINS ** N_LANES

    @property
    def action_space_size(self) -> int:
        return 2

    @staticmethod
    def state_to_index(state: Tuple[int, ...]) -> int:
        idx = 0
        for b in state:
            idx = idx * N_BINS + b
        return idx