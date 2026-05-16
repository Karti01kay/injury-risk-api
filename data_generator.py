"""
Synthetic athlete training data generator.
Produces realistic daily training logs with injury labels
based on sports science literature (ACWR, sleep, strain).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

SPORT_PROFILES = {
    "running":    {"base_load": 55, "load_std": 18, "injury_base": 0.12},
    "cycling":    {"base_load": 60, "load_std": 20, "injury_base": 0.08},
    "football":   {"base_load": 65, "load_std": 22, "injury_base": 0.15},
    "basketball": {"base_load": 62, "load_std": 20, "injury_base": 0.13},
    "swimming":   {"base_load": 50, "load_std": 15, "injury_base": 0.07},
}

def generate_athlete(athlete_id: int, days: int = 180) -> pd.DataFrame:
    sport = np.random.choice(list(SPORT_PROFILES.keys()))
    profile = SPORT_PROFILES[sport]
    age = np.random.randint(18, 45)
    has_prev_injury = np.random.choice([0, 1], p=[0.65, 0.35])

    records = []
    start_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 60))

    # Rolling load history for ACWR
    load_history = [profile["base_load"]] * 28

    for day in range(days):
        date = start_date + timedelta(days=day)
        is_rest_day = np.random.random() < 0.20

        if is_rest_day:
            daily_load = 0.0
        else:
            # Simulate periodic training blocks (build → peak → taper)
            phase_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day / 28)
            daily_load = max(0, np.random.normal(
                profile["base_load"] * phase_factor,
                profile["load_std"]
            ))

        load_history.append(daily_load)
        load_history.pop(0)

        acute_load  = np.mean(load_history[-7:])
        chronic_load = np.mean(load_history[-28:])
        acwr = acute_load / chronic_load if chronic_load > 1 else 1.0

        sleep_hours   = np.clip(np.random.normal(7.0, 1.1), 3, 10)
        sleep_quality = np.clip(np.random.normal(3.0, 0.8), 1, 5)
        rhr           = np.clip(np.random.normal(58, 8), 40, 90)
        soreness      = np.clip(np.random.normal(2.5, 1.0), 1, 5)

        # Days since last rest
        recent = load_history[-14:]
        days_since_rest = 0
        for l in reversed(recent[:-1]):
            if l == 0:
                break
            days_since_rest += 1

        # ── Injury probability (evidence-based heuristics) ──────────────
        inj_prob = profile["injury_base"]

        # ACWR sweet spot 0.8–1.3; danger zones below/above
        if acwr > 1.5:
            inj_prob += 0.25 + (acwr - 1.5) * 0.3
        elif acwr > 1.3:
            inj_prob += 0.10
        elif acwr < 0.6:
            inj_prob += 0.08          # detraining → sudden return

        if sleep_hours < 6:
            inj_prob += 0.12
        elif sleep_hours < 7:
            inj_prob += 0.05

        if sleep_quality < 2:
            inj_prob += 0.08

        if rhr > 70:
            inj_prob += 0.06          # fatigue marker

        if soreness >= 4:
            inj_prob += 0.10
        elif soreness >= 3:
            inj_prob += 0.04

        if days_since_rest >= 7:
            inj_prob += 0.12
        elif days_since_rest >= 5:
            inj_prob += 0.05

        if has_prev_injury:
            inj_prob += 0.08

        if age > 35:
            inj_prob += 0.04

        inj_prob = np.clip(inj_prob, 0.0, 0.95)
        injured  = int(np.random.random() < inj_prob)

        records.append({
            "athlete_id":       athlete_id,
            "date":             date.strftime("%Y-%m-%d"),
            "sport":            sport,
            "age":              age,
            "prev_injury":      has_prev_injury,
            "daily_load":       round(daily_load, 2),
            "acute_load_7d":    round(acute_load,  2),
            "chronic_load_28d": round(chronic_load, 2),
            "acwr":             round(acwr, 4),
            "sleep_hours":      round(sleep_hours, 2),
            "sleep_quality":    round(sleep_quality, 2),
            "resting_hr":       round(rhr, 1),
            "soreness":         round(soreness, 2),
            "days_since_rest":  days_since_rest,
            "is_rest_day":      int(is_rest_day),
            "injury_occurred":  injured,
        })

    return pd.DataFrame(records)


def generate_dataset(n_athletes: int = 300, days: int = 180) -> pd.DataFrame:
    frames = [generate_athlete(i, days) for i in range(n_athletes)]
    df = pd.concat(frames, ignore_index=True)
    print(f"Dataset: {len(df):,} rows | {df['injury_occurred'].mean()*100:.1f}% injury rate")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("/home/claude/injury_risk/raw_data.csv", index=False)
    print("Saved → raw_data.csv")
    print(df.head())
