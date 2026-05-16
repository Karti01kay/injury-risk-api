"""
Feature engineering for the injury risk model.
Adds derived features on top of the raw training log data.
"""

import pandas as pd
import numpy as np


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["athlete_id", "date"])

    grp = df.groupby("athlete_id")["daily_load"]

    # Monotony = mean / std of last 7 days (high monotony = risky)
    df["load_monotony"] = (
        grp.transform(lambda x: x.rolling(7, min_periods=3).mean()) /
        grp.transform(lambda x: x.rolling(7, min_periods=3).std().replace(0, 1))
    ).round(4)

    # Training strain = 7d sum × monotony
    df["training_strain"] = (
        grp.transform(lambda x: x.rolling(7, min_periods=3).sum()) *
        df["load_monotony"]
    ).round(2)

    # Load spike: today vs 7d avg
    df["load_spike"] = (
        df["daily_load"] /
        grp.transform(lambda x: x.rolling(7, min_periods=3).mean()).replace(0, 1)
    ).round(4)

    # 7-day rolling sleep deficit (below 7h target)
    sleep_grp = df.groupby("athlete_id")["sleep_hours"]
    df["sleep_deficit_7d"] = (
        sleep_grp.transform(lambda x: (7.0 - x).clip(lower=0).rolling(7, min_periods=3).sum())
    ).round(2)

    # RHR trend: current vs 7d avg (positive = elevated = bad)
    rhr_grp = df.groupby("athlete_id")["resting_hr"]
    df["rhr_trend"] = (
        df["resting_hr"] -
        rhr_grp.transform(lambda x: x.rolling(7, min_periods=3).mean())
    ).round(2)

    # Cumulative soreness (7d)
    df["soreness_7d"] = (
        df.groupby("athlete_id")["soreness"]
          .transform(lambda x: x.rolling(7, min_periods=3).mean())
    ).round(2)

    # Risk zone flag based on ACWR
    df["acwr_danger"] = ((df["acwr"] > 1.5) | (df["acwr"] < 0.6)).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    sport_dummies = pd.get_dummies(df["sport"], prefix="sport", drop_first=False)
    return pd.concat([df, sport_dummies], axis=1)


def get_feature_columns() -> list:
    base = [
        "age", "prev_injury",
        "daily_load", "acute_load_7d", "chronic_load_28d", "acwr",
        "sleep_hours", "sleep_quality", "sleep_deficit_7d",
        "resting_hr", "rhr_trend",
        "soreness", "soreness_7d",
        "days_since_rest", "is_rest_day",
        "load_monotony", "training_strain", "load_spike",
        "acwr_danger",
    ]
    sport_cols = [f"sport_{s}" for s in
                  ["basketball", "cycling", "football", "running", "swimming"]]
    return base + sport_cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rolling_features(df)
    df = encode_categoricals(df)
    # Fill NaNs from rolling windows with column medians
    feat_cols = get_feature_columns()
    for c in feat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


if __name__ == "__main__":
    raw = pd.read_csv("/home/claude/injury_risk/raw_data.csv")
    engineered = engineer_features(raw)
    engineered.to_csv("/home/claude/injury_risk/engineered_data.csv", index=False)
    print(f"Features: {get_feature_columns()}")
    print(engineered[get_feature_columns()].describe().T[["mean","std","min","max"]])
