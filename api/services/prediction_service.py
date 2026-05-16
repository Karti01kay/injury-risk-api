"""
PredictionService — the core ML inference layer.
Converts a PredictRequest into a fully populated PredictResponse.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from models.schemas import PredictRequest, PredictResponse, RiskCategory
from models.state   import AppState


SPORT_COLS = ["sport_basketball", "sport_cycling",
              "sport_football",   "sport_running", "sport_swimming"]


def _build_feature_vector(req: PredictRequest) -> pd.DataFrame:
    """Assemble the 24-feature vector the model expects."""
    sport_flags = {c: 0 for c in SPORT_COLS}
    sport_key   = f"sport_{req.sport.value}"
    if sport_key in sport_flags:
        sport_flags[sport_key] = 1

    acwr        = req.acwr
    acwr_danger = int(acwr > 1.5 or acwr < 0.6)

    row = {
        "age":              req.age,
        "prev_injury":      req.prev_injury,
        "daily_load":       req.daily_load,
        "acute_load_7d":    req.acute_load_7d,
        "chronic_load_28d": req.chronic_load_28d,
        "acwr":             acwr,
        "sleep_hours":      req.sleep_hours,
        "sleep_quality":    req.sleep_quality,
        "sleep_deficit_7d": req.sleep_deficit_7d,
        "resting_hr":       req.resting_hr,
        "rhr_trend":        req.rhr_trend,
        "soreness":         req.soreness,
        "soreness_7d":      req.soreness_7d,
        "days_since_rest":  req.days_since_rest,
        "is_rest_day":      req.is_rest_day,
        "load_monotony":    req.load_monotony,
        "training_strain":  req.training_strain,
        "load_spike":       req.load_spike,
        "acwr_danger":      acwr_danger,
        **sport_flags,
    }

    features = AppState.meta["features"]
    return pd.DataFrame([row])[features]


def _risk_category(prob: float) -> RiskCategory:
    t = AppState.meta["risk_thresholds"]
    if prob >= t["high"]:     return RiskCategory.high
    if prob >= t["moderate"]: return RiskCategory.moderate
    if prob >= t["low"]:      return RiskCategory.elevated
    return RiskCategory.low


def _recommendations(req: PredictRequest, prob: float) -> List[str]:
    recs  = []
    acwr  = req.acwr

    if acwr > 1.5:
        recs.append("ACWR critically high — cut training load by 20–30% this week.")
    elif acwr > 1.3:
        recs.append("Workload spike detected — avoid any load increases for 3–5 days.")
    elif acwr < 0.6:
        recs.append("Sudden ramp-up after low training period — increase gradually.")

    if req.sleep_hours < 6:
        recs.append("Critical sleep deficit — prioritise 7–9 h sleep before next session.")
    elif req.sleep_hours < 7:
        recs.append("Below-optimal sleep — aim for at least 7 h tonight.")

    if req.sleep_quality <= 2:
        recs.append("Poor sleep quality reported — consider sleep hygiene review.")

    if req.soreness >= 4:
        recs.append("High muscle soreness — take an active recovery or full rest day.")
    elif req.soreness >= 3:
        recs.append("Moderate soreness — keep today's session low-intensity.")

    if req.days_since_rest >= 7:
        recs.append("No rest day in 7+ days — schedule one within the next 48 h.")
    elif req.days_since_rest >= 5:
        recs.append("Consider inserting a rest or recovery day soon.")

    if req.rhr_trend > 5:
        recs.append("Resting HR elevated above your baseline — sign of accumulated fatigue.")

    if req.load_monotony > 1.5:
        recs.append("Training monotony is high — vary session intensity to reduce strain.")

    if not recs:
        recs.append("All wellness indicators look healthy — maintain your current plan.")

    return recs


def _top_risk_factors(req: PredictRequest) -> List[str]:
    """Return the top contributing risk signals for display in the app."""
    factors: List[Tuple[float, str]] = []
    acwr = req.acwr

    if acwr > 1.5:
        factors.append((0.90, f"ACWR = {acwr:.2f} (critical zone > 1.5)"))
    elif acwr > 1.3:
        factors.append((0.65, f"ACWR = {acwr:.2f} (elevated zone > 1.3)"))

    if req.sleep_hours < 6:
        factors.append((0.80, f"Sleep only {req.sleep_hours:.1f} h (< 6 h threshold)"))
    elif req.sleep_hours < 7:
        factors.append((0.45, f"Sleep {req.sleep_hours:.1f} h (below 7 h optimal)"))

    if req.soreness >= 4:
        factors.append((0.75, f"Soreness score {req.soreness:.1f}/5 (high)"))
    elif req.soreness >= 3:
        factors.append((0.40, f"Soreness score {req.soreness:.1f}/5 (moderate)"))

    if req.days_since_rest >= 7:
        factors.append((0.70, f"{req.days_since_rest} consecutive days without rest"))

    if req.rhr_trend > 5:
        factors.append((0.60, f"Resting HR +{req.rhr_trend:.0f} bpm above baseline"))

    if req.prev_injury:
        factors.append((0.35, "Previous injury history increases baseline risk"))

    if req.load_spike > 1.5:
        factors.append((0.55, f"Load spike {req.load_spike:.1f}× above recent average"))

    factors.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in factors[:4]] or ["No significant risk factors identified"]


def predict(req: PredictRequest) -> PredictResponse:
    X         = _build_feature_vector(req)
    X_scaled  = AppState.scaler.transform(X)
    prob      = float(AppState.model.predict_proba(X_scaled)[0][1])
    score     = round(prob * 100, 1)
    category  = _risk_category(prob)
    recs      = _recommendations(req, prob)
    factors   = _top_risk_factors(req)

    return PredictResponse(
        athlete_id       = req.athlete_id,
        risk_score       = score,
        risk_category    = category,
        acwr             = round(req.acwr, 3),
        recommendations  = recs,
        top_risk_factors = factors,
        model_version    = AppState.meta["model_type"],
    )
