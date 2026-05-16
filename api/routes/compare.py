"""
routes/compare.py

POST /api/v1/compare
    Body: { "athlete_ids": ["id1", "id2", ...] }  (2–4 athletes)
    Returns side-by-side latest stats for each athlete,
    plus rankings and deltas for key metrics.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing   import List, Dict, Any

from models.auth_deps       import get_current_user
from services.athlete_store import AthleteStore
from services.user_store    import UserStore

router = APIRouter(prefix="/compare", tags=["Comparison"])


class CompareRequest(BaseModel):
    athlete_ids: List[str] = Field(..., min_length=2, max_length=4)


def _rank(athletes_data: List[Dict], key: str, lower_is_better: bool = True) -> Dict[str, int]:
    """Return {athlete_id: rank} for a given metric. Rank 1 = best."""
    valid = [(a["athlete_id"], a["latest"].get(key)) for a in athletes_data
             if a.get("latest") and a["latest"].get(key) is not None]
    valid.sort(key=lambda x: x[1], reverse=not lower_is_better)
    return {aid: i + 1 for i, (aid, _) in enumerate(valid)}


@router.post("", summary="Compare 2–4 athletes side by side")
async def compare_athletes(
    body: CompareRequest,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:

    results = []

    for athlete_id in body.athlete_ids:
        athlete = AthleteStore.get(athlete_id)
        if not athlete:
            raise HTTPException(status_code=404,
                                detail=f"Athlete '{athlete_id}' not found.")
        # Non-admins can only compare their own athletes
        if current_user["role"] != "admin" and \
                athlete.get("owner_id") != current_user["user_id"]:
            raise HTTPException(status_code=403,
                                detail=f"Access denied for athlete '{athlete_id}'.")

        history = AthleteStore.get_history(athlete_id, limit=7)
        latest  = history[0] if history else None

        # 7-day trend: avg of days 4-7 vs days 1-3 (positive = worsening)
        trend = None
        if len(history) >= 4:
            recent = sum(h["risk_score"] for h in history[:3]) / 3
            older  = sum(h["risk_score"] for h in history[3:6]) / max(len(history[3:6]), 1)
            trend  = round(recent - older, 1)

        # 7-day averages
        avg_7d = {}
        if history:
            week = history[:7]
            avg_7d = {
                "risk_score":  round(sum(h["risk_score"]           for h in week) / len(week), 1),
                "acwr":        round(sum(h["acwr"]                 for h in week) / len(week), 3),
                "sleep_hours": round(sum(h.get("sleep_hours", 7)   for h in week) / len(week), 2),
                "soreness":    round(sum(h.get("soreness", 2)      for h in week) / len(week), 2),
                "daily_load":  round(sum(h.get("daily_load", 0)    for h in week) / len(week), 1),
            }

        results.append({
            "athlete_id":   athlete_id,
            "name":         athlete["name"],
            "sport":        athlete["sport"],
            "age":          athlete["age"],
            "prev_injury":  athlete.get("prev_injury", 0),
            "total_logs":   AthleteStore.history_count(athlete_id),
            "latest":       latest,
            "avg_7d":       avg_7d,
            "risk_trend":   trend,   # positive = risk going up, negative = improving
            "has_data":     latest is not None,
        })

    # ── Rankings across compared athletes ────────────────────────────────────
    # For each metric, rank athletes (1 = best)
    rankings = {
        "risk_score":  _rank(results, "risk_score",  lower_is_better=True),
        "acwr":        _rank(results, "acwr",         lower_is_better=True),
        "sleep_hours": _rank(results, "sleep_hours",  lower_is_better=False),
        "soreness":    _rank(results, "soreness",     lower_is_better=True),
    }

    # ── Deltas vs best performer ──────────────────────────────────────────────
    metrics = ["risk_score", "acwr", "sleep_hours", "soreness", "daily_load"]
    best_vals: Dict[str, float] = {}
    for m in metrics:
        vals = [r["latest"][m] for r in results
                if r.get("latest") and r["latest"].get(m) is not None]
        if vals:
            best_vals[m] = min(vals) if m != "sleep_hours" else max(vals)

    for r in results:
        r["deltas"] = {}
        if r.get("latest"):
            for m in metrics:
                if m in best_vals and r["latest"].get(m) is not None:
                    delta = round(r["latest"][m] - best_vals[m], 2)
                    r["deltas"][m] = delta

    return {
        "athletes":  results,
        "rankings":  rankings,
        "best_vals": best_vals,
        "count":     len(results),
    }