"""
routes/history.py
GET /api/v1/athletes/{id}/history   — auth protected
GET /api/v1/athletes/{id}/analytics — auth protected
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing  import List, Dict, Any

from models.schemas         import HistoryResponse, HistoryEntry
from models.auth_deps       import get_current_user
from services.athlete_store import AthleteStore

router = APIRouter()


def _check_access(athlete_id: str, current_user: dict):
    if not AthleteStore.exists(athlete_id):
        raise HTTPException(status_code=404, detail="Athlete not found.")
    if not AthleteStore.owns(athlete_id, current_user["user_id"]) \
            and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied.")


@router.get("/athletes/{athlete_id}/history", response_model=HistoryResponse,
            summary="Get prediction history for an athlete")
async def get_history(athlete_id: str,
                      limit: int = Query(30, ge=1, le=90),
                      current_user: dict = Depends(get_current_user)):
    _check_access(athlete_id, current_user)
    raw     = AthleteStore.get_history(athlete_id, limit=limit)
    entries = [HistoryEntry(**e) for e in raw]
    return HistoryResponse(athlete_id=athlete_id, entries=entries,
                           total=AthleteStore.history_count(athlete_id))


@router.get("/athletes/{athlete_id}/analytics",
            summary="Summary analytics for an athlete's risk trends")
async def get_analytics(athlete_id: str,
                        current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    _check_access(athlete_id, current_user)
    history = AthleteStore.get_history(athlete_id, limit=90)
    if not history:
        return {"athlete_id": athlete_id, "message": "No prediction history yet."}

    scores = [e["risk_score"]    for e in history]
    cats   = [e["risk_category"] for e in history]
    acwrs  = [e["acwr"]          for e in history]

    streak = 0
    for c in cats:
        if c in ("high", "moderate", "elevated"): streak += 1
        else: break

    cat_counts = {"low": 0, "elevated": 0, "moderate": 0, "high": 0}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    recent_7  = scores[:7]
    recent_30 = scores[:30]

    return {
        "athlete_id":           athlete_id,
        "total_predictions":    len(history),
        "avg_risk_7d":          round(sum(recent_7)  / len(recent_7),  1) if recent_7  else None,
        "avg_risk_30d":         round(sum(recent_30) / len(recent_30), 1) if recent_30 else None,
        "max_risk_score":       round(max(scores), 1),
        "avg_acwr":             round(sum(acwrs) / len(acwrs), 3),
        "risk_category_counts": cat_counts,
        "current_alert_streak": streak,
        "latest_entry":         history[0] if history else None,
    }


@router.delete("/athletes/{athlete_id}/history", status_code=204,
               summary="Clear all prediction history for an athlete")
async def clear_history(athlete_id: str,
                        current_user: dict = Depends(get_current_user)):
    _check_access(athlete_id, current_user)
    AthleteStore._histories[athlete_id] = []