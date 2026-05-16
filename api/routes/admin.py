"""
routes/admin.py
Admin-only endpoints — all require role=admin.

GET    /api/v1/admin/users                — list all users
GET    /api/v1/admin/users/{id}           — get single user
PUT    /api/v1/admin/users/{id}/role      — change user role
DELETE /api/v1/admin/users/{id}           — deactivate user
GET    /api/v1/admin/athletes             — all athletes across all users
GET    /api/v1/admin/team-summary         — risk summary across all athletes
GET    /api/v1/admin/alerts               — athletes at high/moderate risk
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing  import List, Dict, Any
from datetime import datetime

from models.auth_deps       import require_admin
from models.auth_schemas    import UserResponse, UpdateRoleRequest
from services.user_store    import UserStore
from services.athlete_store import AthleteStore

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users", response_model=List[UserResponse],
            summary="List all registered users")
async def list_users(admin: dict = Depends(require_admin)):
    return UserStore.all()


@router.get("/users/{user_id}", response_model=UserResponse,
            summary="Get a single user by ID")
async def get_user(user_id: str, admin: dict = Depends(require_admin)):
    user = UserStore.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@router.put("/users/{user_id}/role", response_model=UserResponse,
            summary="Update a user's role (athlete <-> admin)")
async def update_role(user_id: str, body: UpdateRoleRequest,
                      admin: dict = Depends(require_admin)):
    if user_id == admin["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role.")
    user = UserStore.update_role(user_id, body.role)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Deactivate a user account")
async def deactivate_user(user_id: str, admin: dict = Depends(require_admin)):
    if user_id == admin["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself.")
    user = UserStore.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    UserStore.deactivate(user_id)


# ── Athletes ──────────────────────────────────────────────────────────────────

@router.get("/athletes", summary="List all athletes across all user accounts")
async def list_all_athletes(admin: dict = Depends(require_admin)) -> List[Dict]:
    athletes = AthleteStore.all()
    result = []
    for a in athletes:
        history = AthleteStore.get_history(a["athlete_id"], limit=1)
        latest  = history[0] if history else None
        owner   = UserStore.get(a.get("owner_id", ""))
        result.append({
            **a,
            "owner_name":      owner["name"]  if owner else "Unknown",
            "owner_email":     owner["email"] if owner else "Unknown",
            "latest_risk":     latest["risk_score"]    if latest else None,
            "latest_category": latest["risk_category"] if latest else None,
            "latest_date":     latest["date"]           if latest else None,
            "total_logs":      AthleteStore.history_count(a["athlete_id"]),
        })
    return result


# ── Team Summary ──────────────────────────────────────────────────────────────

@router.get("/team-summary", summary="Aggregated risk summary across all athletes")
async def team_summary(admin: dict = Depends(require_admin)) -> Dict[str, Any]:
    athletes = AthleteStore.all()
    total    = len(athletes)
    if total == 0:
        return {"total_athletes": 0, "message": "No athletes registered yet."}

    risk_counts = {"low": 0, "elevated": 0, "moderate": 0, "high": 0}
    risk_scores = []
    high_risk   = []
    no_data     = 0

    for a in athletes:
        history = AthleteStore.get_history(a["athlete_id"], limit=1)
        if not history:
            no_data += 1
            continue
        latest = history[0]
        cat    = latest["risk_category"]
        score  = latest["risk_score"]
        risk_counts[cat] = risk_counts.get(cat, 0) + 1
        risk_scores.append(score)
        if cat in ("high", "moderate"):
            owner = UserStore.get(a.get("owner_id", ""))
            high_risk.append({
                "athlete_id":    a["athlete_id"],
                "name":          a["name"],
                "sport":         a["sport"],
                "risk_score":    score,
                "risk_category": cat,
                "acwr":          latest.get("acwr"),
                "owner_name":    owner["name"] if owner else "Unknown",
                "date":          latest["date"],
            })

    avg_score = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0
    high_risk.sort(key=lambda x: x["risk_score"], reverse=True)

    return {
        "total_athletes":             total,
        "athletes_with_data":         total - no_data,
        "athletes_no_data":           no_data,
        "avg_risk_score":             avg_score,
        "risk_distribution":          risk_counts,
        "athletes_needing_attention": high_risk,
        "total_users":                len(UserStore.all()),
        "generated_at":               datetime.utcnow().isoformat(),
    }


# ── Alerts ────────────────────────────────────────────────────────────────────

@router.get("/alerts", summary="All athletes currently at high or moderate risk")
async def get_alerts(admin: dict = Depends(require_admin)) -> List[Dict]:
    athletes = AthleteStore.all()
    alerts   = []
    for a in athletes:
        history = AthleteStore.get_history(a["athlete_id"], limit=1)
        if not history:
            continue
        latest = history[0]
        if latest["risk_category"] in ("high", "moderate"):
            owner = UserStore.get(a.get("owner_id", ""))
            alerts.append({
                "athlete_id":    a["athlete_id"],
                "name":          a["name"],
                "sport":         a["sport"],
                "age":           a["age"],
                "risk_score":    latest["risk_score"],
                "risk_category": latest["risk_category"],
                "acwr":          latest.get("acwr"),
                "sleep_hours":   latest.get("sleep_hours"),
                "soreness":      latest.get("soreness"),
                "date":          latest["date"],
                "owner_name":    owner["name"]  if owner else "Unknown",
                "owner_email":   owner["email"] if owner else "Unknown",
            })
    alerts.sort(key=lambda x: x["risk_score"], reverse=True)
    return alerts