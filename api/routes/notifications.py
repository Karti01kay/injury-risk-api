"""
routes/notifications.py

POST /api/v1/notifications/token    — register/update push token for a user
POST /api/v1/notifications/send     — send a push notification (admin only)
GET  /api/v1/notifications/tokens   — list all tokens (admin only)

Uses Expo's push notification service — no APNs/FCM setup needed.
"""

from fastapi  import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing   import List, Dict, Optional
import urllib.request, json

from models.auth_deps  import get_current_user, require_admin
from services.user_store import UserStore

router = APIRouter(prefix="/notifications", tags=["Notifications"])

# In-memory token store: user_id → push token
_tokens: Dict[str, str] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    token: str

class SendRequest(BaseModel):
    user_ids: Optional[List[str]] = None   # None = broadcast to all
    title:    str
    body:     str
    data:     Optional[dict] = None


# ── Token management ──────────────────────────────────────────────────────────

@router.post("/token", status_code=status.HTTP_204_NO_CONTENT,
             summary="Register or update Expo push token for current user")
async def register_token(body: TokenRequest,
                         current_user: dict = Depends(get_current_user)):
    if not body.token.startswith("ExponentPushToken"):
        raise HTTPException(status_code=400,
                            detail="Invalid Expo push token format.")
    _tokens[current_user["user_id"]] = body.token


@router.delete("/token", status_code=status.HTTP_204_NO_CONTENT,
               summary="Remove push token (logout / disable notifications)")
async def remove_token(current_user: dict = Depends(get_current_user)):
    _tokens.pop(current_user["user_id"], None)


@router.get("/tokens", summary="List all registered push tokens (admin only)")
async def list_tokens(admin: dict = Depends(require_admin)) -> List[Dict]:
    result = []
    for uid, token in _tokens.items():
        user = UserStore.get(uid)
        result.append({
            "user_id": uid,
            "name":    user["name"]  if user else "Unknown",
            "email":   user["email"] if user else "Unknown",
            "token":   token,
        })
    return result


# ── Send notifications ────────────────────────────────────────────────────────

def _expo_push(messages: List[dict]) -> List[dict]:
    """Send messages via Expo push API. Returns receipts."""
    if not messages:
        return []
    body = json.dumps(messages).encode()
    req  = urllib.request.Request(
        "https://exp.host/--/api/v2/push/send",
        data=body,
        headers={"Content-Type": "application/json",
                 "Accept":       "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("data", [])
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@router.post("/send", summary="Send push notification to users (admin only)")
async def send_notification(body: SendRequest,
                            admin: dict = Depends(require_admin)) -> Dict:
    # Resolve target tokens
    if body.user_ids:
        tokens = [_tokens[uid] for uid in body.user_ids if uid in _tokens]
    else:
        tokens = list(_tokens.values())

    if not tokens:
        raise HTTPException(status_code=404,
                            detail="No registered push tokens found.")

    messages = [
        {
            "to":    token,
            "title": body.title,
            "body":  body.body,
            "data":  body.data or {},
            "sound": "default",
        }
        for token in tokens
    ]

    receipts = _expo_push(messages)
    sent     = sum(1 for r in receipts if r.get("status") == "ok")

    return {
        "sent":    sent,
        "failed":  len(tokens) - sent,
        "total":   len(tokens),
        "receipts": receipts,
    }


@router.post("/send-risk-alert",
             summary="Send high-risk alert to a specific user (internal / admin)")
async def send_risk_alert_notification(
    user_id:     str,
    athlete_name:str,
    risk_score:  float,
    category:    str,
    admin: dict = Depends(require_admin),
) -> Dict:
    token = _tokens.get(user_id)
    if not token:
        raise HTTPException(status_code=404,
                            detail="User has no registered push token.")

    emoji = "🚨" if category == "high" else "⚠️"
    messages = [{
        "to":    token,
        "title": f"{emoji} {athlete_name} — {category.upper()} RISK",
        "body":  f"Risk score {round(risk_score)}/100. Check the app for recommendations.",
        "data":  {"type": "risk_alert", "athlete_name": athlete_name,
                  "risk_score": risk_score, "category": category},
        "sound": "default",
    }]
    receipts = _expo_push(messages)
    return {"sent": len(receipts), "receipts": receipts}