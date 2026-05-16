"""
models/auth_deps.py
FastAPI dependencies for JWT-protected routes.
"""

from fastapi            import Header, HTTPException, status
from services.auth_service import decode_token
from services.user_store   import UserStore


def _get_token(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header format")
    return authorization.split(" ", 1)[1]


def get_current_user(authorization: str = Header(...)) -> dict:
    """Inject the current authenticated user into any route."""
    token   = _get_token(authorization)
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = UserStore.get(payload["sub"])
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_admin(authorization: str = Header(...)) -> dict:
    """Inject current user and assert admin role."""
    user = get_current_user(authorization)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user