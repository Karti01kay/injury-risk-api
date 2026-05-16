"""
services/auth_service.py
Handles password hashing, JWT creation and verification.
"""

import os
import hashlib
import hmac
import base64
import json
import time
from typing import Optional

# ── Secret key (set via env var in production) ───────────────────────────────
SECRET_KEY  = os.environ.get("JWT_SECRET", "injury-risk-super-secret-key-change-in-prod")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = 60 * 24       # 24 hours
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days


# ── Password hashing (PBKDF2 — no bcrypt dependency needed) ──────────────────
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return base64.b64encode(salt + key).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        raw  = base64.b64decode(hashed.encode())
        salt = raw[:16]
        stored_key = raw[16:]
        new_key = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt, 260_000)
        return hmac.compare_digest(stored_key, new_key)
    except Exception:
        return False


# ── Minimal JWT (header.payload.signature) ───────────────────────────────────
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)

def _sign(msg: str) -> str:
    return _b64url_encode(
        hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
    )

def create_token(data: dict, expires_minutes: int) -> str:
    header  = _b64url_encode(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    payload = dict(data)
    payload["exp"] = int(time.time()) + expires_minutes * 60
    payload["iat"] = int(time.time())
    body    = _b64url_encode(json.dumps(payload).encode())
    sig     = _sign(f"{header}.{body}")
    return f"{header}.{body}.{sig}"

def decode_token(token: str) -> Optional[dict]:
    try:
        header, body, sig = token.split(".")
        if not hmac.compare_digest(_sign(f"{header}.{body}"), sig):
            return None
        payload = json.loads(_b64url_decode(body))
        if payload.get("exp", 0) < int(time.time()):
            return None   # expired
        return payload
    except Exception:
        return None

def create_access_token(user_id: str, role: str) -> str:
    return create_token({"sub": user_id, "role": role, "type": "access"},
                        ACCESS_TOKEN_EXPIRE_MINUTES)

def create_refresh_token(user_id: str) -> str:
    return create_token({"sub": user_id, "type": "refresh"},
                        REFRESH_TOKEN_EXPIRE_MINUTES)