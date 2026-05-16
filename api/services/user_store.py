"""
services/user_store.py
In-memory user store. Swap for SQLite/Supabase in production.
"""

from datetime import datetime
from typing   import Dict, List, Optional
from services.auth_service import hash_password


class UserStore:
    _users: Dict[str, dict] = {}   # user_id → user record

    @classmethod
    def create(cls, email: str, password: str, name: str,
               role: str = "athlete") -> dict:
        if cls.get_by_email(email):
            raise ValueError("Email already registered")
        import uuid
        user_id = str(uuid.uuid4())
        record  = {
            "user_id":    user_id,
            "email":      email.lower().strip(),
            "name":       name,
            "role":       role,          # "athlete" | "admin"
            "password":   hash_password(password),
            "created_at": datetime.utcnow().isoformat(),
            "is_active":  True,
        }
        cls._users[user_id] = record
        return cls._safe(record)

    @classmethod
    def get(cls, user_id: str) -> Optional[dict]:
        r = cls._users.get(user_id)
        return cls._safe(r) if r else None

    @classmethod
    def get_by_email(cls, email: str) -> Optional[dict]:
        email = email.lower().strip()
        for r in cls._users.values():
            if r["email"] == email:
                return r   # includes password hash for verification
        return None

    @classmethod
    def all(cls) -> List[dict]:
        return [cls._safe(r) for r in cls._users.values()]

    @classmethod
    def update_role(cls, user_id: str, role: str) -> Optional[dict]:
        if user_id not in cls._users:
            return None
        cls._users[user_id]["role"] = role
        return cls._safe(cls._users[user_id])

    @classmethod
    def deactivate(cls, user_id: str):
        if user_id in cls._users:
            cls._users[user_id]["is_active"] = False

    @classmethod
    def _safe(cls, r: dict) -> dict:
        """Strip password hash before returning."""
        return {k: v for k, v in r.items() if k != "password"}


def seed_admin():
    """Create a default admin account if none exists."""
    if not UserStore.get_by_email("admin@injuryrisk.app"):
        UserStore.create(
            email    = "admin@injuryrisk.app",
            password = "Admin@1234",
            name     = "Admin",
            role     = "admin",
        )
        print("  Seeded default admin → admin@injuryrisk.app / Admin@1234")