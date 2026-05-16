"""
services/athlete_store.py
In-memory athlete store scoped by user_id.
Each user can own multiple athlete profiles.
"""

from datetime import datetime
from typing   import Dict, List, Optional
from models.schemas import AthleteCreate


class AthleteStore:
    _athletes:  Dict[str, dict]       = {}
    _histories: Dict[str, List[dict]] = {}

    @classmethod
    def create(cls, data: AthleteCreate, owner_id: str) -> dict:
        if cls.exists(data.athlete_id):
            raise ValueError(f"Athlete '{data.athlete_id}' already exists.")
        record = data.model_dump()
        record["owner_id"]   = owner_id
        record["created_at"] = datetime.utcnow().isoformat()
        cls._athletes[data.athlete_id] = record
        cls._histories.setdefault(data.athlete_id, [])
        return record

    @classmethod
    def get(cls, athlete_id: str) -> Optional[dict]:
        return cls._athletes.get(athlete_id)

    @classmethod
    def all_for_user(cls, owner_id: str) -> List[dict]:
        return [a for a in cls._athletes.values() if a.get("owner_id") == owner_id]

    @classmethod
    def all(cls) -> List[dict]:
        return list(cls._athletes.values())

    @classmethod
    def exists(cls, athlete_id: str) -> bool:
        return athlete_id in cls._athletes

    @classmethod
    def owns(cls, athlete_id: str, owner_id: str) -> bool:
        a = cls._athletes.get(athlete_id)
        return a is not None and a.get("owner_id") == owner_id

    @classmethod
    def delete(cls, athlete_id: str):
        cls._athletes.pop(athlete_id, None)
        cls._histories.pop(athlete_id, None)

    @classmethod
    def count_for_user(cls, owner_id: str) -> int:
        return len(cls.all_for_user(owner_id))

    @classmethod
    def append_history(cls, athlete_id: str, entry: dict):
        cls._histories.setdefault(athlete_id, [])
        cls._histories[athlete_id].append(entry)

    @classmethod
    def get_history(cls, athlete_id: str, limit: int = 30) -> List[dict]:
        history = cls._histories.get(athlete_id, [])
        return sorted(history, key=lambda x: x["date"], reverse=True)[:limit]

    @classmethod
    def history_count(cls, athlete_id: str) -> int:
        return len(cls._histories.get(athlete_id, []))