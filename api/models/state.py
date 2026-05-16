"""
AppState — singleton that holds loaded model artifacts.
Loaded once at startup via lifespan, reused across all requests.
"""

import json
import joblib
from pathlib import Path

ARTIFACTS = Path(__file__).parent.parent.parent / "outputs"


class AppState:
    model  = None
    scaler = None
    meta: dict = {}

    @classmethod
    def load(cls):
        cls.model  = joblib.load(ARTIFACTS / "injury_model.joblib")
        cls.scaler = joblib.load(ARTIFACTS / "scaler.joblib")
        with open(ARTIFACTS / "model_meta.json") as f:
            cls.meta = json.load(f)

    @classmethod
    def is_ready(cls) -> bool:
        return cls.model is not None