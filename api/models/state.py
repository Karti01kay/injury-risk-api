"""
models/state.py
AppState — holds loaded model artifacts.
_artifacts_path can be overridden in main.py before load() is called.
"""

import json
import joblib
from pathlib import Path

_DEFAULT = Path(__file__).parent.parent / "outputs"


class AppState:
    model  = None
    scaler = None
    meta:  dict = {}

    # Overrideable — set in main.py for Render deployment
    _artifacts_path: Path = _DEFAULT

    @classmethod
    def load(cls):
        artifacts = cls._artifacts_path
        cls.model  = joblib.load(artifacts / "injury_model.joblib")
        cls.scaler = joblib.load(artifacts / "scaler.joblib")
        with open(artifacts / "model_meta.json") as f:
            cls.meta = json.load(f)

    @classmethod
    def is_ready(cls) -> bool:
        return cls.model is not None