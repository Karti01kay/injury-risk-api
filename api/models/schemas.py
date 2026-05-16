"""
Pydantic schemas for request validation and response serialisation.
"""

from pydantic import BaseModel, Field, field_validator
from typing   import Optional, List
from enum     import Enum


class SportType(str, Enum):
    running    = "running"
    cycling    = "cycling"
    football   = "football"
    basketball = "basketball"
    swimming   = "swimming"
    other      = "other"


class RiskCategory(str, Enum):
    low      = "low"
    elevated = "elevated"
    moderate = "moderate"
    high     = "high"


# ── Prediction request ───────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Athlete profile
    athlete_id:   str        = Field(...,  description="Unique athlete identifier")
    age:          int        = Field(25,   ge=14, le=80)
    sport:        SportType  = Field(SportType.running)
    prev_injury:  int        = Field(0,    ge=0, le=1, description="1 = has prior injury history")

    # Today's training
    daily_load:   float = Field(...,  ge=0, le=200, description="Session load in arbitrary units (RPE × duration)")
    is_rest_day:  int   = Field(0,    ge=0, le=1)

    # Rolling load (last 7 / 28 days) — app computes these
    acute_load_7d:    float = Field(..., ge=0, description="Average daily load over last 7 days")
    chronic_load_28d: float = Field(..., ge=0, description="Average daily load over last 28 days")

    # Wellness inputs (self-reported)
    sleep_hours:   float = Field(7.0, ge=0, le=12)
    sleep_quality: float = Field(3.0, ge=1, le=5,  description="1=terrible 5=excellent")
    soreness:      float = Field(2.0, ge=1, le=5,  description="1=none 5=severe")
    resting_hr:    float = Field(60,  ge=30, le=120)

    # Derived features the app can compute from its local history
    sleep_deficit_7d:  float = Field(0.0, ge=0)
    rhr_trend:         float = Field(0.0, description="Current RHR minus 7-day average")
    soreness_7d:       float = Field(2.0, ge=1, le=5)
    days_since_rest:   int   = Field(1,   ge=0, le=30)
    load_monotony:     float = Field(1.0, ge=0)
    training_strain:   float = Field(50,  ge=0)
    load_spike:        float = Field(1.0, ge=0)

    @field_validator("chronic_load_28d")
    @classmethod
    def chronic_nonzero(cls, v):
        return max(v, 0.01)   # prevent division-by-zero in ACWR

    @property
    def acwr(self) -> float:
        return round(self.acute_load_7d / self.chronic_load_28d, 4)

    class Config:
        json_schema_extra = {
            "example": {
                "athlete_id":        "athlete_007",
                "age":               26,
                "sport":             "running",
                "prev_injury":       1,
                "daily_load":        85,
                "is_rest_day":       0,
                "acute_load_7d":     80,
                "chronic_load_28d":  55,
                "sleep_hours":       5.5,
                "sleep_quality":     2,
                "soreness":          4.2,
                "resting_hr":        68,
                "sleep_deficit_7d":  7.0,
                "rhr_trend":         8.0,
                "soreness_7d":       3.8,
                "days_since_rest":   8,
                "load_monotony":     1.4,
                "training_strain":   112,
                "load_spike":        1.7,
            }
        }


# ── Prediction response ──────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    athlete_id:       str
    risk_score:       float         = Field(..., description="Injury probability 0–100")
    risk_category:    RiskCategory
    acwr:             float         = Field(..., description="Acute:Chronic Workload Ratio")
    recommendations:  List[str]
    top_risk_factors: List[str]
    model_version:    str


# ── Athlete CRUD ─────────────────────────────────────────────────────────────
class AthleteCreate(BaseModel):
    athlete_id:  str
    name:        str
    age:         int       = Field(ge=14, le=80)
    sport:       SportType
    prev_injury: int       = Field(0, ge=0, le=1)


class AthleteResponse(AthleteCreate):
    created_at: str


# ── History ──────────────────────────────────────────────────────────────────
class HistoryEntry(BaseModel):
    date:          str
    risk_score:    float
    risk_category: RiskCategory
    acwr:          float
    daily_load:    float
    sleep_hours:   float
    soreness:      float


class HistoryResponse(BaseModel):
    athlete_id: str
    entries:    List[HistoryEntry]
    total:      int
