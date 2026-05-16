"""
routes/predict.py
POST /api/v1/predict        — single prediction (auth required)
POST /api/v1/predict/batch  — batch prediction  (auth required)
"""

from fastapi  import APIRouter, HTTPException, status, Depends
from datetime import datetime
from typing   import List

from models.schemas              import PredictRequest, PredictResponse
from models.state                import AppState
from models.auth_deps            import get_current_user
from services.prediction_service import predict
from services.athlete_store      import AthleteStore

router = APIRouter()


@router.post("/predict", response_model=PredictResponse,
             summary="Predict injury risk for a single athlete log entry")
async def predict_risk(req: PredictRequest,
                       current_user: dict = Depends(get_current_user)):
    if not AppState.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Verify the athlete belongs to the requesting user (or user is admin)
    if AthleteStore.exists(req.athlete_id):
        if not AthleteStore.owns(req.athlete_id, current_user["user_id"]) \
                and current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Access denied.")

    response = predict(req)

    if AthleteStore.exists(req.athlete_id):
        AthleteStore.append_history(req.athlete_id, {
            "date":          datetime.utcnow().strftime("%Y-%m-%d"),
            "risk_score":    response.risk_score,
            "risk_category": response.risk_category.value,
            "acwr":          response.acwr,
            "daily_load":    req.daily_load,
            "sleep_hours":   req.sleep_hours,
            "soreness":      req.soreness,
        })
    return response


@router.post("/predict/batch", response_model=List[PredictResponse],
             summary="Predict injury risk for multiple athletes at once")
async def predict_batch(requests: List[PredictRequest],
                        current_user: dict = Depends(get_current_user)):
    if not AppState.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(requests) > 50:
        raise HTTPException(status_code=422, detail="Batch limit is 50 athletes.")

    # For non-admins, verify every athlete in the batch
    if current_user["role"] != "admin":
        for r in requests:
            if AthleteStore.exists(r.athlete_id) and \
                    not AthleteStore.owns(r.athlete_id, current_user["user_id"]):
                raise HTTPException(status_code=403,
                                    detail=f"Access denied for athlete {r.athlete_id}.")
    return [predict(r) for r in requests]