"""
routes/athlete.py
CRUD for athlete profiles — all routes are auth-protected.
Users can only see and manage their own athletes.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing  import List

from models.schemas    import AthleteCreate, AthleteResponse
from models.auth_deps  import get_current_user
from services.athlete_store import AthleteStore

router = APIRouter()


@router.post("/athletes", response_model=AthleteResponse,
             status_code=status.HTTP_201_CREATED,
             summary="Create a new athlete profile")
async def create_athlete(body: AthleteCreate,
                         current_user: dict = Depends(get_current_user)):
    try:
        record = AthleteStore.create(body, owner_id=current_user["user_id"])
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return record


@router.get("/athletes", response_model=List[AthleteResponse],
            summary="List all athletes owned by the current user")
async def list_athletes(current_user: dict = Depends(get_current_user)):
    return AthleteStore.all_for_user(current_user["user_id"])


@router.get("/athletes/{athlete_id}", response_model=AthleteResponse,
            summary="Get a single athlete profile")
async def get_athlete(athlete_id: str,
                      current_user: dict = Depends(get_current_user)):
    athlete = AthleteStore.get(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlete not found.")
    if athlete.get("owner_id") != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    return athlete


@router.delete("/athletes/{athlete_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete an athlete and their history")
async def delete_athlete(athlete_id: str,
                         current_user: dict = Depends(get_current_user)):
    athlete = AthleteStore.get(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlete not found.")
    if athlete.get("owner_id") != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    AthleteStore.delete(athlete_id)