"""
routes/auth.py
POST /api/v1/auth/signup
POST /api/v1/auth/login
POST /api/v1/auth/refresh
GET  /api/v1/auth/me
"""

from fastapi import APIRouter, HTTPException, status, Depends
from models.auth_schemas   import (SignupRequest, LoginRequest,
                                   TokenResponse, RefreshRequest, UserResponse)
from models.auth_deps      import get_current_user
from services.user_store   import UserStore
from services.auth_service import (verify_password, create_access_token,
                                   create_refresh_token, decode_token)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/signup", response_model=TokenResponse,
             status_code=status.HTTP_201_CREATED,
             summary="Register a new user account")
async def signup(body: SignupRequest):
    try:
        user = UserStore.create(
            email=body.email, password=body.password,
            name=body.name,   role=body.role,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return TokenResponse(
        access_token  = create_access_token(user["user_id"], user["role"]),
        refresh_token = create_refresh_token(user["user_id"]),
        user_id       = user["user_id"],
        name          = user["name"],
        email         = user["email"],
        role          = user["role"],
    )


@router.post("/login", response_model=TokenResponse,
             summary="Login and get JWT tokens")
async def login(body: LoginRequest):
    # get_by_email returns record WITH password hash
    record = UserStore.get_by_email(body.email)
    if not record or not verify_password(body.password, record["password"]):
        raise HTTPException(status_code=401,
                            detail="Incorrect email or password")
    if not record.get("is_active"):
        raise HTTPException(status_code=403, detail="Account is deactivated")

    return TokenResponse(
        access_token  = create_access_token(record["user_id"], record["role"]),
        refresh_token = create_refresh_token(record["user_id"]),
        user_id       = record["user_id"],
        name          = record["name"],
        email         = record["email"],
        role          = record["role"],
    )


@router.post("/refresh", response_model=TokenResponse,
             summary="Exchange refresh token for a new access token")
async def refresh(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    user = UserStore.get(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token  = create_access_token(user["user_id"], user["role"]),
        refresh_token = create_refresh_token(user["user_id"]),
        user_id       = user["user_id"],
        name          = user["name"],
        email         = user["email"],
        role          = user["role"],
    )


@router.get("/me", response_model=UserResponse,
            summary="Get current authenticated user")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user