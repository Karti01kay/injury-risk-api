"""
Pydantic schemas for auth endpoints.
"""

from pydantic import BaseModel, Field, field_validator
import re


class SignupRequest(BaseModel):
    name:     str   = Field(..., min_length=2, max_length=60)
    email:    str   = Field(...)
    password: str   = Field(..., min_length=8)
    role:     str   = Field("athlete")   # "athlete" | "admin"

    @field_validator("email")
    @classmethod
    def valid_email(cls, v):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email address")
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def strong_password(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v

    @field_validator("role")
    @classmethod
    def valid_role(cls, v):
        if v not in ("athlete", "admin"):
            raise ValueError("Role must be 'athlete' or 'admin'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Alex Johnson",
                "email": "alex@example.com",
                "password": "Secure@123",
                "role": "athlete"
            }
        }


class LoginRequest(BaseModel):
    email:    str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "alex@example.com",
                "password": "Secure@123"
            }
        }


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user_id:       str
    name:          str
    email:         str
    role:          str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    user_id:    str
    name:       str
    email:      str
    role:       str
    created_at: str
    is_active:  bool


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(athlete|admin)$")