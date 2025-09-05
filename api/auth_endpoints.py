"""Authentication endpoints for MPS Connect."""

# pylint: disable=import-error,no-name-in-module
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session  # type: ignore

from database.connection import get_db  # type: ignore
from database.models import User  # type: ignore
from security.auth import (  # type: ignore
    authenticate_user,  # type: ignore
    create_user,  # type: ignore
    create_access_token,  # type: ignore
    create_refresh_token,  # type: ignore
    verify_token,  # type: ignore
    get_current_user,  # type: ignore
    create_session,  # type: ignore
    revoke_session,  # type: ignore
    hash_password,  # type: ignore
    verify_password,  # type: ignore
)
from security.audit import log_user_activity  # type: ignore
from security.encryption import validate_password_strength  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()


# Pydantic models
class UserCreate(BaseModel):
    """User creation request model."""

    email: EmailStr
    password: str
    name: str
    role: str = "user"


class UserLogin(BaseModel):
    """User login request model."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """User response model."""

    id: UUID
    email: str
    name: str
    role: str
    is_active: bool
    created_at: str


class PasswordChange(BaseModel):
    """Password change request model."""

    current_password: str
    new_password: str


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(
    user_data: UserCreate, request: Request, db: Session = Depends(get_db)
):
    """Register a new user.

    What user registration provides:
    - Account creation with validation
    - Password strength enforcement
    - Role-based access control
    - Audit trail creation
    """
    try:
        # Validate password strength
        password_validation = validate_password_strength(user_data.password)
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Password does not meet security requirements",
                    "issues": password_validation["issues"],
                    "recommendations": password_validation["recommendations"],
                },
            )

        # Create user
        user = create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            name=user_data.name,
            role=user_data.role,
        )

        # Log user activity
        log_user_activity(
            db=db,
            user_id=user.id,
            activity_type="user_registration",
            description=f"User registered with email {user_data.email}",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("User registration failed: %s", e)
        raise HTTPException(status_code=500, detail="Registration failed") from e


@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin, request: Request, db: Session = Depends(get_db)
):
    """Authenticate user and return tokens.

    What user login provides:
    - Secure authentication
    - JWT token generation
    - Session management
    - Security logging
    """
    try:
        # Authenticate user
        user = authenticate_user(db, login_data.email, login_data.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create tokens
        access_token = create_access_token({"sub": str(user.id), "email": user.email})
        refresh_token = create_refresh_token({"sub": str(user.id)})

        # Create session
        create_session(db, user, request)

        # Log user activity
        log_user_activity(
            db=db,
            user_id=user.id,
            activity_type="user_login",
            description=f"User logged in from {request.client.host if request.client else 'unknown'}",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=30 * 60,  # 30 minutes
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("User login failed: %s", e)
        raise HTTPException(status_code=500, detail="Login failed") from e


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token_param: str, request: Request, db: Session = Depends(get_db)
):
    """Refresh access token using refresh token.

    What token refresh provides:
    - Extended session without re-authentication
    - Security through token rotation
    - Reduced authentication overhead
    - Better user experience
    """
    try:
        # Verify refresh token
        payload = verify_token(refresh_token_param, token_type="refresh")
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Get user
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()  # type: ignore
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        # Create new tokens
        access_token = create_access_token({"sub": str(user.id), "email": user.email})
        new_refresh_token = create_refresh_token({"sub": str(user.id)})

        # Log activity
        log_user_activity(
            db=db,
            user_id=user.id,
            activity_type="token_refresh",
            description="Access token refreshed",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=30 * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed: %s", e)
        raise HTTPException(status_code=500, detail="Token refresh failed") from e


@router.post("/logout")
async def logout_user(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Logout user and revoke session.

    What user logout provides:
    - Session invalidation
    - Security cleanup
    - Audit trail completion
    - Token revocation
    """
    try:
        # Get session token from request
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

            # Revoke session
            revoke_session(db, token)

        # Log activity
        log_user_activity(
            db=db,
            user_id=current_user.id,
            activity_type="user_logout",
            description="User logged out",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return {"message": "Successfully logged out"}

    except Exception as e:
        logger.error("User logout failed: %s", e)
        raise HTTPException(status_code=500, detail="Logout failed") from e


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information.

    What user info provides:
    - User profile data
    - Role and permissions
    - Account status
    - Security context
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change user password.

    What password change provides:
    - Secure password updates
    - Current password verification
    - Password strength validation
    - Security logging
    """
    try:
        # Verify current password
        if not verify_password(
            password_data.current_password, current_user.password_hash
        ):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Validate new password strength
        password_validation = validate_password_strength(password_data.new_password)
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "New password does not meet security requirements",
                    "issues": password_validation["issues"],
                    "recommendations": password_validation["recommendations"],
                },
            )

        # Update password
        current_user.password_hash = hash_password(password_data.new_password)
        db.commit()

        # Log activity
        log_user_activity(
            db=db,
            user_id=current_user.id,
            activity_type="password_change",
            description="Password changed successfully",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return {"message": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change failed: %s", e)
        raise HTTPException(status_code=500, detail="Password change failed") from e


@router.get("/validate-token")
async def validate_token(current_user: User = Depends(get_current_user)):
    """Validate current token.

    What token validation provides:
    - Token validity check
    - User status verification
    - Security context validation
    - Session health check
    """
    return {
        "valid": True,
        "user_id": str(current_user.id),
        "email": current_user.email,
        "role": current_user.role,
    }
