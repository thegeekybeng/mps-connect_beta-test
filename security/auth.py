"""Authentication and authorization module for MPS Connect."""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import User, Session as DBSession, Permission

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "mps-connect-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()


class AuthenticationError(Exception):
    """Custom authentication error."""

    pass


class AuthorizationError(Exception):
    """Custom authorization error."""

    pass


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string

    What bcrypt does functionally:
    - One-way hashing (cannot be reversed)
    - Salt generation (prevents rainbow table attacks)
    - Adaptive cost (slows down brute force attacks)
    - Time-constant comparison (prevents timing attacks)
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored hashed password

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token.

    Args:
        data: User data to encode in token
        expires_delta: Token expiration time

    Returns:
        JWT token string

    What JWT does functionally:
    - Self-contained token with user information
    - Digital signature prevents tampering
    - Stateless authentication (no server storage)
    - Cross-domain compatibility
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token.

    Args:
        data: User data to encode in token

    Returns:
        JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify and decode a JWT token.

    Args:
        token: JWT token to verify
        token_type: Expected token type ("access" or "refresh")

    Returns:
        Decoded token payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Verify token type
        if payload.get("type") != token_type:
            raise AuthenticationError("Invalid token type")

        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise AuthenticationError("Token expired")

        return payload

    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user with email and password.

    Args:
        db: Database session
        email: User email
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


def create_user(
    db: Session, email: str, password: str, name: str, role: str = "user"
) -> User:
    """Create a new user.

    Args:
        db: Database session
        email: User email
        password: Plain text password
        name: User full name
        role: User role

    Returns:
        Created User object

    Raises:
        ValueError: If user already exists
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise ValueError("User with this email already exists")

    # Create new user
    user = User(
        email=email, password_hash=hash_password(password), name=name, role=role
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("User created: %s", email)
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        Current User object

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Verify token
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")

        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # Get user from database
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise AuthenticationError("User not found")

        return user

    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


def require_auth(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to require authentication.

    Args:
        current_user: Current authenticated user

    Returns:
        User object

    Raises:
        HTTPException: If user is not authenticated
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


def require_role(required_role: str):
    """Dependency factory to require specific role.

    Args:
        required_role: Required user role

    Returns:
        Dependency function

    What Role-Based Access Control (RBAC) does:
    - Assigns permissions based on user roles
    - Centralizes access control logic
    - Simplifies permission management
    - Supports hierarchical permissions
    """

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=403, detail=f"Role '{required_role}' required"
            )
        return current_user

    return role_checker


def check_permission(db: Session, user: User, resource: str, action: str) -> bool:
    """Check if user has permission for resource and action.

    Args:
        db: Database session
        user: User object
        resource: Resource name
        action: Action name

    Returns:
        True if user has permission, False otherwise

    What Permission System does:
    - Granular access control
    - Resource-action based permissions
    - Database-driven permission management
    - Supports complex authorization rules
    """
    # Admin users have all permissions
    if user.role == "admin":
        return True

    # Check specific permission
    permission = (
        db.query(Permission)
        .filter(
            Permission.role == user.role,
            Permission.resource == resource,
            Permission.action == action,
        )
        .first()
    )

    return permission is not None


def create_session(db: Session, user: User, request: Request) -> DBSession:
    """Create a new user session.

    Args:
        db: Database session
        user: User object
        request: FastAPI request object

    Returns:
        Created session object
    """
    # Generate session token
    session_token = create_refresh_token({"sub": str(user.id)})

    # Create session record
    session = DBSession(
        user_id=user.id,
        session_token=session_token,
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info("Session created for user: %s", user.email)
    return session


def revoke_session(db: Session, session_token: str) -> bool:
    """Revoke a user session.

    Args:
        db: Database session
        session_token: Session token to revoke

    Returns:
        True if session was revoked, False if not found
    """
    session = (
        db.query(DBSession)
        .filter(DBSession.session_token == session_token, DBSession.is_active == True)
        .first()
    )

    if session:
        session.is_active = False
        db.commit()
        logger.info("Session revoked: %s", session_token)
        return True

    return False


def cleanup_expired_sessions(db: Session) -> int:
    """Clean up expired sessions.

    Args:
        db: Database session

    Returns:
        Number of sessions cleaned up
    """
    expired_sessions = (
        db.query(DBSession).filter(DBSession.expires_at < datetime.utcnow()).all()
    )

    count = len(expired_sessions)
    for session in expired_sessions:
        session.is_active = False

    db.commit()
    logger.info("Cleaned up %d expired sessions", count)
    return count
