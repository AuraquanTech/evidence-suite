"""
Evidence Suite - JWT Authentication
With rate limiting for security.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
from uuid import UUID
from collections import defaultdict
import time
import asyncio

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import api_settings
from core.database import User
from core.database.session import get_db


# Rate limiting storage (in-memory, use Redis in production)
_rate_limit_store: Dict[str, list] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()

# Rate limit settings
RATE_LIMIT_WINDOW = 60  # 1 minute window
RATE_LIMIT_MAX_ATTEMPTS = 5  # Max 5 attempts per window
LOCKOUT_DURATION = 300  # 5 minute lockout after max attempts


async def check_rate_limit(identifier: str, request: Request) -> None:
    """
    Check rate limit for an identifier (IP or email).
    Raises HTTPException if rate limit exceeded.
    """
    async with _rate_limit_lock:
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW

        # Clean old entries
        _rate_limit_store[identifier] = [
            t for t in _rate_limit_store[identifier] if t > window_start
        ]

        # Check if locked out
        attempts = len(_rate_limit_store[identifier])
        if attempts >= RATE_LIMIT_MAX_ATTEMPTS:
            oldest_attempt = min(_rate_limit_store[identifier])
            lockout_end = oldest_attempt + LOCKOUT_DURATION
            if now < lockout_end:
                wait_time = int(lockout_end - now)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many login attempts. Try again in {wait_time} seconds.",
                    headers={"Retry-After": str(wait_time)},
                )
            # Lockout expired, clear attempts
            _rate_limit_store[identifier] = []

        # Record this attempt
        _rate_limit_store[identifier].append(now)


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


class UserCreate(BaseModel):
    """User creation model."""
    email: str
    password: str
    name: Optional[str] = None
    role: str = "analyst"


class UserResponse(BaseModel):
    """User response model."""
    id: UUID
    email: str
    name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=api_settings.jwt_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        api_settings.jwt_secret,
        algorithm=api_settings.jwt_algorithm
    )
    return encoded_jwt


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user = await get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current authenticated user."""
    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            api_settings.jwt_secret,
            algorithms=[api_settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        token_data = TokenData(user_id=user_id)
    except JWTError:
        return None

    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalar_one_or_none()

    if user and not user.is_active:
        return None

    return user


async def get_current_active_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Get current active user, raise exception if not authenticated."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_role(*roles: str):
    """Dependency to require specific role(s)."""
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


# Route handlers (to be added to router)
from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token. Rate limited to prevent brute force."""
    # Rate limit by IP address
    client_ip = get_client_ip(request)
    await check_rate_limit(f"login_ip:{client_ip}", request)

    # Also rate limit by email to prevent credential stuffing
    await check_rate_limit(f"login_email:{form_data.username}", request)

    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role}
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=api_settings.jwt_expire_minutes * 60
    )


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user. Rate limited to prevent spam."""
    # Rate limit registration by IP
    client_ip = get_client_ip(request)
    await check_rate_limit(f"register_ip:{client_ip}", request)

    # Check if user exists
    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Create user
    user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=get_password_hash(user_data.password),
        role=user_data.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )
