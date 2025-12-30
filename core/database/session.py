"""Evidence Suite - Database Session Management
Supports PostgreSQL (production) and SQLite (testing).
Includes connection retry with exponential backoff.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import TypeVar

from loguru import logger
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base


# Retry configuration
MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("DB_RETRY_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.getenv("DB_RETRY_MAX_DELAY", "30.0"))

T = TypeVar("T")


def get_database_url() -> str:
    """Get database URL from environment."""
    env = os.getenv("EVIDENCE_SUITE_ENV", "development")

    if env == "test":
        return os.getenv("SYNC_DATABASE_URL", "sqlite:///./test.db")

    url = os.getenv("DATABASE_URL", "")

    # In production, require explicit DATABASE_URL
    if env == "production" and not url:
        raise RuntimeError(
            "DATABASE_URL must be set in production mode. "
            "Set EVIDENCE_SUITE_ENV=development for local development."
        )

    # Block SQLite in production
    if env == "production" and url and "sqlite" in url.lower():
        raise RuntimeError("SQLite is not supported in production mode. Use PostgreSQL.")

    return url or "postgresql://postgres:postgres@localhost:5432/evidence_suite"


def get_async_database_url() -> str:
    """Get async database URL from environment."""
    env = os.getenv("EVIDENCE_SUITE_ENV", "development")

    if env == "test":
        url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://")
        return url

    url = os.getenv("DATABASE_URL", "")

    # In production, require explicit DATABASE_URL
    if env == "production" and not url:
        raise RuntimeError(
            "DATABASE_URL must be set in production mode. "
            "Set EVIDENCE_SUITE_ENV=development for local development."
        )

    # Block SQLite in production
    if env == "production" and url and "sqlite" in url.lower():
        raise RuntimeError("SQLite is not supported in production mode. Use PostgreSQL.")

    if not url:
        url = "postgresql://postgres:postgres@localhost:5432/evidence_suite"

    # Convert to async URL if needed
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    return url


# Database URLs
DATABASE_URL = get_database_url()
ASYNC_DATABASE_URL = get_async_database_url()

# Track if we're in test mode
IS_TEST = os.getenv("EVIDENCE_SUITE_ENV") == "test"
IS_SQLITE = "sqlite" in DATABASE_URL


def _get_engine_kwargs():
    """Get engine keyword arguments based on database type."""
    if IS_SQLITE:
        return {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    return {
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
    }


def _get_async_engine_kwargs():
    """Get async engine keyword arguments based on database type."""
    if IS_SQLITE:
        return {}
    return {
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
    }


def get_engine(url: str | None = None):
    """Create synchronous database engine."""
    db_url = url or DATABASE_URL
    eng = create_engine(db_url, **_get_engine_kwargs())

    # Enable foreign keys for SQLite
    if "sqlite" in db_url:

        @event.listens_for(eng, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return eng


def get_async_engine(url: str | None = None):
    """Create async database engine."""
    db_url = url or ASYNC_DATABASE_URL
    return create_async_engine(db_url, **_get_async_engine_kwargs())


# Lazy initialization - only create engines when needed
_engine = None
_async_engine = None
_session_local = None
_async_session_local = None


def _get_sync_engine():
    """Get or create synchronous engine (lazy)."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _get_async_engine_instance():
    """Get or create async engine (lazy)."""
    global _async_engine
    if _async_engine is None:
        _async_engine = get_async_engine()
    return _async_engine


def _get_session_local():
    """Get or create sync session factory (lazy)."""
    global _session_local
    if _session_local is None:
        _session_local = sessionmaker(autocommit=False, autoflush=False, bind=_get_sync_engine())
    return _session_local


def _get_async_session_local():
    """Get or create async session factory (lazy)."""
    global _async_session_local
    if _async_session_local is None:
        _async_session_local = async_sessionmaker(
            _get_async_engine_instance(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_local


# Backward compatibility properties
@property
def engine():
    return _get_sync_engine()


@property
def async_engine():
    return _get_async_engine_instance()


def init_db():
    """Initialize database tables."""
    eng = _get_sync_engine()
    Base.metadata.create_all(bind=eng)


async def init_db_async():
    """Initialize database tables (async)."""
    eng = _get_async_engine_instance()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> Session:
    """Get synchronous database session."""
    SessionLocal = _get_session_local()
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    AsyncSessionLocal = _get_async_session_local()
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    AsyncSessionLocal = _get_async_session_local()
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Export for backward compatibility
def get_async_session_local():
    """Get async session factory (for backward compatibility)."""
    return _get_async_session_local()


async def with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
    **kwargs,
) -> T:
    """Execute a database operation with exponential backoff retry.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except OperationalError as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Database operation failed after {max_retries} retries: {e}")
                raise

            # Calculate delay with exponential backoff and jitter
            import random

            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)

            logger.warning(
                f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)
        except Exception:
            # Don't retry non-operational errors
            raise

    raise last_exception  # type: ignore


async def test_connection() -> bool:
    """Test database connection.

    Returns:
        True if connection successful
    """
    try:
        engine = _get_async_engine_instance()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def wait_for_database(max_wait: float = 60.0, interval: float = 2.0) -> bool:
    """Wait for database to become available.

    Args:
        max_wait: Maximum time to wait in seconds
        interval: Time between connection attempts

    Returns:
        True if database became available
    """
    import time

    start = time.time()

    while time.time() - start < max_wait:
        if await test_connection():
            logger.info("Database connection established")
            return True

        logger.info(f"Waiting for database... ({time.time() - start:.0f}s / {max_wait:.0f}s)")
        await asyncio.sleep(interval)

    logger.error(f"Database not available after {max_wait}s")
    return False
