"""
Evidence Suite - Database Session Management
Supports PostgreSQL (production) and SQLite (testing).
"""
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


def get_database_url() -> str:
    """Get database URL from environment."""
    env = os.getenv("EVIDENCE_SUITE_ENV", "development")

    if env == "test":
        return os.getenv("SYNC_DATABASE_URL", "sqlite:///./test.db")

    return os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/evidence_suite"
    )


def get_async_database_url() -> str:
    """Get async database URL from environment."""
    env = os.getenv("EVIDENCE_SUITE_ENV", "development")

    if env == "test":
        url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://")
        return url

    url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/evidence_suite"
    )
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


def get_engine(url: Optional[str] = None):
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


def get_async_engine(url: Optional[str] = None):
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
        _session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_sync_engine()
        )
    return _session_local


def _get_async_session_local():
    """Get or create async session factory (lazy)."""
    global _async_session_local
    if _async_session_local is None:
        _async_session_local = async_sessionmaker(
            _get_async_engine_instance(),
            class_=AsyncSession,
            expire_on_commit=False
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
