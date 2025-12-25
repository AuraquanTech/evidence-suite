"""
Evidence Suite - Database Session Management
"""
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

# Database URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/evidence_suite"
)

# Convert to async URL if needed
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


def get_engine(url: str = DATABASE_URL):
    """Create synchronous database engine."""
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


def get_async_engine(url: str = ASYNC_DATABASE_URL):
    """Create async database engine."""
    return create_async_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


# Session factories
engine = get_engine()
async_engine = get_async_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


async def init_db_async():
    """Initialize database tables (async)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> Session:
    """Get synchronous database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
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
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
