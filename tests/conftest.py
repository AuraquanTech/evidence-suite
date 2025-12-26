"""
Evidence Suite - Test Configuration
Pytest fixtures and configuration for testing.
"""

import os
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment BEFORE any imports
os.environ["EVIDENCE_SUITE_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["SYNC_DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = ""  # Disable Redis in tests

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

# Import after environment is set
from core.database.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create database tables before tests run."""
    # Create sync engine for table creation
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///./test.db",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield

    # Cleanup - drop all tables
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

    # Remove test database file
    test_db = Path("./test.db")
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass


@pytest.fixture
async def client():
    """Create async test client."""
    # Import here to ensure environment is set
    from api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
