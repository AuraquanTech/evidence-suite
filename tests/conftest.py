"""Evidence Suite - Pytest Configuration and Fixtures

Provides shared fixtures for all tests including database sessions,
test clients, and authenticated users.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# Set test environment before importing app modules
os.environ["EVIDENCE_SUITE_ENV"] = "test"
os.environ["API_JWT_SECRET"] = "test-secret-key-for-testing-only"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test.

    Uses SQLite in-memory for fast, isolated tests.
    """
    from core.database import Base

    # Use in-memory SQLite for tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database session override."""
    from api.main import app
    from core.database.session import get_db_session

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Create a test user in the database."""
    from passlib.hash import bcrypt

    from core.database import User

    user = User(
        id=uuid4(),
        email="test@example.com",
        password_hash=bcrypt.hash("testpassword123"),
        name="Test User",
        role="analyst",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession):
    """Create an admin user in the database."""
    from passlib.hash import bcrypt

    from core.database import User

    user = User(
        id=uuid4(),
        email="admin@example.com",
        password_hash=bcrypt.hash("adminpassword123"),
        name="Admin User",
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def auth_headers(test_user) -> dict[str, str]:
    """Get authorization headers for authenticated requests."""
    from api.auth import create_access_token

    token = create_access_token(data={"sub": test_user.email, "role": test_user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def admin_auth_headers(admin_user) -> dict[str, str]:
    """Get authorization headers for admin requests."""
    from api.auth import create_access_token

    token = create_access_token(data={"sub": admin_user.email, "role": admin_user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def test_case(db_session: AsyncSession):
    """Create a test case in the database."""
    from core.database import Case, CaseStatus

    case = Case(
        id=uuid4(),
        case_number="TEST-0001",
        title="Test Case",
        description="A test case for unit tests",
        status=CaseStatus.ACTIVE,
        client_name="Test Client",
    )
    db_session.add(case)
    await db_session.commit()
    await db_session.refresh(case)
    return case


@pytest_asyncio.fixture
async def test_evidence(db_session: AsyncSession, test_case):
    """Create test evidence in the database."""
    import hashlib

    from core.database import EvidenceRecord, EvidenceStatus, EvidenceTypeDB

    evidence = EvidenceRecord(
        id=uuid4(),
        case_id=test_case.id,
        evidence_type=EvidenceTypeDB.DOCUMENT,
        original_filename="test.pdf",
        mime_type="application/pdf",
        file_size_bytes=1024,
        original_hash=hashlib.sha256(b"test content").hexdigest(),
        storage_path="./evidence_store/test/test.pdf",
        status=EvidenceStatus.PENDING,
    )
    db_session.add(evidence)
    await db_session.commit()
    await db_session.refresh(evidence)
    return evidence


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
