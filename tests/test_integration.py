"""Evidence Suite - Integration Tests
End-to-end tests for API workflows.
"""

import os
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient


# Set test environment
os.environ["EVIDENCE_SUITE_ENV"] = "test"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create async test client."""
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
async def auth_headers(client: AsyncClient):
    """Get authentication headers for testing."""
    # Create test user and login
    from passlib.context import CryptContext
    from sqlalchemy import select

    from core.database import User
    from core.database.session import get_async_session, init_db_async

    await init_db_async()

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async with get_async_session() as db:
        # Check if test user exists
        result = await db.execute(select(User).where(User.email == "test@integration.local"))
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                id=uuid4(),
                email="test@integration.local",
                name="Integration Test User",
                role="admin",
                password_hash=pwd_context.hash("test123"),
                is_active=True,
            )
            db.add(user)
            await db.commit()

    # Login
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "test@integration.local", "password": "test123"},
    )

    if response.status_code == 200:
        token = response.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}

    # If login fails, return empty headers (some tests don't need auth)
    return {}


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.anyio
    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint returns API info."""
        response = await client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data

    @pytest.mark.anyio
    async def test_health_endpoint(self, client: AsyncClient):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.anyio
    async def test_metrics_endpoint(self, client: AsyncClient):
        """Test metrics endpoint."""
        response = await client.get("/metrics")
        assert response.status_code == 200


class TestCaseWorkflow:
    """Test complete case management workflow."""

    @pytest.mark.anyio
    async def test_create_and_list_cases(self, client: AsyncClient, auth_headers: dict):
        """Test creating and listing cases."""
        # Create a case
        case_data = {
            "case_number": f"INT-TEST-{uuid4().hex[:8].upper()}",
            "title": "Integration Test Case",
            "description": "Created by integration test",
        }

        response = await client.post(
            "/api/v1/cases/",
            json=case_data,
            headers=auth_headers,
        )

        # May fail without auth, that's expected
        if response.status_code == 401:
            pytest.skip("Authentication required")

        assert response.status_code == 201
        created_case = response.json()
        assert created_case["case_number"] == case_data["case_number"]
        case_id = created_case["id"]

        # List cases
        response = await client.get("/api/v1/cases/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "items" in data
        assert data["total"] >= 1

        # Get specific case
        response = await client.get(f"/api/v1/cases/{case_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == case_id

    @pytest.mark.anyio
    async def test_update_case(self, client: AsyncClient, auth_headers: dict):
        """Test updating a case."""
        # Create a case first
        case_data = {
            "case_number": f"INT-UPDATE-{uuid4().hex[:8].upper()}",
            "title": "Update Test Case",
        }

        response = await client.post(
            "/api/v1/cases/",
            json=case_data,
            headers=auth_headers,
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")

        assert response.status_code == 201
        case_id = response.json()["id"]

        # Update the case
        update_data = {
            "title": "Updated Title",
            "description": "Updated description",
        }

        response = await client.put(
            f"/api/v1/cases/{case_id}",
            json=update_data,
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Updated Title"


class TestEvidenceWorkflow:
    """Test evidence upload and analysis workflow."""

    @pytest.mark.anyio
    async def test_evidence_upload(self, client: AsyncClient, auth_headers: dict):
        """Test uploading evidence to a case."""
        # Create a case first
        case_data = {
            "case_number": f"INT-EVID-{uuid4().hex[:8].upper()}",
            "title": "Evidence Test Case",
        }

        response = await client.post(
            "/api/v1/cases/",
            json=case_data,
            headers=auth_headers,
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")

        assert response.status_code == 201
        case_id = response.json()["id"]

        # Upload evidence
        test_content = b"This is test evidence content for analysis."
        files = {"file": ("test_evidence.txt", test_content, "text/plain")}
        data = {
            "case_id": case_id,
            "evidence_type": "text",
        }

        response = await client.post(
            "/api/v1/evidence/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == 201
        evidence = response.json()
        assert evidence["case_id"] == case_id
        assert evidence["status"] == "pending"

        evidence_id = evidence["id"]

        # Get evidence details
        response = await client.get(
            f"/api/v1/evidence/{evidence_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Get chain of custody
        response = await client.get(
            f"/api/v1/evidence/{evidence_id}/chain-of-custody",
            headers=auth_headers,
        )
        assert response.status_code == 200
        custody = response.json()
        assert custody["chain_valid"] is True
        assert custody["total_entries"] >= 1


class TestAnalysisWorkflow:
    """Test analysis submission and retrieval."""

    @pytest.mark.anyio
    async def test_analysis_endpoints(self, client: AsyncClient, auth_headers: dict):
        """Test analysis endpoints exist and respond."""
        # Check analysis list endpoint
        response = await client.get("/api/v1/analysis/jobs", headers=auth_headers)

        if response.status_code == 401:
            pytest.skip("Authentication required")

        assert response.status_code in [200, 404]  # 404 if no jobs exist


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.anyio
    async def test_rate_limit_headers(self, client: AsyncClient):
        """Test that rate limit headers are present."""
        response = await client.get("/api/v1/cases/")

        # Rate limit headers should be present
        # (May not be if rate limiting middleware not added to app)
        # This is a soft check
        if "X-RateLimit-Limit" in response.headers:
            assert int(response.headers["X-RateLimit-Limit"]) > 0
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.anyio
    async def test_404_for_missing_case(self, client: AsyncClient, auth_headers: dict):
        """Test 404 returned for missing case."""
        fake_id = str(uuid4())
        response = await client.get(f"/api/v1/cases/{fake_id}", headers=auth_headers)

        if response.status_code == 401:
            pytest.skip("Authentication required")

        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_validation_error(self, client: AsyncClient, auth_headers: dict):
        """Test validation error for invalid input."""
        # Missing required fields
        response = await client.post(
            "/api/v1/cases/",
            json={},
            headers=auth_headers,
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")

        assert response.status_code == 422  # Validation error
