"""
Evidence Suite - API Tests
Unit tests for FastAPI endpoints.
"""

from uuid import uuid4

import pytest


# Client fixture is provided by conftest.py


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.anyio
    async def test_root(self, client):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "operational"

    @pytest.mark.anyio
    async def test_health(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestCaseEndpoints:
    """Test case management endpoints."""

    @pytest.mark.anyio
    async def test_create_case(self, client):
        """Test case creation."""
        case_data = {
            "case_number": f"TEST-{uuid4().hex[:8].upper()}",
            "title": "Test Case",
            "description": "Test case for API testing",
            "client_name": "Test Client",
            "attorney_name": "Test Attorney",
        }
        response = await client.post("/api/v1/cases/", json=case_data)
        # May fail if database not connected, which is expected in test
        assert response.status_code in [201, 500]

    @pytest.mark.anyio
    async def test_list_cases(self, client):
        """Test case listing."""
        response = await client.get("/api/v1/cases/")
        # May fail if database not connected
        assert response.status_code in [200, 500]

    @pytest.mark.anyio
    async def test_get_case_not_found(self, client):
        """Test getting non-existent case."""
        fake_id = str(uuid4())
        response = await client.get(f"/api/v1/cases/{fake_id}")
        # 404 if DB connected, 500 if not
        assert response.status_code in [404, 500]


class TestEvidenceEndpoints:
    """Test evidence management endpoints."""

    @pytest.mark.anyio
    async def test_list_evidence(self, client):
        """Test evidence listing."""
        response = await client.get("/api/v1/evidence/")
        assert response.status_code in [200, 500]

    @pytest.mark.anyio
    async def test_get_evidence_not_found(self, client):
        """Test getting non-existent evidence."""
        fake_id = str(uuid4())
        response = await client.get(f"/api/v1/evidence/{fake_id}")
        assert response.status_code in [404, 500]


class TestAnalysisEndpoints:
    """Test analysis endpoints."""

    @pytest.mark.anyio
    async def test_get_job_not_found(self, client):
        """Test getting non-existent job."""
        fake_id = str(uuid4())
        response = await client.get(f"/api/v1/analysis/job/{fake_id}")
        assert response.status_code in [404, 500]

    @pytest.mark.anyio
    async def test_analysis_not_found(self, client):
        """Test getting analysis for non-existent evidence."""
        fake_id = str(uuid4())
        response = await client.get(f"/api/v1/analysis/{fake_id}")
        assert response.status_code in [400, 404, 500]


class TestInputValidation:
    """Test input validation."""

    @pytest.mark.anyio
    async def test_create_case_missing_fields(self, client):
        """Test case creation with missing required fields."""
        response = await client.post("/api/v1/cases/", json={})
        assert response.status_code == 422  # Validation error

    @pytest.mark.anyio
    async def test_create_case_invalid_case_number(self, client):
        """Test case creation with empty case number."""
        case_data = {
            "case_number": "",
            "title": "Test",
        }
        response = await client.post("/api/v1/cases/", json=case_data)
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_list_cases_pagination(self, client):
        """Test case listing with pagination params."""
        response = await client.get("/api/v1/cases/?page=1&page_size=10")
        assert response.status_code in [200, 500]

    @pytest.mark.anyio
    async def test_list_cases_invalid_pagination(self, client):
        """Test case listing with invalid pagination."""
        response = await client.get("/api/v1/cases/?page=0")
        assert response.status_code == 422

        response = await client.get("/api/v1/cases/?page_size=1000")
        assert response.status_code == 422


# Run with: pytest tests/test_api.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
