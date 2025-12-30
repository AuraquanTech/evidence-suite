"""Evidence Suite - Cases API Tests"""

from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestCasesAPI:
    """Test cases CRUD endpoints."""

    async def test_list_cases_authenticated(self, client: AsyncClient, auth_headers: dict):
        """Test listing cases requires authentication."""
        response = await client.get("/api/v1/cases", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_list_cases_unauthenticated(self, client: AsyncClient):
        """Test listing cases without auth fails."""
        response = await client.get("/api/v1/cases")
        assert response.status_code == 401

    async def test_create_case(self, client: AsyncClient, auth_headers: dict):
        """Test creating a new case."""
        case_data = {
            "case_number": "CASE-TEST-001",
            "title": "Test Case Creation",
            "description": "Testing case creation endpoint",
            "client_name": "Test Client",
        }
        response = await client.post("/api/v1/cases", json=case_data, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["case_number"] == "CASE-TEST-001"
        assert data["title"] == "Test Case Creation"
        assert "id" in data

    async def test_get_case(self, client: AsyncClient, auth_headers: dict, test_case):
        """Test getting a specific case."""
        response = await client.get(f"/api/v1/cases/{test_case.id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_case.id)
        assert data["case_number"] == test_case.case_number

    async def test_get_nonexistent_case(self, client: AsyncClient, auth_headers: dict):
        """Test getting a nonexistent case returns 404."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/cases/{fake_id}", headers=auth_headers)
        assert response.status_code == 404

    async def test_update_case(self, client: AsyncClient, auth_headers: dict, test_case):
        """Test updating a case."""
        update_data = {"title": "Updated Title"}
        response = await client.patch(
            f"/api/v1/cases/{test_case.id}",
            json=update_data,
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"

    async def test_case_evidence_list(
        self, client: AsyncClient, auth_headers: dict, test_case, test_evidence
    ):
        """Test listing evidence for a case."""
        response = await client.get(f"/api/v1/cases/{test_case.id}/evidence", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1


@pytest.mark.asyncio
class TestCasesValidation:
    """Test case input validation."""

    async def test_create_case_missing_required(self, client: AsyncClient, auth_headers: dict):
        """Test creating case without required fields fails."""
        case_data = {"description": "Missing required fields"}
        response = await client.post("/api/v1/cases", json=case_data, headers=auth_headers)
        assert response.status_code == 422

    async def test_create_case_invalid_status(self, client: AsyncClient, auth_headers: dict):
        """Test creating case with invalid status fails."""
        case_data = {
            "case_number": "CASE-TEST-002",
            "title": "Invalid Status Test",
            "status": "invalid_status",
        }
        response = await client.post("/api/v1/cases", json=case_data, headers=auth_headers)
        assert response.status_code == 422
