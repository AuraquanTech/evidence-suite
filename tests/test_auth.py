"""Evidence Suite - Authentication Tests"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestAuthentication:
    """Test authentication endpoints and flows."""

    async def test_login_success(self, client: AsyncClient, test_user):
        """Test successful login returns tokens."""
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "testpassword123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    async def test_login_invalid_credentials(self, client: AsyncClient, test_user):
        """Test login with invalid credentials fails."""
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with nonexistent user fails."""
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent@example.com", "password": "password"},
        )
        assert response.status_code == 401

    async def test_me_endpoint_authenticated(self, client: AsyncClient, auth_headers: dict):
        """Test /me endpoint returns user info when authenticated."""
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["role"] == "analyst"

    async def test_me_endpoint_unauthenticated(self, client: AsyncClient):
        """Test /me endpoint requires authentication."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401

    async def test_logout(self, client: AsyncClient, auth_headers: dict):
        """Test logout blacklists token."""
        # First verify token works
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200

        # Logout
        response = await client.post("/api/v1/auth/logout", headers=auth_headers)
        assert response.status_code == 200

        # Token should now be blacklisted
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 401


@pytest.mark.asyncio
class TestAuthorization:
    """Test role-based access control."""

    async def test_analyst_cannot_delete_case(
        self, client: AsyncClient, auth_headers: dict, test_case
    ):
        """Test analyst role cannot delete cases."""
        response = await client.delete(f"/api/v1/cases/{test_case.id}", headers=auth_headers)
        assert response.status_code == 403

    async def test_admin_can_delete_case(
        self, client: AsyncClient, admin_auth_headers: dict, test_case
    ):
        """Test admin role can delete cases."""
        response = await client.delete(f"/api/v1/cases/{test_case.id}", headers=admin_auth_headers)
        assert response.status_code == 204
