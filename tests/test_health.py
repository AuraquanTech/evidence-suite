"""Evidence Suite - Health Endpoint Tests"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoints:
    """Test health and readiness endpoints."""

    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint returns API info."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "operational"

    async def test_live_endpoint(self, client: AsyncClient):
        """Test liveness probe returns alive."""
        response = await client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    async def test_health_endpoint(self, client: AsyncClient):
        """Test health endpoint returns detailed status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "cache" in data

    async def test_metrics_endpoint(self, client: AsyncClient):
        """Test metrics endpoint returns JSON metrics."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "counters" in data
        assert "gauges" in data
        assert "uptime_seconds" in data

    async def test_prometheus_metrics_endpoint(self, client: AsyncClient):
        """Test Prometheus metrics endpoint returns text format."""
        response = await client.get("/metrics/prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "evidence_suite_uptime_seconds" in response.text
