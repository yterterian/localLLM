"""
System Integration Tests for Optimization Layer

This test suite verifies the integration of the optimization system,
including ComprehensiveOptimizationManager and FastAPI endpoints.
"""

import asyncio
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient
import httpx

from optimization.optimization_integration import (
    ComprehensiveOptimizationManager,
    add_optimization_endpoints,
)

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop

@pytest_asyncio.fixture
async def test_app():
    app = FastAPI()
    optimization_manager = ComprehensiveOptimizationManager()
    add_optimization_endpoints(app, optimization_manager)
    # Start background tasks (simulate)
    await optimization_manager.start_optimization_systems()
    yield app, optimization_manager
    await optimization_manager.stop_optimization_systems()

@pytest.mark.asyncio
async def test_optimization_stats_endpoint(test_app):
    app, _ = test_app
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/optimization/stats")
        assert response.status_code == 200
        data = response.json()
        assert "model_preloading" in data
        assert "hardware_optimization" in data
        assert "context_optimization" in data

@pytest.mark.asyncio
async def test_performance_report_endpoint(test_app):
    app, _ = test_app
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/optimization/performance-report")
        assert response.status_code == 200
        data = response.json()
        assert "performance_baseline" in data
        assert "performance_improvements" in data

@pytest.mark.asyncio
async def test_benchmark_endpoint(test_app):
    app, _ = test_app
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/optimization/benchmark")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Benchmark started"
        assert "timestamp" in data
