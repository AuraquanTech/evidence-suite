"""Evidence Suite - FastAPI Application
Forensic behavioral intelligence REST API.
Production-ready with startup validation and graceful shutdown.
"""

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.middleware import (
    CompressionMiddleware,
    RequestLoggingMiddleware,
    RequestTimeoutMiddleware,
    SecurityHeadersMiddleware,
)
from api.rate_limit import RateLimitMiddleware
from core.config import api_settings, db_settings
from core.database.session import init_db_async, wait_for_database
from core.logging import configure_logging, get_logger


# Shutdown event for graceful termination
shutdown_event = asyncio.Event()


def handle_shutdown_signal(signum, frame):
    """Handle shutdown signals for graceful termination."""
    logger = get_logger()
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name}, initiating graceful shutdown...")
    shutdown_event.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with startup validation."""
    # Initialize logging first
    logger = configure_logging(
        log_dir="./logs",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        json_format=True,
    )

    logger.info("Starting Evidence Suite API...")
    env = os.getenv("EVIDENCE_SUITE_ENV", "development")
    logger.info(f"Environment: {env}")

    # Run startup validation
    try:
        from core.startup import run_startup_validation

        run_startup_validation(strict=(env == "production"))
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        if env == "production":
            raise

    # Wait for database to be available
    db_available = await wait_for_database(
        timeout=float(os.getenv("DB_STARTUP_TIMEOUT", "60")),
        interval=2.0,
    )

    if not db_available:
        if env == "production":
            raise RuntimeError("Database not available - cannot start in production mode")
        logger.warning("Database not available - some features may not work")

    # Initialize database tables
    try:
        await init_db_async()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
        if env == "production":
            raise

    # Initialize Redis cache
    try:
        from core.cache import get_cache

        cache = await get_cache()
        if cache.is_connected:
            logger.info("Redis cache connected")
        else:
            logger.info("Redis cache not available - running without cache")
    except Exception as e:
        logger.warning(f"Redis cache warning: {e}")

    # Register signal handlers for graceful shutdown
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown_signal(s, None))
    else:
        # Windows doesn't support add_signal_handler
        signal.signal(signal.SIGTERM, handle_shutdown_signal)
        signal.signal(signal.SIGINT, handle_shutdown_signal)

    # Mark app as ready
    app.state.ready = True
    logger.info("Evidence Suite API started successfully and ready to accept requests")

    yield

    # Graceful shutdown
    logger.info("Shutting down Evidence Suite API...")
    app.state.ready = False

    # Allow in-flight requests to complete (grace period)
    grace_period = float(os.getenv("SHUTDOWN_GRACE_PERIOD", "10"))
    logger.info(f"Waiting {grace_period}s for in-flight requests to complete...")
    await asyncio.sleep(grace_period)

    # Close worker pool if active
    try:
        from worker.client import close_worker_pool

        await close_worker_pool()
        logger.info("Worker pool closed")
    except Exception:
        pass

    # Close cache connection
    try:
        from core.cache import close_cache

        await close_cache()
        logger.info("Cache connection closed")
    except Exception:
        pass

    logger.info("Evidence Suite API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=api_settings.title,
    version=api_settings.version,
    description=api_settings.description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware (order matters - executed in reverse order of registration)
# 1. Security headers (outermost - always applied)
app.add_middleware(SecurityHeadersMiddleware)
# 2. Rate limiting (prevent abuse)
app.add_middleware(RateLimitMiddleware)
# 3. Compression (compress responses before sending)
app.add_middleware(CompressionMiddleware)
# 4. Timeout (enforce request timeouts)
app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=30)
# 5. Logging (innermost - logs actual request/response)
app.add_middleware(RequestLoggingMiddleware)


# Import and include routers
from api.auth import router as auth_router
from api.routes import (
    analysis_router,
    cases_router,
    evidence_router,
    exports_router,
    reports_router,
)
from api.websocket import router as websocket_router


app.include_router(cases_router, prefix="/api/v1")
app.include_router(evidence_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")
app.include_router(exports_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": api_settings.title,
        "version": api_settings.version,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with database status."""
    from core.database.monitoring import get_monitor

    health = {
        "status": "healthy",
        "database": {"host": db_settings.host, "status": "unknown"},
        "cache": {"status": "unknown"},
    }

    # Check database
    try:
        monitor = get_monitor()
        db_health = monitor.get_health()
        health["database"]["status"] = db_health.get("status", "unknown")
        health["database"]["pool"] = db_health.get("pool", {})
    except Exception as e:
        health["database"]["status"] = "error"
        health["database"]["error"] = str(e)

    # Check cache
    try:
        from core.cache import get_cache

        cache = await get_cache()
        health["cache"]["status"] = "connected" if cache.is_connected else "disconnected"
    except Exception as e:
        health["cache"]["status"] = "error"
        health["cache"]["error"] = str(e)

    # Overall status
    if health["database"]["status"] == "error":
        health["status"] = "degraded"

    return health


@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness probe.

    Returns 200 only when the application is ready to serve traffic.
    """
    if not getattr(app.state, "ready", False):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": "Application is not ready"},
        )

    # Quick database check
    try:
        from core.database.session import test_connection

        if not await test_connection():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "message": "Database not connected"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": f"Database error: {e}"},
        )

    return {"status": "ready"}


@app.get("/live")
async def liveness_check():
    """Kubernetes-style liveness probe.

    Returns 200 if the application is alive (even if not fully ready).
    """
    return {"status": "alive"}


@app.get("/health/db")
async def database_health():
    """Detailed database health metrics."""
    from core.database.monitoring import get_monitor

    monitor = get_monitor()
    return {
        "health": monitor.get_health(),
        "query_stats": monitor.get_query_stats(),
        "slow_queries": monitor.get_slow_queries(limit=5),
        "recent_errors": monitor.get_errors(limit=5),
    }


@app.get("/metrics")
async def get_metrics_json():
    """Get application metrics as JSON."""
    from core.database.monitoring import get_monitor
    from core.metrics import collect_database_metrics, collect_system_metrics, get_metrics

    # Collect latest metrics
    await collect_system_metrics()
    await collect_database_metrics()

    metrics = get_metrics()
    return metrics.get_all()


@app.get("/metrics/prometheus")
async def get_metrics_prometheus():
    """Get application metrics in Prometheus text format."""
    from fastapi.responses import PlainTextResponse

    from core.metrics import collect_database_metrics, collect_system_metrics, get_metrics

    # Collect latest metrics
    await collect_system_metrics()
    await collect_database_metrics()

    metrics = get_metrics()
    return PlainTextResponse(
        content=metrics.export_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = get_logger()
    logger.error(
        f"Unhandled exception: {exc!s}",
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=api_settings.host,
        port=api_settings.port,
        reload=True,
    )
