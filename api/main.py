"""
Evidence Suite - FastAPI Application
Forensic behavioral intelligence REST API.
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import api_settings, db_settings
from core.database.session import init_db_async
from core.logging import configure_logging, get_logger
from api.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Initialize logging
    logger = configure_logging(
        log_dir="./logs",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        json_format=True,
    )

    # Startup
    logger.info("Starting Evidence Suite API...")

    # Initialize database
    try:
        await init_db_async()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")

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

    logger.info("Evidence Suite API started successfully")
    yield

    # Shutdown
    logger.info("Shutting down Evidence Suite API...")

    # Close cache connection
    try:
        from core.cache import close_cache
        await close_cache()
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

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


# Import and include routers
from api.routes import cases_router, evidence_router, analysis_router
from api.auth import router as auth_router
from api.websocket import router as websocket_router

app.include_router(cases_router, prefix="/api/v1")
app.include_router(evidence_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
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
        "database": {
            "host": db_settings.host,
            "status": "unknown"
        },
        "cache": {
            "status": "unknown"
        }
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


@app.get("/health/db")
async def database_health():
    """Detailed database health metrics."""
    from core.database.monitoring import get_monitor

    monitor = get_monitor()
    return {
        "health": monitor.get_health(),
        "query_stats": monitor.get_query_stats(),
        "slow_queries": monitor.get_slow_queries(limit=5),
        "recent_errors": monitor.get_errors(limit=5)
    }


@app.get("/metrics")
async def get_metrics():
    """Get application metrics including database stats."""
    from core.database.monitoring import get_monitor

    logger = get_logger()
    log_metrics = logger.get_metrics()

    # Add database metrics
    try:
        monitor = get_monitor()
        log_metrics["database"] = {
            "pool": monitor.get_pool_stats(),
            "queries": monitor.get_query_stats()
        }
    except Exception:
        log_metrics["database"] = {"status": "unavailable"}

    return log_metrics


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = get_logger()
    logger.error(
        f"Unhandled exception: {str(exc)}",
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
