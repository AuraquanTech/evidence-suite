"""
Evidence Suite - API Middleware
Request logging, metrics, and error handling middleware.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all HTTP requests with metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        logger = get_logger()

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Start timing
        start_time = time.perf_counter()

        # Log request start
        logger.debug(
            f"Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code}",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Record metrics
            logger.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                f"Request failed: {str(e)}",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )

            # Record error metrics
            logger.record_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=duration_ms,
            )

            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
