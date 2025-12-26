"""Evidence Suite - API Middleware
Request logging, metrics, compression, timeout handling, and security.
"""

import asyncio
import gzip
import time
import uuid
from collections.abc import Callable
from io import BytesIO

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from core.logging import get_logger


# Configuration
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
COMPRESSION_MIN_SIZE = 1000  # Minimum response size for compression
COMPRESSION_LEVEL = 6  # gzip compression level (1-9)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout limits.
    Prevents long-running requests from consuming resources.
    """

    def __init__(self, app, timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Allow longer timeout for file uploads
        timeout = self.timeout_seconds
        if request.url.path.endswith("/upload"):
            timeout = 120  # 2 minutes for uploads

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            logger = get_logger()
            logger.warning(
                f"Request timeout after {timeout}s",
                path=request.url.path,
                method=request.method,
            )
            return Response(
                content='{"detail": "Request timeout"}',
                status_code=504,
                media_type="application/json",
            )


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for gzip response compression.
    Compresses responses larger than threshold when client supports it.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return await call_next(request)

        response = await call_next(request)

        # Skip if already streaming or compressed
        if isinstance(response, StreamingResponse):
            return response
        if response.headers.get("content-encoding"):
            return response

        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        # Only compress if large enough
        if len(body) < COMPRESSION_MIN_SIZE:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        # Compress
        buffer = BytesIO()
        with gzip.GzipFile(mode="wb", fileobj=buffer, compresslevel=COMPRESSION_LEVEL) as gz:
            gz.write(body)
        compressed_body = buffer.getvalue()

        # Only use compressed if smaller
        if len(compressed_body) < len(body):
            headers = dict(response.headers)
            headers["content-encoding"] = "gzip"
            headers["content-length"] = str(len(compressed_body))
            return Response(
                content=compressed_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )

        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


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
            "Request started",
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
                f"Request failed: {e!s}",
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
