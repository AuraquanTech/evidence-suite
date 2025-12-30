"""Evidence Suite - Rate Limiting
Redis-backed rate limiting middleware.
"""

import os
import time
from collections.abc import Callable

from fastapi import HTTPException, Request, Response, status
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


# Rate limit settings
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""

    def __init__(self, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


class RateLimiter:
    """Token bucket rate limiter with Redis backend.

    Falls back to in-memory limiting if Redis is unavailable.
    """

    def __init__(
        self,
        requests: int = RATE_LIMIT_REQUESTS,
        window: int = RATE_LIMIT_WINDOW,
    ):
        self.requests = requests
        self.window = window
        self._redis = None
        self._local_cache: dict[str, list[float]] = {}

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                from core.cache import get_cache

                cache = await get_cache()
                if cache.is_connected:
                    self._redis = cache._redis
            except Exception:
                pass
        return self._redis

    def _get_client_key(self, request: Request) -> str:
        """Get rate limit key for client."""
        # Try to get user ID from auth
        user_id = getattr(request.state, "user_id", None)

        if user_id:
            return f"rate_limit:user:{user_id}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"rate_limit:ip:{client_ip}"

    async def is_allowed(self, request: Request) -> tuple[bool, dict]:
        """Check if request is allowed under rate limit.

        Returns:
            (allowed, info) where info contains limit details
        """
        key = self._get_client_key(request)
        now = time.time()

        redis = await self._get_redis()

        if redis:
            return await self._check_redis(redis, key, now)
        return self._check_local(key, now)

    async def _check_redis(self, redis, key: str, now: float) -> tuple[bool, dict]:
        """Check rate limit using Redis."""
        window_start = now - self.window

        try:
            # Use Redis sorted set for sliding window
            pipe = redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Set expiry
            pipe.expire(key, self.window)

            results = await pipe.execute()
            current_count = results[1]

            remaining = max(0, self.requests - current_count - 1)
            reset_time = int(now + self.window)

            info = {
                "limit": self.requests,
                "remaining": remaining,
                "reset": reset_time,
                "window": self.window,
            }

            if current_count >= self.requests:
                return False, info

            return True, info

        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}")
            # Fall back to allowing request
            return True, {"limit": self.requests, "remaining": self.requests}

    def _check_local(self, key: str, now: float) -> tuple[bool, dict]:
        """Check rate limit using local memory (fallback)."""
        window_start = now - self.window

        # Get or create request history
        if key not in self._local_cache:
            self._local_cache[key] = []

        # Remove old entries
        self._local_cache[key] = [ts for ts in self._local_cache[key] if ts > window_start]

        current_count = len(self._local_cache[key])
        remaining = max(0, self.requests - current_count - 1)
        reset_time = int(now + self.window)

        info = {
            "limit": self.requests,
            "remaining": remaining,
            "reset": reset_time,
            "window": self.window,
        }

        if current_count >= self.requests:
            return False, info

        # Add current request
        self._local_cache[key].append(now)

        # Cleanup old keys periodically
        if len(self._local_cache) > 10000:
            self._cleanup_local_cache(now)

        return True, info

    def _cleanup_local_cache(self, now: float) -> None:
        """Remove stale entries from local cache."""
        window_start = now - self.window
        to_delete = []

        for key, timestamps in self._local_cache.items():
            if not timestamps or max(timestamps) < window_start:
                to_delete.append(key)

        for key in to_delete:
            del self._local_cache[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI.

    Adds rate limit headers to all responses:
    - X-RateLimit-Limit: Maximum requests per window
    - X-RateLimit-Remaining: Requests remaining
    - X-RateLimit-Reset: Unix timestamp when limit resets
    """

    def __init__(
        self,
        app,
        requests: int = RATE_LIMIT_REQUESTS,
        window: int = RATE_LIMIT_WINDOW,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.limiter = RateLimiter(requests, window)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/health/db",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Check rate limit
        allowed, info = await self.limiter.is_allowed(request)

        if not allowed:
            retry_after = info.get("reset", 60) - int(time.time())
            raise RateLimitExceeded(retry_after=max(1, retry_after))

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", RATE_LIMIT_REQUESTS))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))

        return response


# Endpoint-specific rate limiters
class EndpointRateLimiter:
    """Decorator for endpoint-specific rate limiting.

    Usage:
        @router.post("/analyze")
        @rate_limit(requests=10, window=60)
        async def analyze_evidence(...):
            ...
    """

    def __init__(self, requests: int = 10, window: int = 60):
        self.limiter = RateLimiter(requests, window)

    async def __call__(self, request: Request) -> None:
        """Check rate limit for endpoint."""
        allowed, info = await self.limiter.is_allowed(request)

        if not allowed:
            retry_after = info.get("reset", 60) - int(time.time())
            raise RateLimitExceeded(retry_after=max(1, retry_after))


def rate_limit(requests: int = 10, window: int = 60):
    """Create a rate limit dependency.

    Usage:
        @router.post("/heavy-endpoint")
        async def heavy_endpoint(
            request: Request,
            _: None = Depends(rate_limit(requests=5, window=60))
        ):
            ...
    """
    limiter = EndpointRateLimiter(requests, window)
    return limiter
