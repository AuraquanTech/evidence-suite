"""Evidence Suite - Redis Caching Layer
Caching for analysis results, embeddings, and computed values.
"""

import hashlib
import json
import pickle
from typing import Any

import redis.asyncio as redis
from loguru import logger

from core.config import redis_settings


class CacheManager:
    """Redis-based caching for Evidence Suite.

    Features:
    - Analysis result caching
    - BERT embedding caching
    - Evidence hash deduplication
    - Configurable TTLs
    """

    def __init__(self, url: str | None = None):
        self.url = url or redis_settings.url
        self._client: redis.Redis | None = None
        self._is_connected = False

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=False,  # We need bytes for pickle
            )
            await self._client.ping()
            self._is_connected = True
            logger.info(f"Connected to Redis at {redis_settings.host}:{redis_settings.port}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self._is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._is_connected = False
            logger.info("Disconnected from Redis")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ----- Analysis Result Caching -----

    async def get_analysis(self, evidence_id: str) -> dict | None:
        """Get cached analysis result."""
        if not self._is_connected:
            return None

        try:
            key = f"analysis:{evidence_id}"
            data = await self._client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    async def set_analysis(self, evidence_id: str, result: dict, ttl: int | None = None) -> bool:
        """Cache analysis result."""
        if not self._is_connected:
            return False

        try:
            key = f"analysis:{evidence_id}"
            ttl = ttl or redis_settings.analysis_cache_ttl
            data = pickle.dumps(result)
            await self._client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    async def invalidate_analysis(self, evidence_id: str) -> bool:
        """Invalidate cached analysis."""
        if not self._is_connected:
            return False

        try:
            key = f"analysis:{evidence_id}"
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache invalidate failed: {e}")
            return False

    # ----- Embedding Caching -----

    async def get_embedding(self, text_hash: str) -> bytes | None:
        """Get cached BERT embedding."""
        if not self._is_connected:
            return None

        try:
            key = f"embedding:{text_hash}"
            return await self._client.get(key)
        except Exception as e:
            logger.warning(f"Embedding cache get failed: {e}")
            return None

    async def set_embedding(self, text_hash: str, embedding: bytes, ttl: int | None = None) -> bool:
        """Cache BERT embedding."""
        if not self._is_connected:
            return False

        try:
            key = f"embedding:{text_hash}"
            ttl = ttl or redis_settings.embedding_cache_ttl
            await self._client.setex(key, ttl, embedding)
            return True
        except Exception as e:
            logger.warning(f"Embedding cache set failed: {e}")
            return False

    @staticmethod
    def hash_text(text: str) -> str:
        """Create hash of text for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    # ----- Evidence Deduplication -----

    async def check_evidence_exists(self, file_hash: str) -> str | None:
        """Check if evidence with hash already exists, return evidence ID if so."""
        if not self._is_connected:
            return None

        try:
            key = f"evidence_hash:{file_hash}"
            evidence_id = await self._client.get(key)
            return evidence_id.decode() if evidence_id else None
        except Exception as e:
            logger.warning(f"Evidence hash check failed: {e}")
            return None

    async def register_evidence_hash(
        self,
        file_hash: str,
        evidence_id: str,
        ttl: int = 86400 * 30,  # 30 days
    ) -> bool:
        """Register evidence hash for deduplication."""
        if not self._is_connected:
            return False

        try:
            key = f"evidence_hash:{file_hash}"
            await self._client.setex(key, ttl, evidence_id.encode())
            return True
        except Exception as e:
            logger.warning(f"Evidence hash registration failed: {e}")
            return False

    # ----- Job Queue -----

    async def enqueue_job(self, job_id: str, job_data: dict) -> bool:
        """Add job to processing queue."""
        if not self._is_connected:
            return False

        try:
            data = json.dumps(job_data)
            await self._client.lpush("job_queue", data)
            await self._client.hset("job_status", job_id, "queued")
            return True
        except Exception as e:
            logger.warning(f"Job enqueue failed: {e}")
            return False

    async def dequeue_job(self) -> dict | None:
        """Get next job from queue."""
        if not self._is_connected:
            return None

        try:
            data = await self._client.rpop("job_queue")
            if data:
                return json.loads(data.decode())
        except Exception as e:
            logger.warning(f"Job dequeue failed: {e}")
        return None

    async def update_job_status(self, job_id: str, status: str) -> bool:
        """Update job status."""
        if not self._is_connected:
            return False

        try:
            await self._client.hset("job_status", job_id, status)
            return True
        except Exception as e:
            logger.warning(f"Job status update failed: {e}")
            return False

    async def get_job_status(self, job_id: str) -> str | None:
        """Get job status."""
        if not self._is_connected:
            return None

        try:
            status = await self._client.hget("job_status", job_id)
            return status.decode() if status else None
        except Exception as e:
            logger.warning(f"Job status get failed: {e}")
            return None

    # ----- Rate Limiting -----

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if rate limit is exceeded."""
        if not self._is_connected:
            return True  # Allow if cache unavailable

        try:
            rate_key = f"ratelimit:{key}"
            current = await self._client.incr(rate_key)

            if current == 1:
                await self._client.expire(rate_key, window_seconds)

            return current <= limit
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True

    # ----- Query Result Caching -----

    async def get_query_result(self, query_key: str) -> Any | None:
        """Get cached query result."""
        if not self._is_connected:
            return None

        try:
            key = f"query:{query_key}"
            data = await self._client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Query cache get failed: {e}")
        return None

    async def set_query_result(
        self,
        query_key: str,
        result: Any,
        ttl: int = 300,  # 5 minutes default
    ) -> bool:
        """Cache query result."""
        if not self._is_connected:
            return False

        try:
            key = f"query:{query_key}"
            data = pickle.dumps(result)
            await self._client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Query cache set failed: {e}")
            return False

    async def invalidate_query_cache(self, pattern: str = "*") -> int:
        """Invalidate query cache entries matching pattern."""
        if not self._is_connected:
            return 0

        try:
            keys = await self._client.keys(f"query:{pattern}")
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Query cache invalidation failed: {e}")
            return 0

    @staticmethod
    def make_query_key(*args, **kwargs) -> str:
        """Create cache key from query parameters."""
        key_parts = [str(a) for a in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    # ----- Batch Operations -----

    async def mget(self, keys: list[str], prefix: str = "") -> dict[str, Any]:
        """Get multiple values at once."""
        if not self._is_connected or not keys:
            return {}

        try:
            full_keys = [f"{prefix}{k}" if prefix else k for k in keys]
            values = await self._client.mget(full_keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = pickle.loads(value)
                    except Exception:
                        result[key] = value.decode() if isinstance(value, bytes) else value
            return result
        except Exception as e:
            logger.warning(f"Batch get failed: {e}")
            return {}

    async def mset(self, items: dict[str, Any], prefix: str = "", ttl: int | None = None) -> bool:
        """Set multiple values at once."""
        if not self._is_connected or not items:
            return False

        try:
            # Use pipeline for atomic operation
            pipe = self._client.pipeline()

            for key, value in items.items():
                full_key = f"{prefix}{key}" if prefix else key
                data = pickle.dumps(value)
                if ttl:
                    pipe.setex(full_key, ttl, data)
                else:
                    pipe.set(full_key, data)

            await pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Batch set failed: {e}")
            return False

    async def mdelete(self, keys: list[str], prefix: str = "") -> int:
        """Delete multiple keys at once."""
        if not self._is_connected or not keys:
            return 0

        try:
            full_keys = [f"{prefix}{k}" if prefix else k for k in keys]
            return await self._client.delete(*full_keys)
        except Exception as e:
            logger.warning(f"Batch delete failed: {e}")
            return 0

    # ----- Cache Info -----

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics and info."""
        if not self._is_connected:
            return {"status": "disconnected"}

        try:
            info = await self._client.info("memory")
            keyspace = await self._client.info("keyspace")

            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
                "connected_clients": (await self._client.info("clients")).get(
                    "connected_clients", 0
                ),
                "keyspace": keyspace,
            }
        except Exception as e:
            logger.warning(f"Cache info failed: {e}")
            return {"status": "error", "error": str(e)}

    # ----- Statistics -----

    async def increment_stat(self, stat_name: str, value: int = 1) -> bool:
        """Increment a statistics counter."""
        if not self._is_connected:
            return False

        try:
            key = f"stats:{stat_name}"
            await self._client.incrby(key, value)
            return True
        except Exception as e:
            logger.warning(f"Stat increment failed: {e}")
            return False

    async def get_stats(self) -> dict[str, int]:
        """Get all statistics."""
        if not self._is_connected:
            return {}

        try:
            keys = await self._client.keys("stats:*")
            stats = {}
            for key in keys:
                stat_name = key.decode().replace("stats:", "")
                value = await self._client.get(key)
                stats[stat_name] = int(value) if value else 0
            return stats
        except Exception as e:
            logger.warning(f"Stats get failed: {e}")
            return {}


# Singleton instance
_cache_manager: CacheManager | None = None


async def get_cache() -> CacheManager:
    """Get singleton cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.connect()

    return _cache_manager


async def close_cache():
    """Close cache connection."""
    global _cache_manager

    if _cache_manager:
        await _cache_manager.disconnect()
        _cache_manager = None
