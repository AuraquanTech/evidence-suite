"""
Evidence Suite - Redis Caching Layer
Caching for analysis results, embeddings, and computed values.
"""
import json
import hashlib
from typing import Optional, Any, Dict, Union
from datetime import timedelta
import pickle

from loguru import logger
import redis.asyncio as redis

from core.config import redis_settings


class CacheManager:
    """
    Redis-based caching for Evidence Suite.

    Features:
    - Analysis result caching
    - BERT embedding caching
    - Evidence hash deduplication
    - Configurable TTLs
    """

    def __init__(self, url: Optional[str] = None):
        self.url = url or redis_settings.url
        self._client: Optional[redis.Redis] = None
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

    async def get_analysis(self, evidence_id: str) -> Optional[Dict]:
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

    async def set_analysis(
        self,
        evidence_id: str,
        result: Dict,
        ttl: Optional[int] = None
    ) -> bool:
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

    async def get_embedding(self, text_hash: str) -> Optional[bytes]:
        """Get cached BERT embedding."""
        if not self._is_connected:
            return None

        try:
            key = f"embedding:{text_hash}"
            return await self._client.get(key)
        except Exception as e:
            logger.warning(f"Embedding cache get failed: {e}")
            return None

    async def set_embedding(
        self,
        text_hash: str,
        embedding: bytes,
        ttl: Optional[int] = None
    ) -> bool:
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

    async def check_evidence_exists(self, file_hash: str) -> Optional[str]:
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
        ttl: int = 86400 * 30  # 30 days
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

    async def enqueue_job(self, job_id: str, job_data: Dict) -> bool:
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

    async def dequeue_job(self) -> Optional[Dict]:
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

    async def get_job_status(self, job_id: str) -> Optional[str]:
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

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> bool:
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

    async def get_stats(self) -> Dict[str, int]:
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
_cache_manager: Optional[CacheManager] = None


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
