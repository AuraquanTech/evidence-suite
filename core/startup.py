"""Evidence Suite - Startup Validation
Validates critical configuration before accepting traffic.
"""

import os
import sys
from typing import Any

from loguru import logger


class StartupError(Exception):
    """Critical startup failure."""

    pass


class StartupValidator:
    """Validates environment and dependencies on startup."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self, strict: bool = True) -> bool:
        """Run all validation checks.

        Args:
            strict: If True, fail on any error. If False, only fail on critical errors.

        Returns:
            True if validation passed
        """
        self._validate_environment()
        self._validate_jwt_secret()
        self._validate_database()
        self._validate_encryption_key()
        self._validate_storage()

        if self.errors:
            for error in self.errors:
                logger.error(f"Startup validation FAILED: {error}")
            if strict:
                raise StartupError(
                    f"Startup validation failed with {len(self.errors)} error(s). "
                    "Fix these issues before starting the application."
                )
            return False

        for warning in self.warnings:
            logger.warning(f"Startup validation WARNING: {warning}")

        logger.info("Startup validation passed")
        return True

    def _validate_environment(self) -> None:
        """Validate environment mode."""
        env = os.getenv("EVIDENCE_SUITE_ENV", "development")

        if env == "production":
            # In production, require explicit configuration
            if not os.getenv("DATABASE_URL"):
                self.errors.append("DATABASE_URL must be set in production mode")
            if not os.getenv("REDIS_HOST"):
                self.warnings.append("REDIS_HOST not set - caching will be disabled")
        elif env == "test":
            logger.info("Running in test mode - relaxed validation")
        else:
            logger.info(f"Running in {env} mode")

    def _validate_jwt_secret(self) -> None:
        """Validate JWT secret is properly configured."""
        secret = os.getenv("API_JWT_SECRET", "change-me-in-production")
        env = os.getenv("EVIDENCE_SUITE_ENV", "development")

        if secret == "change-me-in-production":  # nosec B105
            if env == "production":
                self.errors.append(
                    "API_JWT_SECRET must be changed from default in production. "
                    'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
            else:
                self.warnings.append(
                    "Using default JWT secret - change this before production deployment"
                )

        if len(secret) < 32:
            self.warnings.append(
                f"JWT secret is only {len(secret)} characters - recommend at least 32"
            )

    def _validate_database(self) -> None:
        """Validate database configuration."""
        env = os.getenv("EVIDENCE_SUITE_ENV", "development")
        db_url = os.getenv("DATABASE_URL", "")

        if env == "production":
            if not db_url:
                self.errors.append("DATABASE_URL is required in production")
            elif "sqlite" in db_url.lower():
                self.errors.append("SQLite is not supported in production mode. Use PostgreSQL.")
            elif "localhost" in db_url or "127.0.0.1" in db_url:
                self.warnings.append(
                    "Database URL points to localhost in production - is this intentional?"
                )

    def _validate_encryption_key(self) -> None:
        """Validate encryption key is configured."""
        key = os.getenv("EVIDENCE_ENCRYPTION_KEY")
        env = os.getenv("EVIDENCE_SUITE_ENV", "development")

        if not key:
            if env == "production":
                self.errors.append(
                    "EVIDENCE_ENCRYPTION_KEY must be set in production. "
                    'Generate with: python -c "import secrets; import base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"'
                )
            else:
                self.warnings.append(
                    "EVIDENCE_ENCRYPTION_KEY not set - evidence will be stored unencrypted"
                )

    def _validate_storage(self) -> None:
        """Validate storage configuration."""
        storage_path = os.getenv("EVIDENCE_STORAGE_PATH", "./evidence_store")

        if not os.path.exists(storage_path):
            try:
                os.makedirs(storage_path, exist_ok=True)
                logger.info(f"Created evidence storage directory: {storage_path}")
            except PermissionError:
                self.errors.append(f"Cannot create evidence storage directory: {storage_path}")
            except Exception as e:
                self.errors.append(f"Error creating evidence storage: {e}")

        # Check write permissions
        if os.path.exists(storage_path):
            test_file = os.path.join(storage_path, ".write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except PermissionError:
                self.errors.append(f"No write permission to evidence storage: {storage_path}")
            except Exception as e:
                self.warnings.append(f"Could not verify write permissions: {e}")


async def validate_database_connection(max_wait: float = 10.0) -> bool:
    """Test database connection on startup.

    Args:
        max_wait: Connection timeout in seconds

    Returns:
        True if connection successful
    """
    import asyncio

    from sqlalchemy import text

    from core.database.session import _get_async_engine_instance

    try:
        engine = _get_async_engine_instance()

        async def test_connection():
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True

        result = await asyncio.wait_for(test_connection(), timeout=max_wait)
        logger.info("Database connection verified")
        return result

    except TimeoutError:
        logger.error(f"Database connection timed out after {max_wait}s")
        return False
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def validate_redis_connection(max_wait: float = 5.0) -> bool:
    """Test Redis connection on startup.

    Args:
        max_wait: Connection timeout in seconds

    Returns:
        True if connection successful
    """
    import asyncio

    try:
        from core.cache import get_cache

        async def test_connection():
            cache = await get_cache()
            return cache.is_connected

        result = await asyncio.wait_for(test_connection(), timeout=max_wait)
        if result:
            logger.info("Redis connection verified")
        else:
            logger.warning("Redis not connected - running without cache")
        return result

    except TimeoutError:
        logger.warning(f"Redis connection timed out after {max_wait}s")
        return False
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        return False


def run_startup_validation(strict: bool | None = None) -> bool:
    """Run startup validation.

    Args:
        strict: Override strict mode. If None, uses production mode detection.

    Returns:
        True if validation passed
    """
    env = os.getenv("EVIDENCE_SUITE_ENV", "development")

    if strict is None:
        strict = env == "production"

    validator = StartupValidator()
    return validator.validate_all(strict=strict)
