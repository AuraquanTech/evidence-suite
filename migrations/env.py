"""
Evidence Suite - Alembic Migration Environment
Supports both sync (autogenerate) and async (upgrade/downgrade) modes.
"""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database.models import Base
from core.config import db_settings

# Alembic Config object
config = context.config

# Set the sync URL for autogenerate and async URL for migrations
config.set_main_option("sqlalchemy.url", db_settings.async_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online_sync() -> None:
    """Run migrations using synchronous engine (for autogenerate)."""
    connectable = create_engine(
        db_settings.url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Uses sync engine for autogenerate operations,
    async engine for upgrade/downgrade operations.
    """
    # Check if we're doing autogenerate (revision command)
    # The revision command sets this attribute
    is_autogenerate = getattr(context.config.cmd_opts, "autogenerate", False)

    if is_autogenerate:
        # Use sync engine for autogenerate to avoid async complexity
        run_migrations_online_sync()
    else:
        # Use async engine for actual migrations
        asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
