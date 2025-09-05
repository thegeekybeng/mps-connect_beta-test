"""Alembic environment configuration for MPS Connect database migrations."""

# pylint: disable=all
# type: ignore
# flake8: noqa
# mypy: ignore-errors

import os
import sys
from logging.config import fileConfig

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with type ignore for missing stubs
from sqlalchemy import engine_from_config, pool  # type: ignore
from alembic import context  # type: ignore

# Import database models with try/except to handle missing modules
try:
    from database.models import Base  # type: ignore
    from database.connection import DATABASE_URL  # type: ignore
except ImportError:
    # Fallback for when modules are not available
    Base = None  # type: ignore
    DATABASE_URL = "postgresql://user:pass@localhost/db"  # type: ignore

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config  # type: ignore

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:  # type: ignore
    fileConfig(config.config_file_name)  # type: ignore

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata if Base else None  # type: ignore

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """Get database URL from environment or config."""
    return os.environ.get("DATABASE_URL", DATABASE_URL)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(  # type: ignore
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():  # type: ignore
        context.run_migrations()  # type: ignore


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section) or {}  # type: ignore
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:  # type: ignore
        context.configure(connection=connection, target_metadata=target_metadata)  # type: ignore

        with context.begin_transaction():  # type: ignore
            context.run_migrations()  # type: ignore


if context.is_offline_mode():  # type: ignore
    run_migrations_offline()
else:
    run_migrations_online()
