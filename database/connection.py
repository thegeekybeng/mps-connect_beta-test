"""Database connection and session management for MPS Connect."""

import os
import logging
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    # Fallback for local development
    DATABASE_URL = "postgresql://mpsconnect:password@localhost:5432/mpsconnect"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables: %s", e)
        raise


def drop_tables():
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error("Failed to drop database tables: %s", e)
        raise


def check_connection() -> bool:
    """Check if database connection is working."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        logger.error("Database connection check failed: %s", e)
        return False


def get_database_info() -> dict:
    """Get database information and statistics."""
    try:
        with engine.connect() as connection:
            # Get database version
            version_result = connection.execute(text("SELECT version()"))
            version = version_result.fetchone()[0]

            # Get database size
            size_result = connection.execute(
                text(
                    """
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """
                )
            )
            size = size_result.fetchone()[0]

            # Get table counts
            tables_result = connection.execute(
                text(
                    """
                SELECT 
                    schemaname,
                    relname AS tablename,
                    n_tup_ins AS inserts,
                    n_tup_upd AS updates,
                    n_tup_del AS deletes
                FROM pg_stat_user_tables
                ORDER BY relname
            """
                )
            )
            tables = [dict(row._mapping) for row in tables_result]  # noqa: SLF001

            return {
                "version": version,
                "size": size,
                "tables": tables,
                "connection_status": "connected",
            }
    except Exception as e:
        logger.error("Failed to get database info: %s", e)
        return {
            "version": "unknown",
            "size": "unknown",
            "tables": [],
            "connection_status": "error",
            "error": str(e),
        }


def run_retention_cleanup():
    """Run data retention cleanup procedures."""
    try:
        with engine.connect() as connection:
            # Clean up old audit logs (2 years)
            audit_result = connection.execute(
                text(
                    """
                DELETE FROM audit_logs 
                WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '2 years'
            """
                )
            )
            audit_deleted = audit_result.rowcount

            # Clean up old access logs (1 year)
            access_result = connection.execute(
                text(
                    """
                DELETE FROM access_logs 
                WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 year'
            """
                )
            )
            access_deleted = access_result.rowcount

            # Clean up expired sessions
            session_result = connection.execute(
                text(
                    """
                DELETE FROM sessions 
                WHERE expires_at < CURRENT_TIMESTAMP
            """
                )
            )
            session_deleted = session_result.rowcount

            connection.commit()

            logger.info(
                "Retention cleanup completed: %d audit logs, %d access logs, %d sessions deleted",
                audit_deleted,
                access_deleted,
                session_deleted,
            )

            return {
                "audit_logs_deleted": audit_deleted,
                "access_logs_deleted": access_deleted,
                "sessions_deleted": session_deleted,
            }
    except Exception as e:
        logger.error("Retention cleanup failed: %s", e)
        raise
