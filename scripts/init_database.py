#!/usr/bin/env python3
"""Database initialization script for MPS Connect."""

# pylint: disable=import-error
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import create_tables, check_connection, get_database_info  # type: ignore  # pylint: disable=import-error

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize the database with tables and default data."""
    try:
        # Check database connection
        if not check_connection():
            logger.error("Database connection failed")
            return False

        logger.info("Database connection successful")

        # Create tables
        logger.info("Creating database tables...")
        create_tables()

        # Get database info
        db_info = get_database_info()
        logger.info("Database info: %s", db_info)

        logger.info("Database initialization completed successfully")
        return True

    except (ConnectionError, ImportError, RuntimeError, OSError) as e:
        logger.error("Database initialization failed: %s", e)
        return False
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unexpected error during database initialization: %s", e)
        return False


def main():
    """Main function."""
    logger.info("Starting MPS Connect database initialization...")

    # Check if DATABASE_URL is set
    if not os.environ.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set, using default local configuration")
        os.environ["DATABASE_URL"] = (
            "postgresql://mpsconnect:password@localhost:5432/mpsconnect"
        )

    success = init_database()

    if success:
        logger.info("Database initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("Database initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
