#!/usr/bin/env python3
"""
Database migration runner
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Run the database migration"""
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "ai_trading"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "password"),
    }

    # Connect to postgres database first to create the target database if it doesn't exist
    postgres_config = config.copy()
    postgres_config["database"] = "postgres"

    try:
        # Connect to postgres server
        conn = psycopg2.connect(**postgres_config)
        conn.autocommit = True
        cursor = conn.cursor()

        # Create database if it doesn't exist
        cursor.execute(
            f"SELECT 1 FROM pg_database WHERE datname = '{config['database']}'"
        )
        if not cursor.fetchone():
            logger.info(f"Creating database: {config['database']}")
            cursor.execute(f"CREATE DATABASE {config['database']}")
        else:
            logger.info(f"Database {config['database']} already exists")

        cursor.close()
        conn.close()

        # Connect to the target database
        conn = psycopg2.connect(cursor_factory=RealDictCursor, **config)
        cursor = conn.cursor()

        # Read and execute migration file
        migration_path = os.path.join(
            os.path.dirname(__file__), "migrations", "001_create_tables.sql"
        )
        with open(migration_path, "r") as f:
            migration_sql = f.read()

        logger.info("Running migration...")
        cursor.execute(migration_sql)
        conn.commit()

        # Verify tables were created
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Created tables: {tables}")

        cursor.close()
        conn.close()

        logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    run_migration()
