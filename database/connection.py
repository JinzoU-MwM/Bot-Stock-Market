import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "ai_trading"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password"),
        }

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                cursor_factory=RealDictCursor, **self.config
            )
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def execute(self, query, params=None):
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith("SELECT"):
                    result = cursor.fetchall()
                    return result
                else:
                    self.connection.commit()
                    return cursor.rowcount
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            if self.connection:
                self.connection.rollback()
            return None

    def get_table_names(self):
        result = self.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        return [row["table_name"] for row in result] if result else []

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
