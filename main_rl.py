# main_rl.py
import os
import logging
from flask import Flask, jsonify
from threading import Thread
import time

from services.coordinator import Coordinator
from agents.agent_factory import AgentFactory
from database.connection import DatabaseConnection
from dashboard.app import create_app as create_dashboard_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ai_trading.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Global variables
coordinator = None


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)

    # Health check endpoint
    @app.route("/health")
    def health_check():
        return jsonify(
            {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}
        )

    # Register dashboard blueprints
    from dashboard.app import create_app as create_dashboard_app

    dashboard_app = create_dashboard_app()

    # Copy routes from dashboard app
    for rule in dashboard_app.url_map.iter_rules():
        if rule.endpoint != "static":
            app.add_url_rule(
                rule.rule,
                rule.endpoint,
                dashboard_app.view_functions[rule.endpoint],
                methods=rule.methods,
            )

    return app


def initialize_system():
    """Initialize the AI trading system"""
    global coordinator

    try:
        logger.info("Initializing AI Trading System...")

        # Initialize database
        db = DatabaseConnection()
        if not db.connect():
            raise Exception("Failed to connect to database")

        # Create tables if they don't exist
        _ensure_database_tables(db)

        # Initialize coordinator
        coordinator = Coordinator()

        # Add default agents
        _add_default_agents(coordinator)

        logger.info(f"System initialized with {len(coordinator.agents)} agents")
        return coordinator

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


def _ensure_database_tables(db):
    """Ensure database tables exist"""
    try:
        # Check if tables exist
        tables = db.get_table_names()
        required_tables = [
            "agents",
            "trades",
            "market_data",
            "experiences",
            "models",
            "performance_summary",
        ]

        missing_tables = [table for table in required_tables if table not in tables]

        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            logger.info("Please run database migrations first")
            # In production, you might want to auto-run migrations here
        else:
            logger.info("All required tables exist")

    except Exception as e:
        logger.error(f"Error checking database tables: {e}")


def _add_default_agents(coordinator):
    """Add default agents to coordinator"""
    strategies = ["conservative", "aggressive", "balanced", "trend", "mean_reversion"]

    for i, strategy in enumerate(strategies):
        agent = AgentFactory.create_agent(strategy, f"agent_{i}", 500000000)
        coordinator.add_agent(agent)
        logger.info(f"Added {strategy} agent: {agent.name}")


def start_background_tasks():
    """Start background tasks"""

    def monitor_system():
        """Monitor system health and performance"""
        while True:
            try:
                if coordinator:
                    status = coordinator.get_system_status()

                    # Log any issues
                    if status["total_agents"] == 0:
                        logger.warning("No agents in system")

                    if status["total_pnl"] < -100000:  # More than $100k loss
                        logger.warning(
                            f"High loss detected: ${status['total_pnl']:,.2f}"
                        )

                    # Check training schedule
                    from datetime import datetime, timedelta

                    if coordinator.rl_engine:
                        stats = coordinator.rl_engine.get_training_stats()
                        if stats.get("buffer_size", 0) > 1000:
                            logger.info(f"Training progress: {stats}")

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                time.sleep(60)

    # Start monitor thread
    monitor_thread = Thread(target=monitor_system, daemon=True)
    monitor_thread.start()
    logger.info("Background monitoring started")


def main():
    """Main application entry point"""
    try:
        logger.info("Starting AI Trading System...")

        # Initialize system
        coordinator = initialize_system()

        # Start background tasks
        start_background_tasks()

        # Create Flask app
        app = create_app()

        # Start web server
        port = int(os.getenv("PORT", 5000))
        host = os.getenv("HOST", "0.0.0.0")

        logger.info(f"Starting web server on {host}:{port}")
        app.run(host=host, port=port, debug=False)

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        if coordinator:
            coordinator.stop_simulation()
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
