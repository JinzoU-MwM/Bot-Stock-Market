from flask import Flask, render_template, jsonify
import json
from datetime import datetime
import logging

from services.coordinator import Coordinator
from agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

# Global coordinator instance
coordinator = Coordinator()


def create_app():
    app = Flask(__name__)

    # Initialize coordinator with some agents
    _initialize_coordinator()

    @app.route("/")
    def index():
        """Main dashboard page"""
        return render_template("index.html")

    @app.route("/api/system_status")
    def system_status():
        """Get system status as JSON"""
        try:
            status = coordinator.get_system_status()
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/agents/<agent_id>")
    def agent_details(agent_id):
        """Get detailed agent information"""
        try:
            details = coordinator.get_agent_details(agent_id)
            if details:
                return jsonify(details)
            else:
                return jsonify({"error": "Agent not found"}), 404
        except Exception as e:
            logger.error(f"Error getting agent details: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/start_simulation", methods=["POST"])
    def start_simulation():
        """Start the trading simulation"""
        try:
            coordinator.start_simulation()
            return jsonify({"status": "success", "message": "Simulation started"})
        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stop_simulation", methods=["POST"])
    def stop_simulation():
        """Stop the trading simulation"""
        try:
            coordinator.stop_simulation()
            return jsonify({"status": "success", "message": "Simulation stopped"})
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/add_agent", methods=["POST"])
    def add_agent():
        """Add a new agent"""
        try:
            # For now, just add a balanced agent
            # In a real implementation, you'd get parameters from request
            agent_count = len(coordinator.agents)
            agent = AgentFactory.create_agent(
                "balanced", f"agent_{agent_count}", 500000000
            )
            coordinator.add_agent(agent)

            return jsonify(
                {
                    "status": "success",
                    "message": "Agent added",
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                }
            )
        except Exception as e:
            logger.error(f"Error adding agent: {e}")
            return jsonify({"error": str(e)}), 500

    return app


def _initialize_coordinator():
    """Initialize coordinator with default agents"""
    strategies = ["conservative", "aggressive", "balanced", "trend", "mean_reversion"]

    for i, strategy in enumerate(strategies):
        agent = AgentFactory.create_agent(strategy, f"agent_{i}", 500000000)
        coordinator.add_agent(agent)

    logger.info(f"Initialized coordinator with {len(coordinator.agents)} agents")


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
