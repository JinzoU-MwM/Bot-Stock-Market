import threading
import time
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import numpy as np

from agents.base_agent import BaseAgent
from rl.rl_engine import RLEngine
from services.market_data import MarketDataService
from database.connection import DatabaseConnection
from rl.environment import TradingEnvironment

logger = logging.getLogger(__name__)


class Coordinator:
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.rl_engine = RLEngine()
        self.market_data_service = MarketDataService()
        self.database = DatabaseConnection()
        self.environments = {}  # agent_id -> TradingEnvironment
        self.is_running = False
        self.simulation_thread = None

        # Trading symbols for simulation
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]

        # Training schedule
        self.last_training_time = datetime.now()
        self.training_interval = timedelta(minutes=5)  # Train every 5 minutes
        self.min_experiences_for_training = 1000

        # Performance tracking
        self.performance_history = []

    def add_agent(self, agent: BaseAgent):
        """Add an agent to the simulation"""
        self.agents.append(agent)

        # Create trading environment for this agent
        environment = TradingEnvironment(agent, self.market_data_service, self.database)
        environment.symbols = self.symbols.copy()
        self.environments[agent.id] = environment

        logger.info(f"Added agent {agent.name} with strategy {agent.strategy_type}")

    def remove_agent(self, agent_id: str):
        """Remove an agent from the simulation"""
        self.agents = [agent for agent in self.agents if agent.id != agent_id]
        if agent_id in self.environments:
            del self.environments[agent_id]
        logger.info(f"Removed agent {agent_id}")

    def start_simulation(self):
        """Start the trading simulation"""
        if self.is_running:
            logger.warning("Simulation is already running")
            return

        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

        logger.info("Started trading simulation")

    def stop_simulation(self):
        """Stop the trading simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        logger.info("Stopped trading simulation")

    def _simulation_loop(self):
        """Main simulation loop"""
        step_count = 0

        while self.is_running:
            try:
                step_start_time = time.time()

                # Run one step for each agent
                step_results = self.run_simulation_step()

                # Record performance
                self.performance_history.append(
                    {
                        "timestamp": datetime.now(),
                        "step": step_count,
                        "results": step_results,
                    }
                )

                # Check if we should train the model
                if self._should_train():
                    self._train_model()

                # Calculate delay to maintain step frequency (e.g., 1 step per second)
                step_duration = time.time() - step_start_time
                target_duration = 1.0  # 1 second per step
                sleep_duration = max(0, target_duration - step_duration)

                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                step_count += 1

                # Log progress every 100 steps
                if step_count % 100 == 0:
                    self._log_progress(step_count, step_results)

            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(5)  # Wait before retrying

    def run_simulation_step(self):
        """Run one simulation step for all agents"""
        step_results = {}

        for agent in self.agents:
            if agent.status != "active":
                continue

            try:
                # Get agent environment
                environment = self.environments[agent.id]

                # Get current state
                current_state = environment._get_current_state()

                # Select action using RL model
                action, value = self.rl_engine.select_action(current_state)

                # Convert action back to dictionary format
                action_dict = self._vector_to_action_dict(action, agent)

                # Execute action in environment
                next_state, reward, done, info = environment.step(action)

                # Store experience
                self.rl_engine.add_experience(
                    agent.id,
                    self._state_vector_to_dict(current_state, agent),
                    action_dict,
                    reward,
                    self._state_vector_to_dict(next_state, agent),
                    done,
                )

                step_results[agent.id] = {
                    "action": action_dict,
                    "reward": reward,
                    "info": info,
                    "performance": agent.get_performance_metrics(),
                }

                # Reset environment if episode is done
                if done:
                    environment.reset()
                    logger.info(f"Agent {agent.name} episode completed")

            except Exception as e:
                logger.error(f"Error running step for agent {agent.name}: {e}")
                step_results[agent.id] = {"error": str(e)}

        return step_results

    def run_simulation_steps(self, num_steps: int):
        """Run specified number of simulation steps"""
        results = []

        for _ in range(num_steps):
            step_result = self.run_simulation_step()
            results.append(step_result)
            time.sleep(0.1)  # Small delay between steps

        return results

    def _should_train(self) -> bool:
        """Check if we should train the model"""
        # Check time interval
        if datetime.now() - self.last_training_time < self.training_interval:
            return False

        # Check minimum experiences
        if len(self.rl_engine.experience_buffer) < self.min_experiences_for_training:
            return False

        return True

    def _train_model(self):
        """Train the RL model"""
        try:
            logger.info("Starting model training...")

            loss = self.rl_engine.train()

            if loss is not None:
                self.last_training_time = datetime.now()

                # Save model checkpoint
                model_path = (
                    f"models/ppo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                )
                self.rl_engine.save_model(model_path)

                # Update database with model info
                self._save_model_to_db(model_path, loss)

                logger.info(f"Model training completed. Loss: {loss:.4f}")
            else:
                logger.info("Not enough experiences for training")

        except Exception as e:
            logger.error(f"Error during model training: {e}")

    def _save_model_to_db(self, model_path: str, loss: float):
        """Save model information to database"""
        try:
            # Calculate model statistics
            total_experiences = len(self.rl_engine.experience_buffer)
            avg_reward = np.mean(
                [
                    exp["reward"]
                    for exp in list(self.rl_engine.experience_buffer)[-1000:]
                ]
            )

            query = """
                INSERT INTO models
                (version, model_path, training_loss, reward_average, training_episodes, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            params = (
                f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_path,
                loss,
                avg_reward,
                total_experiences,
                True,
            )

            self.database.execute(query, params)

            # Deactivate previous models
            self.database.execute(
                "UPDATE models SET is_active = FALSE WHERE version != %s",
                (f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",),
            )

        except Exception as e:
            logger.error(f"Failed to save model to database: {e}")

    def _vector_to_action_dict(self, action_vector, agent):
        """Convert action vector to dictionary"""
        action_type, position_size = action_vector

        # Convert continuous action to discrete
        if action_type < 0.5:
            action_dict = {"type": "HOLD"}
        elif action_type < 1.5:
            action_dict = {"type": "BUY"}
        else:
            action_dict = {"type": "SELL"}

        # Add position size for buy/sell actions
        if action_dict["type"] in ["BUY", "SELL"]:
            action_dict["position_size"] = float(position_size)
        else:
            action_dict["position_size"] = 0.5

        return action_dict

    def _state_vector_to_dict(self, state_vector, agent):
        """Convert state vector back to dictionary"""
        # This is a simplified conversion - in practice, you'd want to store more context
        return {
            "agent_id": agent.id,
            "state_vector": state_vector.tolist(),
            "timestamp": datetime.now(),
        }

    def _log_progress(self, step_count: int, step_results: Dict):
        """Log simulation progress"""
        total_pnl = 0
        total_trades = 0
        active_agents = 0

        for agent_id, result in step_results.items():
            if "performance" in result:
                perf = result["performance"]
                total_pnl += perf.get("total_pnl", 0)
                total_trades += perf.get("total_trades", 0)
                active_agents += 1

        avg_pnl = total_pnl / active_agents if active_agents > 0 else 0

        logger.info(
            f"Step {step_count}: "
            f"Active Agents: {active_agents}, "
            f"Total P&L: ${total_pnl:,.2f}, "
            f"Avg P&L: ${avg_pnl:,.2f}, "
            f"Total Trades: {total_trades}, "
            f"Experience Buffer: {len(self.rl_engine.experience_buffer)}"
        )

    def get_system_status(self):
        """Get current system status"""
        agent_statuses = []
        total_pnl = 0

        for agent in self.agents:
            performance = agent.get_performance_metrics()
            agent_statuses.append(
                {
                    "id": agent.id,
                    "name": agent.name,
                    "strategy": agent.strategy_type,
                    "status": agent.status,
                    "balance": agent.current_balance,
                    "pnl": performance["total_pnl"],
                    "trades": performance["total_trades"],
                    "win_rate": performance["win_rate"],
                }
            )
            total_pnl += performance["total_pnl"]

        rl_stats = self.rl_engine.get_training_stats()

        return {
            "timestamp": datetime.now(),
            "is_running": self.is_running,
            "agents": agent_statuses,
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents if a.status == "active"]),
            "total_pnl": total_pnl,
            "experience_buffer_size": len(self.rl_engine.experience_buffer),
            "rl_stats": rl_stats,
            "last_training": self.last_training_time.isoformat()
            if self.last_training_time
            else None,
        }

    def get_agent_details(self, agent_id: str):
        """Get detailed information about a specific agent"""
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            return None

        performance = agent.get_performance_metrics()
        recent_trades = agent.trades[-10:] if agent.trades else []

        return {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "strategy": agent.strategy_type,
                "status": agent.status,
                "initial_capital": agent.initial_capital,
                "current_balance": agent.current_balance,
            },
            "performance": performance,
            "positions": agent.positions,
            "recent_trades": recent_trades,
            "episode_id": agent.episode_id,
            "step_count": agent.step_count,
        }
