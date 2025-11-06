# rl/rl_engine.py
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from typing import List, Dict, Any, Tuple
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Policy head (actions)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head (state value)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Policy output (continuous actions)
        policy = self.tanh(self.policy_head(x))

        # Value output
        value = self.value_head(x)

        return policy, value


class RLEngine:
    def __init__(self, state_dim=20, action_dim=2, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Policy and value networks
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_network = PolicyNetwork(state_dim, 1).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=lr
        )
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)

        # Experience buffer
        self.experience_buffer = deque(maxlen=100000)
        self.batch_size = 64

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # PPO clipping parameter
        self.entropy_coef = 0.01  # Entropy coefficient

        # Training metrics
        self.training_losses = []
        self.episode_rewards = []

        # Create a dummy model attribute to match test expectations
        self.model = self.policy_network

    def add_experience(
        self,
        agent_id: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ):
        """Add experience to replay buffer"""
        experience = {
            "agent_id": agent_id,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "timestamp": datetime.now(),
        }

        self.experience_buffer.append(experience)

        # Store to database
        self._store_experience_to_db(experience)

    def _store_experience_to_db(self, experience):
        """Store experience to database"""
        try:
            from database.connection import DatabaseConnection

            db = DatabaseConnection()

            query = """
                INSERT INTO experiences
                (agent_id, episode_id, step_number, state, action, reward, next_state, done)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Extract episode_id and step from action or create defaults
            episode_id = experience.get("action", {}).get(
                "episode_id", str(uuid.uuid4())
            )
            step_number = experience.get("action", {}).get("step", 0)

            params = (
                experience["agent_id"],
                episode_id,
                step_number,
                experience["state"],
                experience["action"],
                experience["reward"],
                experience["next_state"],
                experience["done"],
            )

            db.execute(query, params)

        except Exception as e:
            logger.error(f"Failed to store experience to database: {e}")

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.policy_network(state_tensor)

        # Add exploration noise
        action = policy.squeeze().cpu().numpy()
        action += np.random.normal(0, 0.1, size=action.shape)

        # Clamp actions to valid range
        action = np.clip(action, [-1.0, 0.0], [2.0, 1.0])

        return action, value.squeeze()

    def train(self) -> float:
        """Train the model using experiences from buffer"""
        if len(self.experience_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(list(self.experience_buffer), self.batch_size)

        # Prepare batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for exp in batch:
            # Convert state dict to numpy array
            state_vec = self._state_dict_to_vector(exp["state"])
            next_state_vec = self._state_dict_to_vector(exp["next_state"])

            states.append(state_vec)
            actions.append(self._action_dict_to_vector(exp["action"]))
            rewards.append(exp["reward"])
            next_states.append(next_state_vec)
            dones.append(exp["done"])

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute returns
        with torch.no_grad():
            _, next_values = self.policy_network(next_states)
            returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)

        # Get current values and policies
        current_policies, current_values = self.policy_network(states)

        # Compute advantages
        advantages = returns - current_values.squeeze()

        # PPO Policy Update
        policy_loss = self._compute_policy_loss(current_policies, actions, advantages)

        # Value Update
        value_loss = nn.MSELoss()(current_values.squeeze(), returns)

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.value_optimizer.step()

        total_loss = policy_loss.item() + value_loss.item()
        self.training_losses.append(total_loss)

        logger.info(
            f"Training step - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}"
        )

        return total_loss

    def _compute_policy_loss(self, policies, actions, advantages):
        """Compute PPO policy loss"""
        # Get old policy (for comparison)
        with torch.no_grad():
            old_policies, _ = self.policy_network(torch.zeros_like(actions))

        # Compute policy ratio
        ratio = policies / (old_policies + 1e-8)

        # Compute surrogate loss
        surrogate1 = ratio * advantages.unsqueeze(-1)
        surrogate2 = torch.clamp(
            ratio, 1 - self.epsilon, 1 + self.epsilon
        ) * advantages.unsqueeze(-1)

        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Add entropy bonus
        entropy = -torch.sum(policies * torch.log(policies + 1e-8), dim=-1).mean()
        policy_loss -= self.entropy_coef * entropy

        return policy_loss

    def _state_dict_to_vector(self, state_dict):
        """Convert state dictionary to vector"""
        # This should match the conversion in environment.py
        state_vector = [
            state_dict.get("cash_balance", 0)
            / 500000000,  # Normalize by default capital
            state_dict.get("total_pnl", 0) / 500000000,
            state_dict.get("available_margin", 0) / 500000000,
            len(state_dict.get("positions", [])),
            state_dict.get("rsi", 50) / 100.0,
            state_dict.get("macd", 0) / 1.0,
            state_dict.get("bollinger_position", 0.5),
            state_dict.get("atr", 0) / 100.0,
            state_dict.get("volume_ratio", 1.0) / 10.0,
            state_dict.get("market_volatility", 0),
            state_dict.get("hour_of_day", 12) / 24.0,
            state_dict.get("day_of_week", 3) / 7.0,
            1.0 if state_dict.get("market_session") == "us" else 0.0,
            1.0 if state_dict.get("market_session") == "asia" else 0.0,
            1.0 if state_dict.get("market_session") == "europe" else 0.0,
            1.0 if state_dict.get("strategy_type") == "conservative" else 0.0,
            1.0 if state_dict.get("strategy_type") == "aggressive" else 0.0,
            1.0 if state_dict.get("strategy_type") == "balanced" else 0.0,
            1.0 if state_dict.get("strategy_type") == "trend" else 0.0,
        ]

        return np.array(state_vector, dtype=np.float32)

    def _action_dict_to_vector(self, action_dict):
        """Convert action dictionary to vector"""
        action_type = action_dict.get("type", "HOLD")

        # Convert action type to continuous value
        if action_type == "HOLD":
            action_val = 0.0
        elif action_type == "BUY":
            action_val = 1.0
        elif action_type == "SELL":
            action_val = 2.0
        else:
            action_val = 0.0

        # Position size (0.0 to 1.0)
        position_size = action_dict.get("position_size", 0.5)

        return np.array([action_val, position_size], dtype=np.float32)

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "value_network": self.value_network.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "training_losses": self.training_losses,
                "episode_rewards": self.episode_rewards,
            },
            path,
        )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.training_losses = checkpoint.get("training_losses", [])
        self.episode_rewards = checkpoint.get("episode_rewards", [])

        logger.info(f"Model loaded from {path}")

    def get_training_stats(self):
        """Get training statistics"""
        if not self.training_losses:
            return {}

        return {
            "total_steps": len(self.training_losses),
            "avg_loss": np.mean(self.training_losses[-100:])
            if len(self.training_losses) >= 100
            else np.mean(self.training_losses),
            "latest_loss": self.training_losses[-1] if self.training_losses else 0,
            "buffer_size": len(self.experience_buffer),
        }
