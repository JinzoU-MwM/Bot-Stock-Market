from .base_agent import BaseAgent
from .strategy_agents import (
    ConservativeAgent,
    AggressiveAgent,
    BalancedAgent,
    TrendAgent,
    MeanReversionAgent,
)


class AgentFactory:
    _strategies = {
        "conservative": ConservativeAgent,
        "aggressive": AggressiveAgent,
        "balanced": BalancedAgent,
        "trend": TrendAgent,
        "mean_reversion": MeanReversionAgent,
    }

    @classmethod
    def create_agent(cls, strategy_type, name, initial_capital):
        """Create an agent with specified strategy"""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        agent_class = cls._strategies[strategy_type]
        return agent_class(name, strategy_type, initial_capital)

    @classmethod
    def get_available_strategies(cls):
        """Get list of available strategy types"""
        return list(cls._strategies.keys())
