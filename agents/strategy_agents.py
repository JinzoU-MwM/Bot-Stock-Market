from .base_agent import BaseAgent


class ConservativeAgent(BaseAgent):
    def get_max_position_size(self):
        return 0.10  # 10% of capital per position

    def get_risk_tolerance(self):
        return "low"


class AggressiveAgent(BaseAgent):
    def get_max_position_size(self):
        return 0.20  # 20% of capital per position

    def get_risk_tolerance(self):
        return "high"


class BalancedAgent(BaseAgent):
    def get_max_position_size(self):
        return 0.15  # 15% of capital per position

    def get_risk_tolerance(self):
        return "medium"


class TrendAgent(BaseAgent):
    def get_max_position_size(self):
        return 0.15  # 15% of capital per position

    def get_risk_tolerance(self):
        return "medium"


class MeanReversionAgent(BaseAgent):
    def get_max_position_size(self):
        return 0.12  # 12% of capital per position

    def get_risk_tolerance(self):
        return "low"
