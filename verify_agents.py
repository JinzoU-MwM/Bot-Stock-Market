#!/usr/bin/env python3

from agents.agent_factory import AgentFactory


def main():
    print("=== Agent Framework Verification ===\n")

    # Test creating different agent types
    strategies = ["conservative", "aggressive", "balanced", "trend", "mean_reversion"]

    print("Agent Strategy Configuration:")
    for strategy in strategies:
        agent = AgentFactory.create_agent(strategy, f"test_{strategy}", 500000000)
        print(
            f"  {strategy:15s}: Max Position={agent.get_max_position_size():.1%}, Risk={agent.get_risk_tolerance()}"
        )

    print("\n" + "=" * 50)
    print("Sample Trading Execution:")

    # Test sample trade
    agent = AgentFactory.create_agent("balanced", "test_trade", 500000000)
    print(f"Initial balance: ${agent.current_balance:,.2f}")

    result = agent.execute_action(
        {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150}
    )
    print(f"Buy result: {result['success']}")
    print(f"Trade cost: ${15000 + 150:.2f}")
    print(f"New balance: ${agent.current_balance:,.2f}")
    print(f"Positions: {list(agent.positions.keys())}")

    # Test sell
    result = agent.execute_action(
        {"type": "SELL", "symbol": "AAPL", "quantity": 100, "price": 160}
    )
    print(f"\nSell result: {result['success']}")
    print(f"Profit/Loss: ${result.get('profit_loss', 0):.2f}")
    print(f"Final balance: ${agent.current_balance:,.2f}")
    print(f"Positions: {list(agent.positions.keys())}")

    # Performance metrics
    metrics = agent.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key:15s}: {value}")

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    main()
