CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    initial_capital DECIMAL(15,2) DEFAULT 500000000.00,
    current_balance DECIMAL(15,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    profit_loss DECIMAL(15,2) DEFAULT 0,
    commission DECIMAL(10,4) DEFAULT 0,
    strategy_reasoning TEXT
);

CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    rsi DECIMAL(5,2),
    macd DECIMAL(10,6),
    bollinger_upper DECIMAL(10,4),
    bollinger_lower DECIMAL(10,4)
);

CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp);

CREATE TABLE experiences (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    episode_id INTEGER NOT NULL,
    step_number INTEGER NOT NULL,
    state JSONB NOT NULL,
    action JSONB NOT NULL,
    reward DECIMAL(10,4) NOT NULL,
    next_state JSONB NOT NULL,
    done BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_experiences_agent_episode ON experiences(agent_id, episode_id);

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    training_loss DECIMAL(10,6),
    reward_average DECIMAL(10,4),
    win_rate DECIMAL(5,4),
    training_episodes INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

CREATE TABLE performance_summary (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    total_pnl DECIMAL(15,2),
    profit_factor DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4)
);

CREATE INDEX idx_performance_agent_date ON performance_summary(agent_id, date);
