# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the FinRL Contest 2024 repository for the ACM ICAIF 2024 competition, featuring two main tasks:
- **Task 1**: Cryptocurrency Trading with Ensemble Learning (Bitcoin LOB data)
- **Task 2**: LLM-Engineered Signals with Reinforcement Learning from Market Feedback (RLMF)

## Development Principles

- No mock models or fallback plans or shortcuts unless explicitly approved by the user

## Commands

### Task 1: Cryptocurrency Trading
```bash
# Train ensemble models (optional: specify GPU ID)
python task1_ensemble.py [GPU_ID]

# Evaluate trained models
python task1_eval.py [GPU_ID]

# Install dependencies
pip install -r requirements.txt
```

### Task 2: LLM-based Signal Generation
```bash
# Train LLM with RLMF
python task2_train.py

# Evaluate fine-tuned model
python task2_eval.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Task 1 Structure
- **trade_simulator.py**: Core trading environment with vectorized market replay
- **erl_agent.py**: DQN-based reinforcement learning agents
- **erl_net.py**: Neural network architectures (DQN, LSTM variants)
- **task1_ensemble.py**: Ensemble training with multiple DRL agents
- **task1_eval.py**: Evaluation with voting mechanism
- **data_config.py**: Data paths and configuration

Key classes:
- `Ensemble`: Manages multiple DRL agents for ensemble learning
- `TradeSimulatorVecEnv`: Vectorized trading environment
- Various Agent classes: AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

### Task 2 Structure
- **task2_env.py**: RLMF environment for fine-tuning
- **task2_signal.py**: LLM prompt construction and signal generation
- **task2_train.py**: LoRA fine-tuning with market feedback
- **task2_eval.py**: Signal evaluation with long/short strategy
- **task2_config.py**: Model and hyperparameter configuration

Key components:
- Uses Llama-3.2-3B-Instruct with LoRA adaptation
- Custom reward computation based on lookahead returns
- Fixed long/short strategy (top/bottom 3 stocks)

## Data Requirements

### Task 1
- Download Bitcoin LOB data to `data/` directory
- Files needed: `BTC_1sec.csv` and `BTC_1sec_predict.npy`
- Data source: Google Drive link in README

### Task 2
- Extract `task2_dsets.zip` for train/test datasets
- Contains stock OHLCV data and news headlines
- Pre-split into training and testing periods

## Key Development Workflows

### Task 1 Development
1. Download and place data in `data/` directory
2. Modify ensemble configuration in `task1_ensemble.py` (agents, hyperparameters)
3. Train ensemble: `python task1_ensemble.py`
4. Models saved to `ensemble_teamname/ensemble_models/`
5. Evaluate: `python task1_eval.py`
6. Review metrics: Sharpe ratio, max drawdown, RoMaD

### Task 2 Development
1. Extract dataset from `task2_dsets.zip`
2. Configure dates and parameters in `task2_train.py`
3. Ensure HuggingFace access for Llama model
4. Train: `python task2_train.py` (requires 20GB+ GPU)
5. Model saved to `path_to_save_model/`
6. Evaluate: `python task2_eval.py`
7. Review cumulative returns and win/loss metrics

## Important Technical Details

- **GPU Usage**: Both tasks support CUDA acceleration. Pass GPU ID as command line argument.
- **Encoding Issue**: Some requirements.txt files are UTF-16LE encoded (may need conversion)
- **Model Storage**: Task 1 saves multiple agent models; Task 2 saves LoRA adapters
- **Evaluation Constraints**: Evaluation scripts must maintain compatibility with contest framework

## Submission Requirements

Both tasks require:
- All trained model files
- Modified Python scripts (maintaining original function signatures)
- README explaining methodology and changes
- requirements.txt for any additional dependencies
- Working evaluation scripts that match expected outputs

### Feature Engineering Strategy

Our strategy is to create a robust feature set by focusing on three key areas, in order of priority: market microstructure, data transformation, and external data sources.

#### 1. Market Microstructure (Highest Priority)

These features capture immediate supply and demand from the Limit Order Book (LOB) and are critical for short-term prediction.

*   **Core LOB Indicators (Implement First):**
    *   **Bid-Ask Spread:** Measures market liquidity.
    *   **Order Book Depth:** Shows the volume of buy/sell orders at different price levels.
    *   **Volume Weighted Average Price (VWAP):** Provides the average price weighted by volume.
*   **Order Flow & Imbalance (Key Predictive Signals):**
    *   **Order Flow Imbalance (OFI):** Captures the net direction of order flow.
    *   **Normalized Order Book Imbalance (NOBI):** Measures the imbalance between bid and ask volume.
    *   **Trade Flow Imbalance (TFI):** Tracks the imbalance between buy and sell trades.
*   **Advanced Microstructure (Future Work):**
    *   **Microprice:** A more accurate valuation than the mid-price.
    *   **Order Arrival & Cancellation Rates:** Indicates market participant intentions.

#### 2. Data Transformation (Essential for Model Stability)

Raw data is often non-stationary. These transformations create statistically robust features that improve model performance.

*   **Stationarity:**
    *   **Standard Method:** Use log returns (`log(price_t) - log(price_{t-1})`) to stabilize the time series. This will be our primary approach.
    *   **Advanced Method (Future Research):** Explore Fractional Differentiation to achieve stationarity while preserving data memory.
*   **Normalization:**
    *   Apply `StandardScaler` to the final feature set to standardize the input for the neural network.

#### 3. External & Alternative Data (For a Broader Market View)

Integrating external data can provide context that is not available in the LOB data alone.

*   **On-Chain Analysis (Future Work):**
    *   Transaction Volume & Active Addresses
    *   Exchange Inflows/Outflows
*   **Social Sentiment Analysis (Future Work):**
    *   Sentiment Scores (Twitter, Reddit)
    *   Fear & Greed Index

#### Other High-Impact Techniques to Consider

*   **Volatility Features:** Use Historical Volatility or ATR as features to help the model identify market regimes.
*   **Time-Based Features:** Encode time of day/week using sinusoidal transformations to capture cyclical patterns.
*   **Market Regime Identification:** Use indicators like ADX to classify the market as trending or ranging.

### Real-Time Feature Engineering Considerations (For Live Deployment)

This section outlines the critical challenges and solutions for implementing feature engineering in a live, real-time trading environment.

1.  **Adopt a Streaming Architecture:**
    *   **Challenge:** Batch processing is too slow for live trading.
    *   **Solution:** Use a streaming architecture with a message queue (e.g., Kafka, Redis) to ingest WebSocket data feeds. Process this data using a stream processing framework (e.g., Flink, Spark Streaming) that can maintain the state required for stateful features.

2.  **Implement Incremental Calculations:**
    *   **Challenge:** Re-calculating features from scratch on each tick is computationally expensive.
    *   **Solution:** Use incremental update algorithms. EMAs are naturally incremental. For SMAs and rolling standard deviations, use efficient algorithms that add the new value and subtract the oldest value from the rolling window.

3.  **Handle Standardization with a "Frozen" Scaler:**
    *   **Challenge:** `StandardScaler` is a batch tool and cannot be fit on a live, growing dataset.
    *   **Solution:** Pre-train the scaler on a large, representative historical dataset to get a "frozen" mean and standard deviation. Use these frozen values to scale new data points in real-time. Periodically retrain the scaler (e.g., daily or weekly) to adapt to concept drift.

4.  **Optimize Computational Resources:**
    *   **Challenge:** Latency is critical.
    *   **Solution:** Use high-performance libraries (NumPy, Pandas), pre-compute expensive features less frequently, and use optimized hardware. For ultra-low latency, consider dedicated hardware like FPGAs.

5.  **Separate Concerns with a Modular Pipeline:**
    *   **Challenge:** A monolithic system is hard to scale and debug.
    *   **Solution:** Build a modular pipeline with separate services for: 
        *   **Data Ingestion:** Connects to the exchange.
        *   **Feature Calculation:** Consumes data and computes features.
        *   **RL Agent:** Makes decisions based on features.
        *   **Execution:** Sends orders to the exchange.
    *   This allows for independent scaling and improved reliability.

Here's a systematic approach to optimizing reward functions in reinforcement learning
  for financial trading, based on common practices and research:

   1. Multi-Objective Reward Design:
       * Combine Key Metrics: Instead of a single metric (e.g., profit), design a reward
         function that is a weighted sum of multiple financial objectives. This could
         include:
           * Profit/Loss: The primary goal.
           * Risk-Adjusted Returns: Sharpe Ratio, Sortino Ratio, Calmar Ratio (or
             components thereof).
           * Drawdown: Penalize large drawdowns.
           * Transaction Costs: Directly penalize commissions, slippage, and spread.
           * Action Diversity/Conservatism: Penalize excessive trading or holding too
             long (as identified in your problem).
       * Weighting: The weights for each component can be fixed, or dynamically adjusted
         based on market regimes or desired risk profiles.

   2. Reward Shaping (Potential-Based):
       * Guide Exploration: Introduce an auxiliary reward function that guides the agent
         towards desirable states or behaviors without changing the optimal policy.
         Potential-based reward shaping is theoretically sound as it preserves the
         optimal policy.
       * Examples:
           * Reward for being in a profitable position.
           * Reward for taking actions that reduce risk.
           * Reward for maintaining a balanced portfolio.

   3. Adaptive/Dynamic Reward Functions:
       * Market Regime Awareness: Adjust reward components or their weights based on
         identified market regimes (e.g., trending, ranging, high volatility, low
         volatility). For example, in a high-volatility regime, the risk penalty might be
         increased.
       * Performance-Based Adjustment: Dynamically modify the reward function based on the
         agent's recent performance. If the agent is consistently making losses, the reward
          function might temporarily emphasize risk reduction more heavily.

   4. Inverse Reinforcement Learning (IRL) / Learning from Demonstrations:
       * Expert Behavior: If you have access to historical data of successful trading
         strategies or expert human traders, IRL can be used to infer a reward function
         that explains their observed behavior. The agent then tries to optimize this
         learned reward function.

   5. Automated Reward Design / Meta-Learning:
       * Evolutionary Algorithms: Use evolutionary algorithms (e.g., genetic algorithms)
         to search for optimal reward function parameters or structures. This involves
         defining a search space for reward functions and evaluating their performance.
       * Reinforcement Learning for Reward Design: Train a meta-RL agent to design the
         reward function for the primary trading agent.

   6. Considerations for Financial Trading:
       * Sparsity: Financial rewards can be sparse (e.g., only at the end of a trade or
         episode). Reward shaping can help alleviate this.
       * Non-Stationarity: Financial markets are non-stationary. Reward functions might
         need to adapt over time.
       * Risk Management: Explicitly incorporate risk metrics and penalties into the
         reward function to prevent overly aggressive or risky behavior.