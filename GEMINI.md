### Project Overview

This project is for the FinRL Contest 2024, Task 1, which focuses on cryptocurrency trading using ensemble methods. The goal is to develop robust trading agents that can learn from Limit Order Book (LOB) data for Bitcoin.

### Key Components

*   **Environment:** `trade_simulator.py` provides a market replay simulator for both training (`TradeSimulator`) and evaluation (`EvalTradeSimulator`).
*   **Agents:** The `erl_agent.py` file defines the reinforcement learning agents, including `AgentDoubleDQN`, `AgentD3QN`, and `AgentTwinD3QN`. These are variations of the Deep Q-Network (DQN) algorithm.
*   **Configuration:** `erl_config.py` manages the configuration for the training process, including hyperparameters, environment settings, and agent parameters.
*   **Training:** `task1_ensemble.py` is the main script for training an ensemble of agents. It trains multiple agents and saves their models.
*   **Evaluation:** `task1_eval.py` is used to evaluate the performance of the trained ensemble. It loads the saved models and simulates trading on a validation dataset.
*   **Data:** The project uses pre-processed Bitcoin LOB data (`BTC_1sec_predict.npy`) for training and evaluation.

### Workflow

1.  **Training:** The `task1_ensemble.py` script is run to train a collection of agents (e.g., `AgentD3QN`, `AgentDoubleDQN`). The trained models are saved to a directory (e.g., `ensemble_teamname/ensemble_models`).
2.  **Evaluation:** The `task1_eval.py` script is then used to evaluate the trained ensemble. It loads the saved models and runs them in the `EvalTradeSimulator` environment. The evaluation metrics include Sharpe ratio, max drawdown, and return over max drawdown.

### Conventions

*   **Framework:** The project uses PyTorch for building and training the neural network models.
*   **Agents:** The agents are based on DQN and its variants (Double DQN, Dueling DQN).
*   **Ensemble Method:** The default ensemble method is majority voting, but the project encourages participants to explore other methods.
*   **Configuration:** A centralized `Config` class is used to manage all hyperparameters and settings.
*   **File Naming:** The files are well-named and organized, with a clear separation of concerns (e.g., `erl_agent.py` for agents, `erl_config.py` for configuration).

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