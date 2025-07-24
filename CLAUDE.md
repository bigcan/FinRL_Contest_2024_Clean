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