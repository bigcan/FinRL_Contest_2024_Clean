# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the FinRL Contest 2024 repository for the ACM ICAIF 2024 competition, featuring two main tasks:
- **Task 1**: Cryptocurrency Trading with Ensemble Learning (Bitcoin LOB data)
- **Task 2**: LLM-Engineered Signals with Reinforcement Learning from Market Feedback (RLMF)

## Development Principles

- No mock models or fallback plans or shortcuts unless explicitly approved by the user
- **IMPORTANT**: Update README.md files after each development phase to maintain accurate documentation
  - Update main project README with major changes
  - Update task-specific READMEs with implementation details
  - Document critical bug fixes, new features, and architectural changes
  - Include update logs with dates at the top of README files
- When you create a temporary solution to fix an issue, don't forget to go back to original plan

## Commands

### Task 1: Cryptocurrency Trading

**CURRENT WORKING COMMANDS (Updated 2025-07-29 - Multi-Episode Training):**

#### Option 1: Multi-Episode Training Framework (100% Reliable) ⭐ RECOMMENDED
```bash
cd development/task1
# Validate train/validation split configuration
python validate_train_val_split.py

# Train ensemble models (65 episodes, proper data split)
python src/task1_ensemble.py

# Evaluate trained models on holdout validation set
python src/task1_eval.py

# Quick functionality test (3 episodes)
python quick_multi_episode_test.py
```
**Features**: 65 episodes, 10K samples per episode, zero data leakage, ~39 min training

#### Option 2: Refactored Framework (83% Working, 25% Faster)
```bash
cd development/task1
# First validate framework
python run_refactored_validation.py

# If 5/6 tests pass, use refactored training
python src_refactored/train_enhanced_ensemble.py
```

#### Quick Reference
```bash
# For contest use, follow this guide:
cat CONTEST_USAGE.md

# Check validated models:
ls complete_production_results/production_models/
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

## Current Project Status (Updated 2025-07-29)

### Task 1: Multi-Episode Training Ready ✅
**Multi-Episode Training Implementation:**
- ✅ **65-episode training** with proper train/validation split (80/20)
- ✅ **Zero data leakage** - validation data completely preserved
- ✅ **650K training samples** (98.6% of available training data)
- ✅ **Episode-based logging** and convergence analysis
- ✅ **~39 minute training time** for full ensemble

**Validated Models Available:**
- `complete_production_results/production_models/` contains 3 verified models
- All models pass integrity checks (no NaN/Inf, proper loading)
- Training verification results available in `training_verification_results.json`

**Framework Status:**
- **Multi-Episode Framework**: 100% working, proper data split validation
- **Refactored Framework**: 83% working (5/6 tests pass), 25% performance improvement
- **Legacy Single-Episode**: 100% working but deprecated (data leakage risk)

**Recent Major Updates (2025-07-29):**
- ✅ **IMPLEMENTED**: Multi-episode training with 65 episodes
- ✅ **FIXED**: Train/validation data leakage (eliminated 91K sample contamination)
- ✅ **ADDED**: Comprehensive data split validation tools
- ✅ **OPTIMIZED**: Configuration for ensemble diversity and convergence analysis
- ✅ **VALIDATED**: Episode boundaries respect temporal data structure

**Current Known Issues:**
- CUDA device mismatch during episode transitions (under investigation)
- Some import warnings in refactored framework (non-blocking, fallbacks work)

### Task 2: Status Unknown
- Commands listed above are placeholder
- Requires separate assessment

## Important Context for AI Assistants

### What Works Reliably
1. **Original training framework** (`python src/task1_ensemble.py`)
2. **Model validation system** (`python training_verification_test.py`)
3. **Validated production models** in `complete_production_results/`

### What's Experimental/In-Progress
1. **Refactored framework** - mostly working but has fallbacks
2. **GPU scripts** in `archive_experiments/` - archived experimental code
3. **Various training approaches** - multiple attempts archived

### Contest Strategy
- **Primary**: Use original framework for reliability
- **Secondary**: Use refactored framework when performance boost needed
- **Fallback**: Validated models always available for immediate use
- **Documentation**: Follow `CONTEST_USAGE.md` for quick reference

### Maintenance Philosophy
- **Contest-first approach**: Winning > clean code during contest
- **Working solutions preserved**: Don't break what works
- **Documentation updated**: Reflects current working state
- **Post-contest cleanup planned**: Major refactoring after contest

### Commands to Remember
```bash
# Validate train/validation split (NEW)
python validate_train_val_split.py

# Multi-episode training (UPDATED)
python src/task1_ensemble.py

# Quick multi-episode test (NEW) 
python quick_multi_episode_test.py

# Validate refactored framework  
python run_refactored_validation.py

# Contest quick reference
cat CONTEST_USAGE.md

# Find working models
ls complete_production_results/production_models/
```