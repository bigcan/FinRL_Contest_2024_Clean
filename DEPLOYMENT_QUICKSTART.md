# 🚀 FinRL Contest 2024 - Deployment Quick Start Guide

## Overview
This guide provides step-by-step instructions for deploying the FinRL Contest 2024 cryptocurrency trading system to production.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, or better)
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 32GB minimum
- **Storage**: 100GB SSD

### Software Requirements
```bash
# Python environment
Python 3.8-3.10
CUDA 11.7+
PyTorch 1.13+

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Step 1: Pre-Deployment Validation

### 1.1 Run System Tests
```bash
cd development/task1/src

# Test extended training framework
python test_extended_training.py

# Validate model performance across market periods
python market_period_validator.py ensemble_optimized_phase2
```

### 1.2 Run Production Evaluation
```bash
# Comprehensive production readiness check
python production_evaluation_suite.py --ensemble_path ensemble_optimized_phase2 --gpu 0
```

**Expected Output:**
- Production Readiness Score > 80%
- All critical checks passed
- Performance metrics within thresholds

## 🔧 Step 2: Environment Setup

### 2.1 Create Production Environment
```bash
# Create virtual environment
python -m venv finrl_prod_env
source finrl_prod_env/bin/activate  # Linux/Mac
# or
finrl_prod_env\Scripts\activate  # Windows

# Install production dependencies
pip install -r requirements.txt
pip install gunicorn uvicorn  # For API serving
```

### 2.2 Configure Environment Variables
```bash
# Create .env file
cat > .env << EOF
FINRL_ENV=production
GPU_DEVICE=0
MODEL_PATH=ensemble_optimized_phase2
LOG_LEVEL=INFO
MONITORING_PORT=8080
API_PORT=8000
EOF
```

## 📦 Step 3: Deploy Models

### 3.1 Package Models
```bash
# Create deployment package
python development/shared/submission_package_creator.py

# Verify package contents
ls -la finrl_deployment_package/
```

### 3.2 Deploy to Production Server
```bash
# Copy to production server
scp -r finrl_deployment_package/ user@production-server:/opt/finrl/

# SSH to production server
ssh user@production-server

# Set up model directory
cd /opt/finrl
chmod +x scripts/*.sh
```

## 🚦 Step 4: Start Services

### 4.1 Start Monitoring System
```bash
# Start monitoring daemon
python development/task1/src/production_monitoring_system.py &

# Verify monitoring is running
curl http://localhost:8080/status
```

### 4.2 Start Trading Service
```bash
# Start with paper trading first
python scripts/start_trading.py --mode paper --duration 24h

# After validation, start live trading
python scripts/start_trading.py --mode live --risk-limit 0.02
```

## 📊 Step 5: Monitor Performance

### 5.1 Access Monitoring Dashboard
```
http://your-server:8080/dashboard
```

### 5.2 Set Up Alerts
```python
# Configure alerts in monitoring_config.json
{
  "alerts": {
    "email": "your-email@example.com",
    "slack_webhook": "https://hooks.slack.com/...",
    "thresholds": {
      "max_drawdown": 0.03,
      "min_sharpe": 0.5,
      "error_rate": 0.001
    }
  }
}
```

## 🔄 Step 6: Continuous Operations

### Daily Tasks
1. **Morning Check** (9:00 AM)
   ```bash
   python scripts/daily_health_check.py
   ```

2. **Performance Review** (4:00 PM)
   ```bash
   python scripts/generate_daily_report.py
   ```

3. **EOD Backup** (6:00 PM)
   ```bash
   python scripts/backup_data.py
   ```

### Weekly Tasks
1. **Deep Performance Analysis**
   ```bash
   python scripts/weekly_analysis.py
   ```

2. **Model Performance Review**
   ```bash
   python development/task1/src/production_evaluation_suite.py
   ```

### Monthly Tasks
1. **Model Retraining Evaluation**
   ```bash
   python scripts/evaluate_retraining_need.py
   ```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory Error**
   ```bash
   # Reduce batch size in config
   export BATCH_SIZE=128
   ```

2. **High Latency**
   ```bash
   # Check GPU utilization
   nvidia-smi
   # Restart with performance mode
   sudo nvidia-smi -pm 1
   ```

3. **Connection Issues**
   ```bash
   # Check network connectivity
   ping exchange-api.com
   # Restart with retry logic
   python scripts/start_trading.py --max-retries 5
   ```

## 📈 Performance Optimization

### Quick Wins
1. **Enable GPU Persistence**
   ```bash
   sudo nvidia-smi -pm 1
   ```

2. **Optimize Batch Processing**
   ```python
   # In config.py
   BATCH_SIZE = 256  # Increase if GPU memory allows
   ```

3. **Use Compiled Models**
   ```python
   # Enable torch.jit compilation
   model = torch.jit.script(model)
   ```

## 🔐 Security Checklist

- [ ] API keys stored in environment variables
- [ ] SSL/TLS enabled for all connections
- [ ] Access logs enabled and monitored
- [ ] Regular security updates applied
- [ ] Backup encryption enabled
- [ ] Rate limiting configured
- [ ] IP whitelist implemented

## 📞 Support

### Internal Resources
- Documentation: `/docs/production_guide.pdf`
- Runbooks: `/runbooks/`
- Scripts: `/scripts/`

### Monitoring Endpoints
- Health Check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Dashboard: `http://localhost:8080`

### Emergency Procedures
1. **Stop Trading**
   ```bash
   python scripts/emergency_stop.py
   ```

2. **Export Positions**
   ```bash
   python scripts/export_positions.py
   ```

3. **Generate Incident Report**
   ```bash
   python scripts/incident_report.py
   ```

## ✅ Post-Deployment Checklist

### Hour 1
- [ ] All services running
- [ ] No critical alerts
- [ ] Latency < 50ms
- [ ] Paper trading active

### Day 1
- [ ] Performance metrics normal
- [ ] No unexpected errors
- [ ] Monitoring data collected
- [ ] Daily report generated

### Week 1
- [ ] Sharpe ratio > 0.5
- [ ] Drawdown < 5%
- [ ] System stable
- [ ] Ready for full deployment

## 🎉 Success Criteria

Your deployment is successful when:
1. ✅ Production Readiness Score > 80%
2. ✅ All services running without errors
3. ✅ Performance metrics meet thresholds
4. ✅ Monitoring and alerts functional
5. ✅ Paper trading profitable for 1 week

---

**Need Help?** Check the comprehensive [DEPLOYMENT_AND_EVALUATION_PLAN.md](DEPLOYMENT_AND_EVALUATION_PLAN.md) for detailed information.

*Last Updated: 2025-07-25 | Version: 1.0*