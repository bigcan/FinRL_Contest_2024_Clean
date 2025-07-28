import numpy as np
import pandas as pd

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    if isinstance(returns_pct, (pd.Series, pd.DataFrame)):
        returns = returns_pct.copy()
    else:
        returns = pd.Series(returns_pct)
    
    # Calculate cumulative returns: (1 + r1) * (1 + r2) * ... - 1
    return (1 + returns).cumprod() - 1

def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return float
    """
    returns = np.array(returns_pct)
    if returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (returns.mean()-risk_free) / returns.std()
    return sharpe_ratio

def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    if isinstance(returns_pct, (pd.Series, pd.DataFrame)):
        returns = returns_pct.copy()
    else:
        returns = pd.Series(returns_pct)
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    peak = cum_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cum_returns - peak) / peak
    
    # Return maximum drawdown (most negative value)
    return drawdown.min()

def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct).iloc[-1] if hasattr(cumulative_returns(returns_pct), 'iloc') else cumulative_returns(returns_pct)[-1]
    if mdd == 0:
        return np.inf
    return returns/mdd