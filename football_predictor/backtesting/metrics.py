"""Metriche di performance per il backtesting."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container per tutte le metriche di performance."""
    total_return: float
    roi: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    profit_factor: float
    avg_win:  float
    avg_loss: float
    win_loss_ratio:  float
    total_bets: int
    wins: int
    losses: int
    expected_value: float
    kelly_fraction: float


def calculate_returns(pnl_series: pd.Series) -> pd.Series:
    """Calcola i rendimenti giornalieri."""
    return pnl_series.pct_change().fillna(0)


def calculate_sharpe_ratio(returns:  pd.Series, 
                           risk_free_rate: float = 0.02,
                           periods_per_year: int = 365) -> float:
    """
    Calcola lo Sharpe Ratio annualizzato.
    
    Sharpe = (Return - Risk Free) / Std Dev
    """
    if returns.std() == 0:
        return 0. 0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np. sqrt(periods_per_year) * excess_returns. mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series,
                            risk_free_rate: float = 0.02,
                            periods_per_year: int = 365) -> float:
    """
    Calcola il Sortino Ratio (usa solo la volatilità negativa).
    """
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0 or negative_returns. std() == 0:
        return 0.0
    
    excess_returns = returns. mean() - risk_free_rate / periods_per_year
    downside_std = negative_returns. std()
    
    return np.sqrt(periods_per_year) * excess_returns / downside_std


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
    """
    Calcola il Maximum Drawdown e la sua durata.
    
    Returns: 
        (max_drawdown_pct, duration_in_periods)
    """
    peak = equity_curve. expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    
    max_dd = drawdown.min()
    
    # Calcola durata
    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return 0.0, 0
    
    # Trova la durata del drawdown più lungo
    drawdown_starts = (~in_drawdown).cumsum()
    drawdown_groups = in_drawdown.groupby(drawdown_starts)
    
    max_duration = 0
    for _, group in drawdown_groups:
        if group.any():
            max_duration = max(max_duration, len(group))
    
    return abs(max_dd), max_duration


def calculate_calmar_ratio(total_return: float, 
                           max_drawdown: float,
                           years: float = 1.0) -> float:
    """
    Calcola il Calmar Ratio. 
    
    Calmar = Annual Return / Max Drawdown
    """
    if max_drawdown == 0:
        return 0.0
    
    annual_return = total_return / years
    return annual_return / max_drawdown


def calculate_profit_factor(wins: List[float], losses: List[float]) -> float:
    """
    Calcola il Profit Factor. 
    
    Profit Factor = Gross Profit / Gross Loss
    """
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expected_value(win_rate: float, 
                             avg_win: float, 
                             avg_loss: float) -> float:
    """
    Calcola il valore atteso per scommessa.
    
    EV = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    """
    return (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))


def calculate_optimal_kelly(win_rate: float, avg_odds: float) -> float:
    """
    Calcola la frazione di Kelly ottimale.
    
    Kelly = (bp - q) / b
    dove b = odds - 1, p = win_rate, q = 1 - p
    """
    b = avg_odds - 1
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly)


def calculate_all_metrics(pnl_list: List[float],
                          bankroll_history: List[float],
                          bets:  List[Dict]) -> PerformanceMetrics: 
    """Calcola tutte le metriche di performance."""
    
    if not bets:
        return PerformanceMetrics(
            total_return=0, roi=0, win_rate=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, max_drawdown_duration=0,
            calmar_ratio=0, profit_factor=0, avg_win=0, avg_loss=0,
            win_loss_ratio=0, total_bets=0, wins=0, losses=0,
            expected_value=0, kelly_fraction=0
        )
    
    # Separa vincite e perdite
    wins_pnl = [b['pnl'] for b in bets if b. get('pnl', 0) > 0]
    losses_pnl = [b['pnl'] for b in bets if b.get('pnl', 0) < 0]
    
    wins = len(wins_pnl)
    losses = len(losses_pnl)
    total_bets = wins + losses
    
    win_rate = wins / total_bets if total_bets > 0 else 0
    avg_win = np.mean(wins_pnl) if wins_pnl else 0
    avg_loss = np.mean(losses_pnl) if losses_pnl else 0
    
    # Equity curve
    equity = pd.Series(bankroll_history)
    returns = calculate_returns(equity)
    
    # Metriche
    initial_bankroll = bankroll_history[0] if bankroll_history else 1000
    final_bankroll = bankroll_history[-1] if bankroll_history else 1000
    total_return = (final_bankroll - initial_bankroll) / initial_bankroll
    
    total_staked = sum(b. get('stake', 0) for b in bets)
    roi = sum(pnl_list) / total_staked if total_staked > 0 else 0
    
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, max_dd_duration = calculate_max_drawdown(equity)
    calmar = calculate_calmar_ratio(total_return, max_dd)
    profit_factor = calculate_profit_factor(wins_pnl, losses_pnl)
    ev = calculate_expected_value(win_rate, avg_win, abs(avg_loss))
    
    avg_odds = np.mean([b.get('odds', 2. 0) for b in bets]) if bets else 2.0
    kelly = calculate_optimal_kelly(win_rate, avg_odds)
    
    return PerformanceMetrics(
        total_return=total_return,
        roi=roi,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=calmar,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        expected_value=ev,
        kelly_fraction=kelly
    )