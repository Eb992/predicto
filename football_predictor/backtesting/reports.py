"""Generazione report di backtesting."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from .metrics import PerformanceMetrics, calculate_all_metrics


class BacktestReport: 
    """Genera report dettagliati del backtesting."""
    
    def __init__(self, results, bets:  List[Dict], config: Dict = None):
        self.results = results
        self.bets = bets
        self.config = config or {}
    
    def generate_summary(self) -> str:
        """Genera un riepilogo testuale."""
        r = self.results
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST REPORT                           ║
╠══════════════════════════════════════════════════════════════╣
║  Periodo:      {r.start_date} → {r.end_date}
║  
║  PERFORMANCE
║  ───────────────────────────────────────────────────────────
║  Bankroll Iniziale:     €{r.initial_bankroll:,.2f}
║  Bankroll Finale:      €{r.final_bankroll:,.2f}
║  Profitto/Perdita:     €{r.total_profit:+,. 2f}
║  ROI:                  {r.roi:+.2%}
║  
║  SCOMMESSE
║  ───────────────────────────────────────────────────────────
║  Totale Scommesse:      {r.total_bets}
║  Vinte:                {r. wins}
║  Perse:                {r. losses}
║  Win Rate:             {r.win_rate:. 2%}
║  
║  RISCHIO
║  ───────────────────────────────────────────────────────────
║  Max Drawdown:         {r.max_drawdown:. 2%}
║  Sharpe Ratio:         {r.sharpe_ratio:.2f}
║  
║  ACCURACY MODELLI
║  ───────────────────────────────────────────────────────────
"""
        for model, acc in r.model_accuracies.items():
            status = "✓" if acc >= 0.60 else "✗"
            summary += f"║  {model: 25} {acc:.2%} {status}\n"
        
        summary += """╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    def generate_html_report(self, output_path: str):
        """Genera un report HTML completo."""
        r = self.results
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family:  Arial, sans-serif; margin: 40px; background:  #f5f5f5; }}
        .container {{ max-width: 1200px; margin:  0 auto; background: white; padding:  30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color:  #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        . metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius:  10px; text-align: center; }}
        .metric-card. positive {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .metric-card.negative {{ background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }}
        .metric-value {{ font-size:  28px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; margin-top: 5px; }}
        table {{ width: 100%; border-collapse:  collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .status-ok {{ color: #27ae60; font-weight: bold; }}
        .status-warning {{ color: #e74c3c; font-weight: bold; }}
        . chart {{ height: 300px; margin: 20px 0; background: #f8f9fa; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Football Prediction - Backtest Report</h1>
        <p>Periodo:  <strong>{r.start_date}</strong> → <strong>{r.end_date}</strong></p>
        
        <h2>📊 Metriche Principali</h2>
        <div class="metrics">
            <div class="metric-card {'positive' if r. total_profit > 0 else 'negative'}">
                <div class="metric-value">€{r.total_profit:+,.2f}</div>
                <div class="metric-label">Profitto/Perdita</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{r.roi:+.2%}</div>
                <div class="metric-label">ROI</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{r. win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{r.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        </div>
        
        <h2>🎯 Performance Scommesse</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{r.total_bets}</div>
                <div class="metric-label">Totale Scommesse</div>
            </div>
            <div class="metric-card positive">
                <div class="metric-value">{r.wins}</div>
                <div class="metric-label">Vinte</div>
            </div>
            <div class="metric-card negative">
                <div class="metric-value">{r.losses}</div>
                <div class="metric-label">Perse</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{r.max_drawdown:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>
        
        <h2>🤖 Accuracy Modelli</h2>
        <table>
            <tr>
                <th>Modello</th>
                <th>Accuracy</th>
                <th>Soglia (60%)</th>
                <th>Status</th>
            </tr>
            {"". join(f'''
            <tr>
                <td>{model}</td>
                <td>{acc:. 2%}</td>
                <td>60%</td>
                <td class="{'status-ok' if acc >= 0.60 else 'status-warning'}">{'✓ OK' if acc >= 0.60 else '✗ Sotto soglia'}</td>
            </tr>
            ''' for model, acc in r.model_accuracies.items())}
        </table>
        
        <h2>💰 Equity Curve</h2>
        <div class="chart" id="equity-chart">
            <!-- In produzione, usare una libreria come Chart. js o Plotly -->
            <p style="text-align:  center; padding-top: 130px; color: #888;">
                Bankroll: €{r.initial_bankroll:,.2f} → €{r.final_bankroll:,. 2f}
            </p>
        </div>
        
        <h2>📈 Rendimenti Giornalieri</h2>
        <p>Media: {np.mean(r. daily_returns) if r.daily_returns else 0:. 4%} | 
           Std Dev: {np.std(r.daily_returns) if r.daily_returns else 0:.4%} |
           Giorni Positivi: {sum(1 for d in r.daily_returns if d > 0)}/{len(r.daily_returns)}</p>
        
        <hr>
        <p style="text-align:  center; color: #888; font-size: 12px;">
            Report generato il {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
            Football Prediction System
        </p>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def export_to_excel(self, output_path: str):
        """Esporta i dati in Excel con multiple sheets."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary
            summary_data = {
                'Metrica': ['Bankroll Iniziale', 'Bankroll Finale', 'Profitto', 'ROI',
                           'Totale Scommesse', 'Vinte', 'Perse', 'Win Rate',
                           'Max Drawdown', 'Sharpe Ratio'],
                'Valore': [
                    f"€{self.results.initial_bankroll:,.2f}",
                    f"€{self. results.final_bankroll:,.2f}",
                    f"€{self.results.total_profit:+,.2f}",
                    f"{self.results.roi:. 2%}",
                    self.results.total_bets,
                    self.results. wins,
                    self.results.losses,
                    f"{self.results. win_rate:.2%}",
                    f"{self. results.max_drawdown:.2%}",
                    f"{self.results. sharpe_ratio:. 2f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Bets
            if self.bets:
                pd.DataFrame(self. bets).to_excel(writer, sheet_name='Bets', index=False)
            
            # Equity Curve
            equity_df = pd.DataFrame({
                'Period': range(len(self. results.bankroll_history)),
                'Bankroll': self.results. bankroll_history
            })
            equity_df.to_excel(writer, sheet_name='Equity', index=False)
            
            # Model Accuracy
            acc_df = pd. DataFrame([
                {'Model': k, 'Accuracy': v, 'Above Threshold': v >= 0.60}
                for k, v in self. results.model_accuracies.items()
            ])
            acc_df.to_excel(writer, sheet_name='Models', index=False)
    
    def save_json(self, output_path: str):
        """Salva il report in formato JSON."""
        data = {
            'generated_at': datetime. now().isoformat(),
            'config': self.config,
            'results':  {
                'period': {
                    'start': str(self.results. start_date),
                    'end':  str(self.results.end_date)
                },
                'performance': {
                    'initial_bankroll': self.results.initial_bankroll,
                    'final_bankroll': self.results.final_bankroll,
                    'total_profit': self.results.total_profit,
                    'roi': self. results.roi
                },
                'bets': {
                    'total':  self.results.total_bets,
                    'wins':  self.results.wins,
                    'losses': self.results.losses,
                    'win_rate': self. results.win_rate
                },
                'risk': {
                    'max_drawdown': self.results. max_drawdown,
                    'sharpe_ratio': self.results.sharpe_ratio
                },
                'models': self.results.model_accuracies
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)