"""Tracking delle scommesse e performance."""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging. getLogger(__name__)


@dataclass
class BetRecord:
    """Record di una scommessa."""
    id: str
    date: str
    match_id: str
    home_team:  str
    away_team: str
    league: str
    market: str
    selection: str
    odds: float
    stake: float
    model_probability: float
    edge: float
    status: str  # 'pending', 'won', 'lost', 'void'
    result: Optional[str] = None
    profit_loss: Optional[float] = None
    bankroll_after:  Optional[float] = None
    notes: Optional[str] = None


class BetTracker: 
    """Traccia le scommesse e calcola le performance."""
    
    def __init__(self, storage_path: str = "data/bet_history.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.bets:  List[BetRecord] = []
        self._load()
    
    def _load(self):
        """Carica lo storico dal file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json. load(f)
                    self.bets = [BetRecord(**b) for b in data]
            except Exception as e:
                logger.warning(f"Errore caricamento storico: {e}")
                self.bets = []
    
    def _save(self):
        """Salva lo storico su file."""
        with open(self.storage_path, 'w') as f:
            json.dump([asdict(b) for b in self.bets], f, indent=2, default=str)
    
    def add_bet(self, bet: BetRecord):
        """Aggiunge una nuova scommessa."""
        bet. id = f"{bet. date}_{bet. match_id}_{bet.market}_{len(self.bets)}"
        self.bets.append(bet)
        self._save()
        logger.info(f"Scommessa aggiunta: {bet.id}")
    
    def update_bet_result(self, bet_id: str, status: str, 
                          result: str, bankroll_after: float):
        """Aggiorna il risultato di una scommessa."""
        for bet in self.bets:
            if bet.id == bet_id:
                bet.status = status
                bet.result = result
                bet. bankroll_after = bankroll_after
                
                if status == 'won':
                    bet.profit_loss = bet.stake * (bet.odds - 1)
                elif status == 'lost':
                    bet.profit_loss = -bet.stake
                else:
                    bet.profit_loss = 0
                
                self._save()
                logger.info(f"Scommessa {bet_id} aggiornata:  {status}")
                return True
        
        return False
    
    def get_pending_bets(self) -> List[BetRecord]:
        """Ritorna le scommesse in attesa."""
        return [b for b in self. bets if b. status == 'pending']
    
    def get_bets_by_date(self, date:  str) -> List[BetRecord]: 
        """Ritorna le scommesse per una data specifica."""
        return [b for b in self.bets if b.date == date]
    
    def get_performance_summary(self, 
                                start_date: str = None,
                                end_date: str = None,
                                market:  str = None) -> Dict:
        """Calcola le statistiche di performance."""
        filtered = self.bets. copy()
        
        if start_date:
            filtered = [b for b in filtered if b.date >= start_date]
        if end_date:
            filtered = [b for b in filtered if b. date <= end_date]
        if market:
            filtered = [b for b in filtered if b.market == market]
        
        # Solo scommesse con risultato
        settled = [b for b in filtered if b. status in ['won', 'lost']]
        
        if not settled:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_staked': 0,
                'total_profit':  0,
                'roi': 0,
                'avg_odds': 0,
                'avg_edge': 0
            }
        
        wins = len([b for b in settled if b.status == 'won'])
        losses = len(settled) - wins
        total_staked = sum(b.stake for b in settled)
        total_profit = sum(b.profit_loss or 0 for b in settled)
        
        return {
            'total_bets': len(settled),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(settled) if settled else 0,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': total_profit / total_staked if total_staked > 0 else 0,
            'avg_odds': sum(b.odds for b in settled) / len(settled),
            'avg_edge': sum(b.edge for b in settled) / len(settled),
            'best_bet':  max(settled, key=lambda x: x.profit_loss or 0),
            'worst_bet': min(settled, key=lambda x: x.profit_loss or 0)
        }
    
    def get_performance_by_market(self) -> Dict[str, Dict]: 
        """Ritorna performance separate per mercato."""
        markets = set(b.market for b in self.bets)
        return {market: self.get_performance_summary(market=market) 
                for market in markets}
    
    def get_streak_info(self) -> Dict:
        """Calcola informazioni sulle serie di risultati."""
        settled = [b for b in self.bets if b.status in ['won', 'lost']]
        settled. sort(key=lambda x: x.date)
        
        if not settled:
            return {'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0}
        
        # Streak corrente
        current_streak = 0
        for bet in reversed(settled):
            if current_streak == 0:
                current_streak = 1 if bet.status == 'won' else -1
            elif (current_streak > 0 and bet.status == 'won') or \
                 (current_streak < 0 and bet.status == 'lost'):
                current_streak += 1 if current_streak > 0 else -1
            else:
                break
        
        # Max streaks
        max_win, max_loss = 0, 0
        win_streak, loss_streak = 0, 0
        
        for bet in settled:
            if bet.status == 'won': 
                win_streak += 1
                loss_streak = 0
                max_win = max(max_win, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss = max(max_loss, loss_streak)
        
        return {
            'current_streak': current_streak,
            'max_win_streak': max_win,
            'max_loss_streak':  max_loss
        }
    
    def export_to_csv(self, path: str):
        """Esporta lo storico in CSV."""
        df = pd.DataFrame([asdict(b) for b in self. bets])
        df.to_csv(path, index=False)
        logger.info(f"Storico esportato in {path}")
    
    def get_monthly_report(self) -> pd.DataFrame:
        """Genera report mensile."""
        settled = [b for b in self.bets if b.status in ['won', 'lost']]
        
        if not settled: 
            return pd. DataFrame()
        
        df = pd. DataFrame([asdict(b) for b in settled])
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        monthly = df.groupby('month').agg({
            'stake': 'sum',
            'profit_loss': 'sum',
            'id': 'count',
            'status':  lambda x: (x == 'won').sum()
        }).rename(columns={'id': 'total_bets', 'status': 'wins'})
        
        monthly['win_rate'] = monthly['wins'] / monthly['total_bets']
        monthly['roi'] = monthly['profit_loss'] / monthly['stake']
        
        return monthly