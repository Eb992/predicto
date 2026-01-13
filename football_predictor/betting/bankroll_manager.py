"""Gestione avanzata del bankroll."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .kelly_criterion import KellyCriterion, Bet

logger = logging.getLogger(__name__)


@dataclass
class BankrollState:
    """Stato corrente del bankroll."""
    current_value: float
    initial_value: float
    peak_value: float
    drawdown: float
    daily_pnl: float
    weekly_pnl:  float
    monthly_pnl: float


class BankrollManager:
    """
    Gestione avanzata del bankroll con: 
    - Stop loss / Profit targets
    - Limiti giornalieri
    - Gestione esposizione
    - Tracking performance
    """
    
    def __init__(self,
                 initial_bankroll: float = 1000.0,
                 kelly_fraction: float = 0.25,
                 stop_loss_pct: float = 0.50,
                 profit_target_pct: float = 1.00,
                 daily_loss_limit_pct: float = 0.10,
                 max_exposure_pct: float = 0.20):
        """
        Inizializza il BankrollManager. 
        
        Args: 
            initial_bankroll: Capitale iniziale
            kelly_fraction: Frazione Kelly da usare
            stop_loss_pct:  Stop loss (es. 0.50 = stop se perdi 50%)
            profit_target_pct: Profit target (es. 1.00 = raddoppio)
            daily_loss_limit_pct: Limite perdita giornaliera
            max_exposure_pct:  Esposizione massima simultanea
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        
        self.kelly = KellyCriterion(
            bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction
        )
        
        self.stop_loss_pct = stop_loss_pct
        self. profit_target_pct = profit_target_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_exposure_pct = max_exposure_pct
        
        self.daily_start_bankroll = initial_bankroll
        self.daily_pnl = 0.0
        self.pending_bets: List[Bet] = []
        self. history: List[Dict] = []
    
    def get_state(self) -> BankrollState: 
        """Ritorna lo stato corrente del bankroll."""
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        
        return BankrollState(
            current_value=self. current_bankroll,
            initial_value=self.initial_bankroll,
            peak_value=self.peak_bankroll,
            drawdown=drawdown,
            daily_pnl=self.daily_pnl,
            weekly_pnl=self._calculate_period_pnl(7),
            monthly_pnl=self._calculate_period_pnl(30)
        )
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """
        Verifica se dovremmo fermare le operazioni.
        
        Returns: 
            (should_stop, reason)
        """
        # Stop loss
        loss_pct = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        if loss_pct >= self.stop_loss_pct: 
            return True, f"STOP_LOSS:  Perdita {loss_pct:. 1%} >= {self.stop_loss_pct:. 1%}"
        
        # Profit target
        profit_pct = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        if profit_pct >= self.profit_target_pct: 
            return True, f"PROFIT_TARGET: Profitto {profit_pct:. 1%} >= {self.profit_target_pct:.1%}"
        
        # Daily loss limit
        daily_loss_pct = -self.daily_pnl / self. daily_start_bankroll if self.daily_pnl < 0 else 0
        if daily_loss_pct >= self.daily_loss_limit_pct:
            return True, f"DAILY_LIMIT: Perdita giornaliera {daily_loss_pct:.1%} >= {self.daily_loss_limit_pct:.1%}"
        
        return False, ""
    
    def can_place_bet(self, bet:  Bet) -> Tuple[bool, str]:
        """
        Verifica se è possibile piazzare una scommessa.
        
        Args: 
            bet: La scommessa da verificare
        
        Returns:
            (can_place, reason)
        """
        # Controlla stop conditions
        should_stop, reason = self.should_stop_trading()
        if should_stop: 
            return False, reason
        
        # Controlla esposizione
        current_exposure = sum(b.stake for b in self.pending_bets)
        if (current_exposure + bet.stake) > self.current_bankroll * self.max_exposure_pct:
            return False, f"MAX_EXPOSURE:  Esposizione supererebbe {self.max_exposure_pct:.1%}"
        
        # Controlla stake vs bankroll
        if bet.stake > self. current_bankroll: 
            return False, "INSUFFICIENT_FUNDS:  Stake superiore al bankroll"
        
        return True, ""
    
    def calculate_optimal_stakes(self, bets: List[Bet]) -> List[Bet]:
        """
        Calcola gli stake ottimali per una lista di scommesse. 
        
        Considera correlazioni e limita l'esposizione totale.
        
        Args:
            bets: Lista di scommesse
        
        Returns: 
            Lista di scommesse con stake aggiornati
        """
        # Filtra solo value bets
        value_bets = [b for b in bets if b.is_value_bet and b.stake > 0]
        
        if not value_bets:
            return bets
        
        # Calcola stake totale
        total_stake = sum(b.stake for b in value_bets)
        max_exposure = self.current_bankroll * self.max_exposure_pct
        
        # Scala se necessario
        if total_stake > max_exposure:
            scale_factor = max_exposure / total_stake
            for bet in value_bets:
                bet.stake *= scale_factor
            logger.info(f"Stakes scalati di {scale_factor:. 2%} per rispettare esposizione massima")
        
        # Diversificazione:  limita stake per singola partita
        match_stakes = {}
        for bet in value_bets: 
            if bet. match_id not in match_stakes: 
                match_stakes[bet.match_id] = []
            match_stakes[bet.match_id].append(bet)
        
        # Max 2 bet per partita
        for match_id, match_bets in match_stakes.items():
            if len(match_bets) > 2:
                # Tieni solo le 2 con edge maggiore
                sorted_bets = sorted(match_bets, key=lambda x: x.expected_value, reverse=True)
                for bet in sorted_bets[2:]:
                    bet.stake = 0
                    bet.is_value_bet = False
        
        return bets
    
    def place_bet(self, bet:  Bet) -> bool:
        """
        Piazza una scommessa. 
        
        Args: 
            bet: La scommessa da piazzare
        
        Returns:
            True se piazzata con successo
        """
        can_place, reason = self.can_place_bet(bet)
        
        if not can_place: 
            logger.warning(f"Scommessa rifiutata: {reason}")
            return False
        
        self.pending_bets.append(bet)
        logger.info(f"Scommessa piazzata: {bet.match_id} - {bet.market} - €{bet.stake:. 2f}")
        return True
    
    def settle_bet(self, bet:  Bet, won: bool):
        """
        Chiude una scommessa e aggiorna il bankroll.
        
        Args:
            bet: La scommessa da chiudere
            won: Se la scommessa è stata vinta
        """
        if bet in self.pending_bets:
            self.pending_bets.remove(bet)
        
        old_bankroll = self.current_bankroll
        
        if won: 
            profit = bet.stake * (bet.odds - 1)
            self.current_bankroll += profit
        else:
            self.current_bankroll -= bet.stake
        
        pnl = self. current_bankroll - old_bankroll
        self.daily_pnl += pnl
        
        # Aggiorna peak
        if self.current_bankroll > self.peak_bankroll: 
            self.peak_bankroll = self. current_bankroll
        
        # Aggiorna Kelly
        self.kelly.current_bankroll = self.current_bankroll
        self.kelly.place_bet(bet, won)
        
        # Salva in history
        self. history.append({
            'timestamp': datetime.now().isoformat(),
            'match_id': bet. match_id,
            'market': bet.market,
            'selection': bet.selection,
            'stake':  bet.stake,
            'odds': bet.odds,
            'won':  won,
            'pnl': pnl,
            'bankroll_after': self.current_bankroll
        })
        
        logger.info(f"Scommessa chiusa:  {'VINTA' if won else 'PERSA'} - PnL: €{pnl: +.2f}")
    
    def new_day(self):
        """Inizia un nuovo giorno di trading."""
        self.daily_start_bankroll = self.current_bankroll
        self.daily_pnl = 0.0
    
    def _calculate_period_pnl(self, days: int) -> float:
        """Calcola il PnL per un periodo."""
        if not self.history:
            return 0.0
        
        cutoff = datetime.now() - timedelta(days=days)
        period_history = [
            h for h in self.history
            if datetime.fromisoformat(h['timestamp']) >= cutoff
        ]
        
        return sum(h['pnl'] for h in period_history)
    
    def get_statistics(self) -> Dict: 
        """Ritorna statistiche complete."""
        state = self.get_state()
        kelly_stats = self.kelly. get_statistics()
        
        return {
            **kelly_stats,
            'current_bankroll': state.current_value,
            'peak_bankroll': state.peak_value,
            'drawdown': state.drawdown,
            'daily_pnl':  state.daily_pnl,
            'weekly_pnl': state.weekly_pnl,
            'monthly_pnl': state.monthly_pnl,
            'pending_bets': len(self.pending_bets),
            'pending_exposure': sum(b. stake for b in self.pending_bets)
        }
    
    def reset(self):
        """Resetta il manager."""
        self. current_bankroll = self.initial_bankroll
        self. peak_bankroll = self.initial_bankroll
        self. daily_start_bankroll = self.initial_bankroll
        self.daily_pnl = 0.0
        self.pending_bets = []
        self. history = []
        self.kelly.reset()