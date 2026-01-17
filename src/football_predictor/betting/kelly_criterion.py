"""Implementazione del Kelly Criterion per la gestione del bankroll."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Bet:
    """Rappresenta una scommessa."""

    match_id: str
    market: str  # es:  '1X2', 'BTTS', 'Over2. 5'
    selection: str  # es: 'H', 'D', 'A', 'Yes', 'Over'
    probability: float  # Probabilità stimata dal modello
    odds: float  # Quote offerte dal bookmaker
    stake: float = 0.0
    is_value_bet: bool = False
    expected_value: float = 0.0
    kelly_fraction: float = 0.0


class KellyCriterion:
    """Implementazione del Kelly Criterion per il betting."""

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,  # Kelly frazionato
        max_stake_pct: float = 0.05,  # Max 5% per scommessa
        min_edge: float = 0.05,
    ):  # Edge minimo 5%
        """
        Inizializza il Kelly Criterion.

        Args:
            bankroll: Capitale iniziale
            kelly_fraction:  Frazione del Kelly da usare (0.25 = quarter Kelly)
            max_stake_pct:  Percentuale massima del bankroll per singola scommessa
            min_edge:  Edge minimo richiesto per considerare una value bet
        """
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.min_edge = min_edge
        self.bet_history: List[Dict] = []

    def calculate_edge(self, probability: float, odds: float) -> float:
        """
        Calcola l'edge (vantaggio) su una scommessa.

        Edge = (p * odds) - 1
        Dove p è la probabilità stimata e odds le quote decimali.
        """
        return (probability * odds) - 1

    def calculate_implied_probability(self, odds: float) -> float:
        """Calcola la probabilità implicita dalle quote."""
        return 1 / odds

    def calculate_kelly_stake(self, probability: float, odds: float) -> float:
        """
        Calcola lo stake ottimale secondo Kelly.

        Formula Kelly: f* = (bp - q) / b
        Dove:
            b = odds - 1 (profitto netto per unità)
            p = probabilità di vincita
            q = 1 - p (probabilità di perdita)

        Returns:
            Frazione del bankroll da scommettere (0 se non value bet)
        """
        b = odds - 1
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b

        # Se Kelly negativo, non è una value bet
        if kelly <= 0:
            return 0.0

        # Applica Kelly frazionato
        kelly_fractional = kelly * self.kelly_fraction

        # Limita allo stake massimo
        kelly_capped = min(kelly_fractional, self.max_stake_pct)

        return kelly_capped

    def evaluate_bet(
        self, probability: float, odds: float, match_id: str, market: str, selection: str
    ) -> Bet:
        """
        Valuta se una scommessa è una value bet e calcola lo stake.

        Args:
            probability: Probabilità stimata dal modello
            odds: Quote del bookmaker
            match_id: ID della partita
            market: Tipo di mercato
            selection: Selezione effettuata

        Returns:
            Oggetto Bet con tutti i dettagli
        """
        edge = self.calculate_edge(probability, odds)
        kelly_fraction = self.calculate_kelly_stake(probability, odds)
        stake_amount = kelly_fraction * self.current_bankroll

        is_value = edge >= self.min_edge and kelly_fraction > 0

        return Bet(
            match_id=match_id,
            market=market,
            selection=selection,
            probability=probability,
            odds=odds,
            stake=stake_amount if is_value else 0.0,
            is_value_bet=is_value,
            expected_value=edge,
            kelly_fraction=kelly_fraction,
        )

    def place_bet(self, bet: Bet, won: bool) -> float:
        """
        Registra il risultato di una scommessa e aggiorna il bankroll.

        Args:
            bet: La scommessa piazzata
            won: Se la scommessa è stata vinta

        Returns:
            Nuovo bankroll
        """
        if not bet.is_value_bet or bet.stake == 0:
            return self.current_bankroll

        if won:
            profit = bet.stake * (bet.odds - 1)
            self.current_bankroll += profit
        else:
            self.current_bankroll -= bet.stake

        self.bet_history.append(
            {
                "match_id": bet.match_id,
                "market": bet.market,
                "selection": bet.selection,
                "stake": bet.stake,
                "odds": bet.odds,
                "won": won,
                "bankroll_after": self.current_bankroll,
            }
        )

        return self.current_bankroll

    def get_statistics(self) -> Dict:
        """Ritorna statistiche del betting."""
        if not self.bet_history:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "current_bankroll": self.current_bankroll,
            }

        wins = sum(1 for b in self.bet_history if b["won"])
        losses = len(self.bet_history) - wins
        total_staked = sum(b["stake"] for b in self.bet_history)
        profit = self.current_bankroll - self.initial_bankroll

        return {
            "total_bets": len(self.bet_history),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self.bet_history) if self.bet_history else 0,
            "total_staked": total_staked,
            "total_profit": profit,
            "roi": profit / total_staked if total_staked > 0 else 0,
            "current_bankroll": self.current_bankroll,
            "bankroll_growth": (self.current_bankroll / self.initial_bankroll - 1) * 100,
        }

    def reset(self):
        """Resetta il bankroll e la storia."""
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []


class BankrollManager:
    """Gestione avanzata del bankroll con multiple strategie."""

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.kelly = KellyCriterion(bankroll=initial_bankroll)

        # Limiti di sicurezza
        self.stop_loss_pct = 0.50  # Stop se perdi il 50%
        self.profit_target_pct = 1.00  # Target raddoppio
        self.daily_loss_limit_pct = 0.10  # Max 10% perdita giornaliera

    def should_stop(self) -> Tuple[bool, str]:
        """Verifica se dovremmo fermarci (stop loss/profit target)."""
        if self.current_bankroll <= self.initial_bankroll * (1 - self.stop_loss_pct):
            return True, "STOP_LOSS"
        if self.current_bankroll >= self.initial_bankroll * (1 + self.profit_target_pct):
            return True, "PROFIT_TARGET"
        return False, ""

    def calculate_optimal_stakes(self, bets: List[Bet]) -> List[Bet]:
        """
        Calcola stakes ottimali per una lista di scommesse simultanee.
        Considera correlazioni e limita l'esposizione totale.
        """
        # Filtra solo value bets
        value_bets = [b for b in bets if b.is_value_bet]

        if not value_bets:
            return bets

        # Limita esposizione totale al 20% del bankroll
        max_total_exposure = self.current_bankroll * 0.20
        total_kelly_stake = sum(b.stake for b in value_bets)

        if total_kelly_stake > max_total_exposure:
            # Scala proporzionalmente
            scale_factor = max_total_exposure / total_kelly_stake
            for bet in value_bets:
                bet.stake *= scale_factor

        return bets
