"""Identificazione delle value bets."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from . kelly_criterion import KellyCriterion, Bet


@dataclass
class ValueBet:
    """Rappresenta una value bet identificata."""
    match_id: str
    date: str
    home_team: str
    away_team: str
    league: str
    market:  str
    selection:  str
    model_probability: float
    bookmaker_probability: float
    odds:  float
    edge: float
    kelly_stake: float
    recommended_stake: float
    confidence: str
    
    @property
    def expected_value(self) -> float:
        """Valore atteso della scommessa."""
        return self.model_probability * self. odds - 1


class ValueBetFinder:
    """Trova le value bets confrontando probabilità modello vs bookmaker."""
    
    def __init__(self, 
                 kelly:  KellyCriterion,
                 min_edge: float = 0.05,
                 min_probability: float = 0.30,
                 max_odds: float = 10.0):
        """
        Args:
            kelly:  Istanza KellyCriterion per calcolo stakes
            min_edge: Edge minimo richiesto
            min_probability: Probabilità minima modello
            max_odds: Quote massime accettate
        """
        self.kelly = kelly
        self.min_edge = min_edge
        self. min_probability = min_probability
        self.max_odds = max_odds
    
    def find_value_bets(self, 
                        predictions: List[Dict],
                        odds_data: pd.DataFrame = None) -> List[ValueBet]: 
        """
        Trova tutte le value bets dalle predizioni.
        
        Args: 
            predictions: Lista di predizioni per partita
            odds_data: DataFrame con le quote dei bookmaker
        
        Returns:
            Lista di ValueBet ordinate per edge
        """
        value_bets = []
        
        for match_pred in predictions:
            match_id = f"{match_pred['date']}_{match_pred['home_team']}_{match_pred['away_team']}"
            
            for market, market_preds in match_pred.get('predictions', {}).items():
                for pred in market_preds:
                    # Ottieni quote
                    odds = self._get_odds(
                        match_pred, market, pred['selection'], odds_data
                    )
                    
                    if odds is None or odds > self.max_odds or odds < 1.01:
                        continue
                    
                    probability = pred['probability']
                    
                    if probability < self.min_probability:
                        continue
                    
                    # Calcola edge
                    bookmaker_prob = 1 / odds
                    edge = probability - bookmaker_prob
                    
                    if edge < self.min_edge:
                        continue
                    
                    # Calcola stake con Kelly
                    bet = self. kelly.evaluate_bet(
                        probability=probability,
                        odds=odds,
                        match_id=match_id,
                        market=market,
                        selection=pred['selection']
                    )
                    
                    if not bet.is_value_bet: 
                        continue
                    
                    # Determina confidenza
                    if edge >= 0.15 and probability >= 0.60:
                        confidence = 'high'
                    elif edge >= 0.10 or probability >= 0.55:
                        confidence = 'medium'
                    else: 
                        confidence = 'low'
                    
                    value_bets. append(ValueBet(
                        match_id=match_id,
                        date=match_pred['date'],
                        home_team=match_pred['home_team'],
                        away_team=match_pred['away_team'],
                        league=match_pred. get('league', 'N/A'),
                        market=market,
                        selection=pred['selection'],
                        model_probability=probability,
                        bookmaker_probability=bookmaker_prob,
                        odds=odds,
                        edge=edge,
                        kelly_stake=bet.kelly_fraction,
                        recommended_stake=bet. stake,
                        confidence=confidence
                    ))
        
        # Ordina per edge decrescente
        value_bets.sort(key=lambda x: x.edge, reverse=True)
        
        return value_bets
    
    def _get_odds(self, match:  Dict, market: str, selection: str,
                  odds_data: pd.DataFrame = None) -> Optional[float]:
        """Recupera le quote per una selezione."""
        
        # Prima cerca in odds_data se fornito
        if odds_data is not None and len(odds_data) > 0:
            try:
                match_odds = odds_data[
                    (odds_data['home_team'] == match['home_team']) &
                    (odds_data['away_team'] == match['away_team'])
                ]
                
                if len(match_odds) > 0:
                    row = match_odds. iloc[0]
                    odds_col = self._get_odds_column(market, selection)
                    if odds_col and odds_col in row: 
                        return float(row[odds_col])
            except Exception:
                pass
        
        # Fallback: simula quote basate sulla probabilità
        # In produzione, qui si collegherebbe a un'API di quote reali
        prob = match. get('predictions', {}).get(market, [{}])[0].get('probability', 0.5)
        if prob > 0:
            # Aggiungi margine bookmaker ~5%
            return round(1 / prob * 0.95, 2)
        
        return None
    
    def _get_odds_column(self, market: str, selection: str) -> Optional[str]: 
        """Mappa market/selection alla colonna quote nel DataFrame."""
        mappings = {
            ('1x2', 'H'): ['B365H', 'PSH', 'odds_home_win', 'home_odds'],
            ('1x2', 'D'): ['B365D', 'PSD', 'odds_draw', 'draw_odds'],
            ('1x2', 'A'): ['B365A', 'PSA', 'odds_away_win', 'away_odds'],
            ('1X2_Result', 'H'): ['B365H', 'PSH', 'odds_home_win'],
            ('1X2_Result', 'D'): ['B365D', 'PSD', 'odds_draw'],
            ('1X2_Result', 'A'): ['B365A', 'PSA', 'odds_away_win'],
            ('btts', 'True'): ['odds_btts_yes', 'BTS_Yes'],
            ('btts', 'False'): ['odds_btts_no', 'BTS_No'],
            ('over_under', 'True'): ['odds_over_2. 5', 'B365>2.5', 'Over25'],
            ('over_under', 'False'): ['odds_under_2.5', 'B365<2.5', 'Under25'],
        }
        
        key = (market. lower() if market else '', str(selection))
        cols = mappings.get(key, mappings.get((market, str(selection)), []))
        
        return cols[0] if cols else None
    
    def filter_by_confidence(self, value_bets: List[ValueBet],
                             min_confidence: str = 'medium') -> List[ValueBet]:
        """Filtra value bets per livello di confidenza."""
        confidence_order = {'low': 0, 'medium': 1, 'high':  2}
        min_level = confidence_order. get(min_confidence, 1)
        
        return [vb for vb in value_bets 
                if confidence_order. get(vb. confidence, 0) >= min_level]
    
    def get_daily_picks(self, value_bets: List[ValueBet],
                        max_bets: int = 5,
                        max_exposure: float = 0.15) -> List[ValueBet]:
        """
        Seleziona i picks giornalieri ottimali. 
        
        Args:
            value_bets: Lista di value bets
            max_bets:  Numero massimo di scommesse
            max_exposure: Esposizione massima come frazione del bankroll
        """
        # Filtra per confidenza alta/media
        filtered = self.filter_by_confidence(value_bets, 'medium')
        
        # Diversifica per partita (max 1 bet per partita)
        seen_matches = set()
        diversified = []
        
        for vb in filtered:
            if vb.match_id not in seen_matches:
                diversified.append(vb)
                seen_matches.add(vb.match_id)
        
        # Prendi top N per edge
        top_picks = diversified[:max_bets]
        
        # Limita esposizione totale
        total_stake = sum(vb.recommended_stake for vb in top_picks)
        max_stake = self.kelly.current_bankroll * max_exposure
        
        if total_stake > max_stake: 
            scale = max_stake / total_stake
            for vb in top_picks:
                vb.recommended_stake *= scale
        
        return top_picks