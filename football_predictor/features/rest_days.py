"""Calcolo giorni di riposo e fattore fatica."""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta


class RestDaysCalculator:
    """Calcola i giorni di riposo e il fattore fatica."""
    
    # Pesi per tipo di competizione (più importante = più stancante)
    COMPETITION_WEIGHTS = {
        'Champions League': 1.3,
        'Europa League': 1.2,
        'Conference League': 1.1,
        'FA Cup': 1.0,
        'Coppa Italia': 1.0,
        'League Cup': 0.9,
        'League':  1.0
    }
    
    def __init__(self, matches: pd.DataFrame):
        self.matches = matches. sort_values('date')
    
    def get_rest_days(self, team: str, match_date: datetime) -> int:
        """Calcola i giorni dalla partita precedente."""
        team_matches = self. matches[
            ((self.matches['home_team'] == team) | (self.matches['away_team'] == team)) &
            (self.matches['date'] < match_date)
        ].sort_values('date', ascending=False)
        
        if len(team_matches) == 0:
            return 7  # Default una settimana
        
        last_match = team_matches.iloc[0]['date']
        if isinstance(last_match, str):
            last_match = pd.to_datetime(last_match)
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)
        
        return (match_date - last_match).days
    
    def get_fatigue_score(self, team:  str, match_date: datetime,
                          window_days: int = 14) -> Dict[str, float]: 
        """
        Calcola un punteggio di fatica basato sulle partite recenti.
        
        Considera:
        - Numero di partite
        - Giorni di riposo
        - Importanza delle competizioni
        - Viaggi (future:  distanza trasferte)
        """
        window_start = match_date - timedelta(days=window_days)
        
        team_matches = self. matches[
            ((self.matches['home_team'] == team) | (self.matches['away_team'] == team)) &
            (self. matches['date'] >= window_start) &
            (self. matches['date'] < match_date)
        ]
        
        if len(team_matches) == 0:
            return {
                'fatigue_score': 0.0,
                'matches_in_window': 0,
                'avg_rest_in_window': 7.0,
                'rest_days_last_match': 7
            }
        
        # Numero partite nel periodo
        n_matches = len(team_matches)
        
        # Media giorni riposo
        dates = sorted(team_matches['date'].tolist())
        rest_days_list = []
        for i in range(1, len(dates)):
            d1 = pd.to_datetime(dates[i-1])
            d2 = pd.to_datetime(dates[i])
            rest_days_list. append((d2 - d1).days)
        
        avg_rest = sum(rest_days_list) / len(rest_days_list) if rest_days_list else 7
        
        # Giorni dall'ultima partita
        last_match_rest = self.get_rest_days(team, match_date)
        
        # Score di fatica (più alto = più stanco)
        # Formula: più partite e meno riposo = più fatica
        base_fatigue = n_matches / (window_days / 3.5)  # Normalizza su ~4 partite in 2 settimane
        rest_factor = max(0, (4 - avg_rest) / 4)  # Penalizza se media < 4 giorni
        
        fatigue_score = min(1.0, base_fatigue * (1 + rest_factor * 0.5))
        
        return {
            'fatigue_score':  fatigue_score,
            'matches_in_window': n_matches,
            'avg_rest_in_window': avg_rest,
            'rest_days_last_match': last_match_rest
        }
    
    def get_rest_advantage(self, home_team: str, away_team: str,
                           match_date: datetime) -> Dict[str, float]: 
        """Calcola il vantaggio di riposo tra due squadre."""
        home_rest = self. get_rest_days(home_team, match_date)
        away_rest = self.get_rest_days(away_team, match_date)
        
        home_fatigue = self.get_fatigue_score(home_team, match_date)
        away_fatigue = self.get_fatigue_score(away_team, match_date)
        
        return {
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'rest_days_diff': home_rest - away_rest,
            'home_fatigue': home_fatigue['fatigue_score'],
            'away_fatigue': away_fatigue['fatigue_score'],
            'fatigue_diff': away_fatigue['fatigue_score'] - home_fatigue['fatigue_score']
        }