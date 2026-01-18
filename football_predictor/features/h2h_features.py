"""Features Head-to-Head."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class H2HFeatureExtractor:
    """Estrae features dagli scontri diretti."""
    
    def __init__(self, matches: pd.DataFrame):
        self.matches = matches. sort_values('date')
    
    def get_h2h_record(self, team1: str, team2: str, 
                       before_date: datetime,
                       n_matches: int = 10) -> Dict[str, float]:
        """Calcola il record negli scontri diretti."""
        h2h_matches = self.matches[
            (((self.matches['home_team'] == team1) & (self.matches['away_team'] == team2)) |
             ((self.matches['home_team'] == team2) & (self.matches['away_team'] == team1))) &
            (self.matches['date'] < before_date)
        ].sort_values('date', ascending=False).head(n_matches)
        
        if len(h2h_matches) == 0:
            return self._default_h2h()
        
        team1_wins, draws, team2_wins = 0, 0, 0
        team1_goals, team2_goals = 0, 0
        btts_count, over_2_5_count = 0, 0
        
        for _, match in h2h_matches.iterrows():
            hg = match.get('home_goals', 0) or 0
            ag = match.get('away_goals', 0) or 0
            
            if match['home_team'] == team1:
                t1_goals, t2_goals = hg, ag
            else:
                t1_goals, t2_goals = ag, hg
            
            team1_goals += t1_goals
            team2_goals += t2_goals
            
            if t1_goals > t2_goals: 
                team1_wins += 1
            elif t1_goals < t2_goals:
                team2_wins += 1
            else:
                draws += 1
            
            if hg > 0 and ag > 0:
                btts_count += 1
            if hg + ag > 2.5:
                over_2_5_count += 1
        
        n = len(h2h_matches)
        return {
            'h2h_matches':  n,
            'h2h_team1_win_rate': team1_wins / n,
            'h2h_draw_rate': draws / n,
            'h2h_team2_win_rate': team2_wins / n,
            'h2h_team1_goals_avg': team1_goals / n,
            'h2h_team2_goals_avg': team2_goals / n,
            'h2h_total_goals_avg': (team1_goals + team2_goals) / n,
            'h2h_btts_rate': btts_count / n,
            'h2h_over_2_5_rate': over_2_5_count / n,
            'h2h_goal_diff':  (team1_goals - team2_goals) / n
        }
    
    def get_h2h_at_venue(self, home_team: str, away_team: str,
                         before_date: datetime,
                         n_matches: int = 5) -> Dict[str, float]:
        """H2H solo per partite con stesso assetto casa/trasferta."""
        venue_matches = self. matches[
            (self.matches['home_team'] == home_team) &
            (self. matches['away_team'] == away_team) &
            (self.matches['date'] < before_date)
        ].sort_values('date', ascending=False).head(n_matches)
        
        if len(venue_matches) == 0:
            return {
                'venue_h2h_matches': 0,
                'venue_home_win_rate': 0.45,
                'venue_away_win_rate': 0.30,
                'venue_draw_rate': 0.25
            }
        
        home_wins = len(venue_matches[venue_matches['home_goals'] > venue_matches['away_goals']])
        away_wins = len(venue_matches[venue_matches['home_goals'] < venue_matches['away_goals']])
        draws = len(venue_matches) - home_wins - away_wins
        
        n = len(venue_matches)
        return {
            'venue_h2h_matches': n,
            'venue_home_win_rate':  home_wins / n,
            'venue_away_win_rate': away_wins / n,
            'venue_draw_rate': draws / n
        }
    
    def get_recent_form_comparison(self, team1: str, team2: str,
                                   before_date: datetime,
                                   n_matches: int = 5) -> Dict[str, float]:
        """Confronta la forma recente delle due squadre."""
        def get_team_form(team):
            team_matches = self. matches[
                ((self.matches['home_team'] == team) | (self.matches['away_team'] == team)) &
                (self. matches['date'] < before_date)
            ].sort_values('date', ascending=False).head(n_matches)
            
            if len(team_matches) == 0:
                return 1.0  # Media 1 punto a partita
            
            points = 0
            for _, match in team_matches.iterrows():
                is_home = match['home_team'] == team
                hg = match.get('home_goals', 0) or 0
                ag = match.get('away_goals', 0) or 0
                
                if is_home:
                    if hg > ag:  points += 3
                    elif hg == ag: points += 1
                else:
                    if ag > hg:  points += 3
                    elif ag == hg: points += 1
            
            return points / len(team_matches)
        
        team1_form = get_team_form(team1)
        team2_form = get_team_form(team2)
        
        return {
            'form_team1_ppg': team1_form,
            'form_team2_ppg': team2_form,
            'form_diff': team1_form - team2_form,
            'form_ratio': team1_form / team2_form if team2_form > 0 else 1.5
        }
    
    def _default_h2h(self) -> Dict[str, float]: 
        """Default quando non ci sono scontri diretti."""
        return {
            'h2h_matches':  0,
            'h2h_team1_win_rate': 0.35,
            'h2h_draw_rate': 0.30,
            'h2h_team2_win_rate':  0.35,
            'h2h_team1_goals_avg':  1.3,
            'h2h_team2_goals_avg':  1.3,
            'h2h_total_goals_avg': 2.6,
            'h2h_btts_rate': 0.50,
            'h2h_over_2_5_rate': 0.50,
            'h2h_goal_diff': 0.0
        }