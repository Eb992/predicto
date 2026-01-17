"""Feature Engineering per il sistema predittivo."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MatchFeatures:
    """Container per le features di una partita."""

    match_id: str
    date: datetime
    home_team: str
    away_team: str
    features: Dict[str, float]
    target: Optional[Dict[str, any]] = None


class FeatureEngineer:
    """Classe principale per la creazione delle features."""

    def __init__(self, lookback_games: int = 10):
        self.lookback_games = lookback_games
        self._team_history: Dict[str, pd.DataFrame] = {}
        self._player_stats: Dict[str, pd.DataFrame] = {}
        self._referee_stats: Dict[str, pd.DataFrame] = {}

    def set_historical_data(
        self,
        matches: pd.DataFrame,
        player_stats: pd.DataFrame,
        referee_stats: Optional[pd.DataFrame] = None,
    ):
        """Imposta i dati storici per il calcolo delle features."""
        self._matches = matches
        self._player_stats_df = player_stats
        self._referee_stats_df = referee_stats

    def compute_team_form(
        self, team: str, before_date: datetime, n_games: int = 5
    ) -> Dict[str, float]:
        """Calcola la forma recente della squadra."""
        team_matches = (
            self._matches[
                ((self._matches["home_team"] == team) | (self._matches["away_team"] == team))
                & (self._matches["date"] < before_date)
            ]
            .sort_values("date", ascending=False)
            .head(n_games)
        )

        if len(team_matches) == 0:
            return self._default_form_features()

        wins, draws, losses = 0, 0, 0
        goals_scored, goals_conceded = 0, 0

        for _, match in team_matches.iterrows():
            is_home = match["home_team"] == team
            if is_home:
                gs = match.get("home_goals", 0)
                gc = match.get("away_goals", 0)
            else:
                gs = match.get("away_goals", 0)
                gc = match.get("home_goals", 0)

            goals_scored += gs
            goals_conceded += gc

            if gs > gc:
                wins += 1
            elif gs == gc:
                draws += 1
            else:
                losses += 1

        n = len(team_matches)
        return {
            "form_win_rate": wins / n,
            "form_draw_rate": draws / n,
            "form_loss_rate": losses / n,
            "form_points_per_game": (wins * 3 + draws) / n,
            "form_goals_scored_avg": goals_scored / n,
            "form_goals_conceded_avg": goals_conceded / n,
            "form_goal_diff_avg": (goals_scored - goals_conceded) / n,
        }

    def compute_rest_days(self, team: str, match_date: datetime) -> int:
        """Calcola i giorni di riposo dalla partita precedente."""
        team_matches = self._matches[
            ((self._matches["home_team"] == team) | (self._matches["away_team"] == team))
            & (self._matches["date"] < match_date)
        ].sort_values("date", ascending=False)

        if len(team_matches) == 0:
            return 7  # Default

        last_match = team_matches.iloc[0]["date"]
        if isinstance(last_match, str):
            last_match = pd.to_datetime(last_match)
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

        return (match_date - last_match).days

    def compute_h2h_features(
        self, home_team: str, away_team: str, before_date: datetime, n_games: int = 10
    ) -> Dict[str, float]:
        """Calcola statistiche head-to-head."""
        h2h_matches = (
            self._matches[
                (
                    (
                        (self._matches["home_team"] == home_team)
                        & (self._matches["away_team"] == away_team)
                    )
                    | (
                        (self._matches["home_team"] == away_team)
                        & (self._matches["away_team"] == home_team)
                    )
                )
                & (self._matches["date"] < before_date)
            ]
            .sort_values("date", ascending=False)
            .head(n_games)
        )

        if len(h2h_matches) == 0:
            return {
                "h2h_home_win_rate": 0.33,
                "h2h_draw_rate": 0.33,
                "h2h_away_win_rate": 0.33,
                "h2h_total_goals_avg": 2.5,
                "h2h_btts_rate": 0.5,
            }

        home_wins, draws, away_wins = 0, 0, 0
        total_goals = 0
        btts_count = 0

        for _, match in h2h_matches.iterrows():
            hg = match.get("home_goals", 0)
            ag = match.get("away_goals", 0)
            total_goals += hg + ag

            if hg > 0 and ag > 0:
                btts_count += 1

            if match["home_team"] == home_team:
                if hg > ag:
                    home_wins += 1
                elif hg == ag:
                    draws += 1
                else:
                    away_wins += 1
            else:
                if ag > hg:
                    home_wins += 1
                elif ag == hg:
                    draws += 1
                else:
                    away_wins += 1

        n = len(h2h_matches)
        return {
            "h2h_home_win_rate": home_wins / n,
            "h2h_draw_rate": draws / n,
            "h2h_away_win_rate": away_wins / n,
            "h2h_total_goals_avg": total_goals / n,
            "h2h_btts_rate": btts_count / n,
        }

    def compute_player_impact(self, team: str, season: int) -> Dict[str, float]:
        """Calcola l'impatto dei giocatori chiave sulla squadra."""
        if self._player_stats_df is None or len(self._player_stats_df) == 0:
            return {
                "team_avg_player_rating": 70.0,
                "team_top_scorer_goals": 0,
                "team_key_player_influence": 0.5,
            }

        try:
            team_players = (
                self._player_stats_df[self._player_stats_df.index.get_level_values("team") == team]
                if "team" in self._player_stats_df.index.names
                else pd.DataFrame()
            )

            if len(team_players) == 0:
                return {
                    "team_avg_player_rating": 70.0,
                    "team_top_scorer_goals": 0,
                    "team_key_player_influence": 0.5,
                }

            # Calcola metriche aggregate
            return {
                "team_avg_player_rating": 70.0,  # Placeholder - da integrare con SoFIFA
                "team_top_scorer_goals": (
                    team_players.get("goals", pd.Series([0])).max()
                    if "goals" in team_players.columns
                    else 0
                ),
                "team_key_player_influence": 0.5,
            }
        except Exception:
            return {
                "team_avg_player_rating": 70.0,
                "team_top_scorer_goals": 0,
                "team_key_player_influence": 0.5,
            }

    def compute_referee_features(self, referee: str) -> Dict[str, float]:
        """Calcola statistiche dell'arbitro."""
        if self._referee_stats_df is None or referee not in self._referee_stats_df.index:
            return {
                "referee_avg_fouls": 25.0,
                "referee_avg_yellows": 4.0,
                "referee_avg_reds": 0.1,
                "referee_home_bias": 0.0,
            }

        ref_stats = self._referee_stats_df.loc[referee]
        return {
            "referee_avg_fouls": ref_stats.get("avg_fouls", 25.0),
            "referee_avg_yellows": ref_stats.get("avg_yellows", 4.0),
            "referee_avg_reds": ref_stats.get("avg_reds", 0.1),
            "referee_home_bias": ref_stats.get("home_bias", 0.0),
        }

    def compute_all_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        referee: Optional[str] = None,
        season: int = 2024,
    ) -> Dict[str, float]:
        """Calcola tutte le features per una partita."""
        features = {}

        # Form features per entrambe le squadre
        home_form = self.compute_team_form(home_team, match_date)
        away_form = self.compute_team_form(away_team, match_date)

        for k, v in home_form.items():
            features[f"home_{k}"] = v
        for k, v in away_form.items():
            features[f"away_{k}"] = v

        # Rest days
        features["home_rest_days"] = self.compute_rest_days(home_team, match_date)
        features["away_rest_days"] = self.compute_rest_days(away_team, match_date)
        features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]

        # H2H
        h2h = self.compute_h2h_features(home_team, away_team, match_date)
        features.update(h2h)

        # Player impact
        home_player = self.compute_player_impact(home_team, season)
        away_player = self.compute_player_impact(away_team, season)
        for k, v in home_player.items():
            features[f"home_{k}"] = v
        for k, v in away_player.items():
            features[f"away_{k}"] = v

        # Referee
        if referee:
            referee_features = self.compute_referee_features(referee)
            features.update(referee_features)

        # Differenze chiave
        features["form_diff"] = (
            features["home_form_points_per_game"] - features["away_form_points_per_game"]
        )
        features["attack_diff"] = (
            features["home_form_goals_scored_avg"] - features["away_form_goals_scored_avg"]
        )
        features["defense_diff"] = (
            features["away_form_goals_conceded_avg"] - features["home_form_goals_conceded_avg"]
        )

        return features

    def _default_form_features(self) -> Dict[str, float]:
        """Features di default quando mancano i dati."""
        return {
            "form_win_rate": 0.33,
            "form_draw_rate": 0.33,
            "form_loss_rate": 0.33,
            "form_points_per_game": 1.0,
            "form_goals_scored_avg": 1.5,
            "form_goals_conceded_avg": 1.5,
            "form_goal_diff_avg": 0.0,
        }
