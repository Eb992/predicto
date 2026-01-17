"""Features specifiche per le squadre."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class TeamFeatureExtractor:
    """Estrae features relative alle squadre."""

    def __init__(self, historical_matches: pd.DataFrame):
        self.matches = historical_matches
        self._cache = {}

    def get_team_stats(
        self, team: str, before_date: datetime, n_games: int = 10, venue: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calcola statistiche aggregate per una squadra.

        Args:
            team: Nome squadra
            before_date: Data limite
            n_games: Numero partite da considerare
            venue:  'home', 'away' o None per tutte
        """
        if venue == "home":
            mask = self.matches["home_team"] == team
        elif venue == "away":
            mask = self.matches["away_team"] == team
        else:
            mask = (self.matches["home_team"] == team) | (self.matches["away_team"] == team)

        team_matches = self.matches[mask & (self.matches["date"] < before_date)]
        team_matches = team_matches.sort_values("date", ascending=False).head(n_games)

        if len(team_matches) == 0:
            return self._default_stats()

        stats = {
            "matches_played": len(team_matches),
            "goals_scored": 0,
            "goals_conceded": 0,
            "xg_for": 0,
            "xg_against": 0,
            "shots": 0,
            "shots_on_target": 0,
            "possession_avg": 0,
            "corners": 0,
            "fouls": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "clean_sheets": 0,
            "failed_to_score": 0,
        }

        for _, match in team_matches.iterrows():
            is_home = match["home_team"] == team

            if is_home:
                gs = match.get("home_goals", 0) or 0
                gc = match.get("away_goals", 0) or 0
                xg = match.get("home_xg", match.get("xg_home", 0)) or 0
                xga = match.get("away_xg", match.get("xg_away", 0)) or 0
            else:
                gs = match.get("away_goals", 0) or 0
                gc = match.get("home_goals", 0) or 0
                xg = match.get("away_xg", match.get("xg_away", 0)) or 0
                xga = match.get("home_xg", match.get("xg_home", 0)) or 0

            stats["goals_scored"] += gs
            stats["goals_conceded"] += gc
            stats["xg_for"] += xg
            stats["xg_against"] += xga

            if gc == 0:
                stats["clean_sheets"] += 1
            if gs == 0:
                stats["failed_to_score"] += 1

        n = len(team_matches)
        return {
            "avg_goals_scored": stats["goals_scored"] / n,
            "avg_goals_conceded": stats["goals_conceded"] / n,
            "avg_xg_for": stats["xg_for"] / n,
            "avg_xg_against": stats["xg_against"] / n,
            "clean_sheet_rate": stats["clean_sheets"] / n,
            "failed_to_score_rate": stats["failed_to_score"] / n,
            "goal_diff_avg": (stats["goals_scored"] - stats["goals_conceded"]) / n,
            "xg_diff_avg": (stats["xg_for"] - stats["xg_against"]) / n,
        }

    def get_scoring_patterns(
        self, team: str, before_date: datetime, n_games: int = 20
    ) -> Dict[str, float]:
        """Analizza i pattern di gol della squadra."""
        mask = (self.matches["home_team"] == team) | (self.matches["away_team"] == team)
        team_matches = self.matches[mask & (self.matches["date"] < before_date)]
        team_matches = team_matches.sort_values("date", ascending=False).head(n_games)

        if len(team_matches) == 0:
            return {
                "over_1_5_rate": 0.5,
                "over_2_5_rate": 0.5,
                "over_3_5_rate": 0.25,
                "btts_rate": 0.5,
                "first_half_goals_avg": 0.5,
                "second_half_goals_avg": 0.5,
            }

        over_1_5, over_2_5, over_3_5, btts = 0, 0, 0, 0

        for _, match in team_matches.iterrows():
            hg = match.get("home_goals", 0) or 0
            ag = match.get("away_goals", 0) or 0
            total = hg + ag

            if total > 1.5:
                over_1_5 += 1
            if total > 2.5:
                over_2_5 += 1
            if total > 3.5:
                over_3_5 += 1
            if hg > 0 and ag > 0:
                btts += 1

        n = len(team_matches)
        return {
            "over_1_5_rate": over_1_5 / n,
            "over_2_5_rate": over_2_5 / n,
            "over_3_5_rate": over_3_5 / n,
            "btts_rate": btts / n,
        }

    def get_home_away_splits(
        self, team: str, before_date: datetime, n_games: int = 10
    ) -> Dict[str, float]:
        """Confronta performance casa/trasferta."""
        home_stats = self.get_team_stats(team, before_date, n_games, "home")
        away_stats = self.get_team_stats(team, before_date, n_games, "away")

        return {
            "home_goals_avg": home_stats["avg_goals_scored"],
            "away_goals_avg": away_stats["avg_goals_scored"],
            "home_conceded_avg": home_stats["avg_goals_conceded"],
            "away_conceded_avg": away_stats["avg_goals_conceded"],
            "home_advantage": home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
        }

    def _default_stats(self) -> Dict[str, float]:
        """Stats di default quando mancano dati."""
        return {
            "avg_goals_scored": 1.3,
            "avg_goals_conceded": 1.3,
            "avg_xg_for": 1.3,
            "avg_xg_against": 1.3,
            "clean_sheet_rate": 0.3,
            "failed_to_score_rate": 0.25,
            "goal_diff_avg": 0.0,
            "xg_diff_avg": 0.0,
        }
