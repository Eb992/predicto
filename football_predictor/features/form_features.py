"""Features sulla forma recente."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FormFeatureExtractor:
    """Estrae features sulla forma recente delle squadre."""

    def __init__(self, matches: pd.DataFrame):
        self.matches = matches.sort_values("date")

    def get_form_features(
        self, team: str, before_date: datetime, n_matches: int = 5
    ) -> Dict[str, float]:
        """Calcola le features di forma recente."""
        team_matches = (
            self.matches[
                ((self.matches["home_team"] == team) | (self.matches["away_team"] == team))
                & (self.matches["date"] < before_date)
            ]
            .sort_values("date", ascending=False)
            .head(n_matches)
        )

        if len(team_matches) == 0:
            return self._default_form()

        # Calcola risultati
        points, goals_scored, goals_conceded = 0, 0, 0
        wins, draws, losses = 0, 0, 0
        clean_sheets, failed_to_score = 0, 0

        for _, match in team_matches.iterrows():
            is_home = match["home_team"] == team
            hg = match.get("home_goals", 0) or 0
            ag = match.get("away_goals", 0) or 0

            if is_home:
                gs, gc = hg, ag
            else:
                gs, gc = ag, hg

            goals_scored += gs
            goals_conceded += gc

            if gc == 0:
                clean_sheets += 1
            if gs == 0:
                failed_to_score += 1

            if gs > gc:
                wins += 1
                points += 3
            elif gs == gc:
                draws += 1
                points += 1
            else:
                losses += 1

        n = len(team_matches)

        # Calcola streak (serie di risultati consecutivi)
        streak = self._calculate_streak(team_matches, team)

        return {
            "form_points": points,
            "form_ppg": points / n,
            "form_wins": wins,
            "form_draws": draws,
            "form_losses": losses,
            "form_win_rate": wins / n,
            "form_draw_rate": draws / n,
            "form_loss_rate": losses / n,
            "form_goals_scored": goals_scored,
            "form_goals_conceded": goals_conceded,
            "form_goals_scored_avg": goals_scored / n,
            "form_goals_conceded_avg": goals_conceded / n,
            "form_goal_diff": (goals_scored - goals_conceded) / n,
            "form_clean_sheet_rate": clean_sheets / n,
            "form_failed_to_score_rate": failed_to_score / n,
            "form_streak_type": streak["type"],
            "form_streak_length": streak["length"],
        }

    def _calculate_streak(self, matches: pd.DataFrame, team: str) -> Dict:
        """Calcola la serie di risultati corrente."""
        if len(matches) == 0:
            return {"type": 0, "length": 0}

        streak_type = None
        streak_length = 0

        for _, match in matches.iterrows():
            is_home = match["home_team"] == team
            hg = match.get("home_goals", 0) or 0
            ag = match.get("away_goals", 0) or 0

            if is_home:
                gs, gc = hg, ag
            else:
                gs, gc = ag, hg

            if gs > gc:
                result = 1  # Win
            elif gs < gc:
                result = -1  # Loss
            else:
                result = 0  # Draw

            if streak_type is None:
                streak_type = result
                streak_length = 1
            elif result == streak_type:
                streak_length += 1
            else:
                break

        return {"type": streak_type or 0, "length": streak_length}

    def get_momentum_score(self, team: str, before_date: datetime) -> float:
        """
        Calcola un punteggio di momentum pesato.
        Partite più recenti hanno peso maggiore.
        """
        team_matches = (
            self.matches[
                ((self.matches["home_team"] == team) | (self.matches["away_team"] == team))
                & (self.matches["date"] < before_date)
            ]
            .sort_values("date", ascending=False)
            .head(10)
        )

        if len(team_matches) == 0:
            return 0.5

        momentum = 0
        total_weight = 0

        for i, (_, match) in enumerate(team_matches.iterrows()):
            weight = 1 / (i + 1)  # Peso decrescente
            total_weight += weight

            is_home = match["home_team"] == team
            hg = match.get("home_goals", 0) or 0
            ag = match.get("away_goals", 0) or 0

            if is_home:
                gs, gc = hg, ag
            else:
                gs, gc = ag, hg

            if gs > gc:
                momentum += weight * 1.0
            elif gs == gc:
                momentum += weight * 0.5
            # Loss = 0

        return momentum / total_weight if total_weight > 0 else 0.5

    def get_scoring_form(
        self, team: str, before_date: datetime, n_matches: int = 10
    ) -> Dict[str, float]:
        """Analizza la tendenza al gol recente."""
        team_matches = (
            self.matches[
                ((self.matches["home_team"] == team) | (self.matches["away_team"] == team))
                & (self.matches["date"] < before_date)
            ]
            .sort_values("date", ascending=False)
            .head(n_matches)
        )

        if len(team_matches) == 0:
            return {"scoring_trend": 0, "conceding_trend": 0, "goals_variance": 0.5}

        goals_scored_list = []
        goals_conceded_list = []

        for _, match in team_matches.iterrows():
            is_home = match["home_team"] == team
            hg = match.get("home_goals", 0) or 0
            ag = match.get("away_goals", 0) or 0

            if is_home:
                goals_scored_list.append(hg)
                goals_conceded_list.append(ag)
            else:
                goals_scored_list.append(ag)
                goals_conceded_list.append(hg)

        # Trend:  confronta prima metà vs seconda metà
        mid = len(goals_scored_list) // 2
        recent_avg = np.mean(goals_scored_list[:mid]) if mid > 0 else 0
        older_avg = np.mean(goals_scored_list[mid:]) if mid > 0 else 0

        return {
            "scoring_trend": recent_avg - older_avg,
            "conceding_trend": (
                np.mean(goals_conceded_list[:mid]) - np.mean(goals_conceded_list[mid:])
                if mid > 0
                else 0
            ),
            "goals_variance": np.var(goals_scored_list) if goals_scored_list else 0.5,
            "goals_consistency": 1 / (1 + np.var(goals_scored_list)) if goals_scored_list else 0.5,
        }

    def _default_form(self) -> Dict[str, float]:
        """Form di default."""
        return {
            "form_points": 5,
            "form_ppg": 1.0,
            "form_wins": 1,
            "form_draws": 2,
            "form_losses": 2,
            "form_win_rate": 0.33,
            "form_draw_rate": 0.33,
            "form_loss_rate": 0.33,
            "form_goals_scored": 5,
            "form_goals_conceded": 5,
            "form_goals_scored_avg": 1.0,
            "form_goals_conceded_avg": 1.0,
            "form_goal_diff": 0.0,
            "form_clean_sheet_rate": 0.2,
            "form_failed_to_score_rate": 0.2,
            "form_streak_type": 0,
            "form_streak_length": 0,
        }
