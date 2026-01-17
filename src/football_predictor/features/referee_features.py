"""Features relative agli arbitri."""

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


class RefereeFeatureExtractor:
    """Estrae features relative agli arbitri."""

    def __init__(self, matches: pd.DataFrame):
        self.matches = matches
        self._referee_cache = {}
        self._compute_referee_stats()

    def _compute_referee_stats(self):
        """Pre-calcola statistiche per ogni arbitro."""
        if "referee" not in self.matches.columns:
            return

        for referee in self.matches["referee"].dropna().unique():
            ref_matches = self.matches[self.matches["referee"] == referee]

            if len(ref_matches) < 5:  # Minimo 5 partite
                continue

            stats = {
                "matches_officiated": len(ref_matches),
                "avg_yellow_cards": ref_matches.get("yellow_cards", pd.Series([4])).mean(),
                "avg_red_cards": ref_matches.get("red_cards", pd.Series([0.1])).mean(),
                "avg_fouls": ref_matches.get("fouls", pd.Series([24])).mean(),
                "avg_penalties": ref_matches.get("penalties", pd.Series([0.2])).mean(),
                "avg_total_goals": 0,
                "home_win_rate": 0,
                "cards_per_foul": 0,
            }

            # Calcola totale gol
            home_goals = ref_matches.get("home_goals", pd.Series([0]))
            away_goals = ref_matches.get("away_goals", pd.Series([0]))
            stats["avg_total_goals"] = (home_goals + away_goals).mean()

            # Home bias
            home_wins = len(
                ref_matches[ref_matches.get("home_goals", 0) > ref_matches.get("away_goals", 0)]
            )
            stats["home_win_rate"] = home_wins / len(ref_matches)

            # Cards per foul
            total_cards = stats["avg_yellow_cards"] + stats["avg_red_cards"]
            if stats["avg_fouls"] > 0:
                stats["cards_per_foul"] = total_cards / stats["avg_fouls"]

            self._referee_cache[referee] = stats

    def get_referee_features(self, referee: str) -> Dict[str, float]:
        """Ottieni features per un arbitro specifico."""
        if referee in self._referee_cache:
            return self._referee_cache[referee]

        # Default per arbitri sconosciuti
        return {
            "matches_officiated": 0,
            "avg_yellow_cards": 4.0,
            "avg_red_cards": 0.15,
            "avg_fouls": 24.0,
            "avg_penalties": 0.2,
            "avg_total_goals": 2.7,
            "home_win_rate": 0.46,
            "cards_per_foul": 0.17,
        }

    def get_referee_card_tendency(self, referee: str) -> str:
        """Classifica la tendenza dell'arbitro con i cartellini."""
        stats = self.get_referee_features(referee)

        if stats["avg_yellow_cards"] > 5:
            return "strict"
        elif stats["avg_yellow_cards"] < 3:
            return "lenient"
        return "average"

    def get_referee_impact_on_match(
        self, referee: str, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """Stima l'impatto dell'arbitro sulla partita."""
        ref_stats = self.get_referee_features(referee)

        # Fattori di aggiustamento basati sullo stile dell'arbitro
        goals_adjustment = (ref_stats["avg_total_goals"] - 2.7) / 2.7  # Differenza dalla media
        cards_adjustment = (ref_stats["avg_yellow_cards"] - 4.0) / 4.0

        return {
            "referee_goals_factor": 1 + goals_adjustment * 0.1,
            "referee_cards_factor": 1 + cards_adjustment * 0.2,
            "referee_home_bias": ref_stats["home_win_rate"] - 0.46,
            "referee_strictness": ref_stats["cards_per_foul"],
        }
