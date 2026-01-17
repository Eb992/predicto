"""Features relative ai giocatori."""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class PlayerFeatureExtractor:
    """Estrae features relative ai giocatori."""

    def __init__(self, player_stats: pd.DataFrame):
        self.player_stats = player_stats
        self._player_ratings = {}

    def get_team_player_quality(self, team: str, season: int) -> Dict[str, float]:
        """Calcola metriche aggregate sulla qualità dei giocatori."""
        try:
            if "team" in self.player_stats.index.names:
                team_players = self.player_stats.xs(team, level="team")
            else:
                team_players = self.player_stats[self.player_stats["team"] == team]

            if len(team_players) == 0:
                return self._default_quality()

            return {
                "squad_depth": len(team_players),
                "avg_minutes": team_players.get("minutes", pd.Series([0])).mean(),
                "top_scorer_goals": team_players.get("goals", pd.Series([0])).max(),
                "team_total_goals": team_players.get("goals", pd.Series([0])).sum(),
                "team_total_assists": team_players.get("assists", pd.Series([0])).sum(),
            }
        except Exception:
            return self._default_quality()

    def get_key_player_availability(
        self, team: str, injured_players: List[str] = None, suspended_players: List[str] = None
    ) -> Dict[str, float]:
        """Valuta l'impatto delle assenze."""
        if injured_players is None:
            injured_players = []
        if suspended_players is None:
            suspended_players = []

        absent = set(injured_players + suspended_players)

        # Calcola impatto basato su statistiche
        # In produzione, usare dati reali su infortuni/squalifiche
        return {
            "key_players_missing": len(absent),
            "estimated_impact": min(len(absent) * 0.05, 0.25),  # Max 25% impatto
        }

    def get_player_form(self, player_name: str, n_games: int = 5) -> Dict[str, float]:
        """Calcola la forma recente di un giocatore."""
        try:
            if player_name in self.player_stats.index:
                player = self.player_stats.loc[player_name]
                return {
                    "goals_per_90": player.get("goals_per_90", 0),
                    "assists_per_90": player.get("assists_per_90", 0),
                    "shots_per_90": player.get("shots_per_90", 0),
                    "xg_per_90": player.get("xg_per_90", 0),
                }
        except Exception:
            pass

        return {"goals_per_90": 0, "assists_per_90": 0, "shots_per_90": 0, "xg_per_90": 0}

    def get_player_prop_features(
        self, player_name: str, team: str, prop_type: str = "shots_on_target"
    ) -> Dict[str, float]:
        """Features per scommesse sui giocatori (tiri, falli, etc.)."""
        try:
            if "player" in self.player_stats.index.names:
                player_data = self.player_stats.xs(player_name, level="player")
            else:
                player_data = self.player_stats[self.player_stats["player"] == player_name]

            if len(player_data) == 0:
                return self._default_player_props()

            # Media statistiche per partita
            return {
                f"{prop_type}_avg": player_data.get(prop_type, pd.Series([0])).mean(),
                f"{prop_type}_std": player_data.get(prop_type, pd.Series([0])).std(),
                "shots_avg": player_data.get("shots", pd.Series([0])).mean(),
                "shots_on_target_avg": player_data.get("shots_on_target", pd.Series([0])).mean(),
                "fouls_committed_avg": player_data.get("fouls_committed", pd.Series([0])).mean(),
                "fouls_drawn_avg": player_data.get("fouls_drawn", pd.Series([0])).mean(),
            }
        except Exception:
            return self._default_player_props()

    def _default_quality(self) -> Dict[str, float]:
        return {
            "squad_depth": 25,
            "avg_minutes": 60,
            "top_scorer_goals": 5,
            "team_total_goals": 30,
            "team_total_assists": 25,
        }

    def _default_player_props(self) -> Dict[str, float]:
        return {
            "shots_on_target_avg": 0.5,
            "shots_on_target_std": 0.5,
            "shots_avg": 1.5,
            "fouls_committed_avg": 1.0,
            "fouls_drawn_avg": 1.0,
        }
