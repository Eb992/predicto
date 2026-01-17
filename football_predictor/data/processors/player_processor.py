"""Preprocessamento dati giocatori."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PlayerProcessor:
    """Preprocessa i dati dei giocatori."""

    def __init__(self):
        self.position_groups = {
            "GK": ["GK", "Goalkeeper"],
            "DEF": ["DF", "CB", "LB", "RB", "LWB", "RWB", "Defender"],
            "MID": ["MF", "CM", "DM", "AM", "LM", "RM", "Midfielder"],
            "FWD": ["FW", "CF", "ST", "LW", "RW", "Forward", "Striker"],
        }

    def process_player_stats(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Processa le statistiche dei giocatori."""
        df = stats.copy()

        # Normalizza nomi colonne
        df.columns = [self._normalize_column_name(c) for c in df.columns]

        # Converti tipi numerici
        numeric_cols = [
            "goals",
            "assists",
            "minutes",
            "shots",
            "shots_on_target",
            "xg",
            "xa",
            "fouls_committed",
            "fouls_drawn",
            "yellow_cards",
            "red_cards",
            "passes",
            "pass_accuracy",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calcola metriche per 90 minuti
        if "minutes" in df.columns:
            minutes_90 = df["minutes"] / 90
            minutes_90 = minutes_90.replace(0, np.nan)

            for col in ["goals", "assists", "shots", "shots_on_target", "xg", "xa"]:
                if col in df.columns:
                    df[f"{col}_per_90"] = df[col] / minutes_90

        # Aggiungi gruppo posizione
        if "position" in df.columns or "pos" in df.columns:
            pos_col = "position" if "position" in df.columns else "pos"
            df["position_group"] = df[pos_col].apply(self._get_position_group)

        return df

    def _normalize_column_name(self, col: str) -> str:
        """Normalizza il nome della colonna."""
        if isinstance(col, tuple):
            col = "_".join(str(c) for c in col if c and str(c) != "Unnamed")
        return str(col).lower().replace(" ", "_").replace("-", "_")

    def _get_position_group(self, position: str) -> str:
        """Determina il gruppo di posizione."""
        if pd.isna(position):
            return "Unknown"

        position = str(position).upper()

        for group, positions in self.position_groups.items():
            if any(p.upper() in position for p in positions):
                return group

        return "Unknown"

    def aggregate_team_stats(
        self, player_stats: pd.DataFrame, team: str, season: int = None
    ) -> Dict[str, float]:
        """Aggrega le statistiche dei giocatori per squadra."""
        try:
            # Filtra per squadra
            if "team" in player_stats.index.names:
                team_df = player_stats.xs(team, level="team")
            elif "team" in player_stats.columns:
                team_df = player_stats[player_stats["team"] == team]
            else:
                return self._default_team_aggregates()

            if len(team_df) == 0:
                return self._default_team_aggregates()

            # Calcola aggregati
            return {
                "squad_size": len(team_df),
                "total_goals": team_df.get("goals", pd.Series([0])).sum(),
                "total_assists": team_df.get("assists", pd.Series([0])).sum(),
                "avg_age": team_df.get("age", pd.Series([25])).mean(),
                "top_scorer_goals": team_df.get("goals", pd.Series([0])).max(),
                "avg_minutes": team_df.get("minutes", pd.Series([0])).mean(),
                "avg_xg_per_90": team_df.get("xg_per_90", pd.Series([0])).mean(),
                "total_yellows": team_df.get("yellow_cards", pd.Series([0])).sum(),
                "total_reds": team_df.get("red_cards", pd.Series([0])).sum(),
            }
        except Exception as e:
            logger.debug(f"Errore aggregazione stats {team}: {e}")
            return self._default_team_aggregates()

    def get_key_players(self, player_stats: pd.DataFrame, team: str, n: int = 5) -> List[Dict]:
        """Identifica i giocatori chiave di una squadra."""
        try:
            if "team" in player_stats.index.names:
                team_df = player_stats.xs(team, level="team").reset_index()
            elif "team" in player_stats.columns:
                team_df = player_stats[player_stats["team"] == team]
            else:
                return []

            if len(team_df) == 0:
                return []

            # Score di importanza:  goals + assists + minutes
            if "goals" in team_df.columns and "assists" in team_df.columns:
                team_df["importance_score"] = (
                    team_df.get("goals", 0) * 3
                    + team_df.get("assists", 0) * 2
                    + team_df.get("minutes", 0) / 100
                )

                top_players = team_df.nlargest(n, "importance_score")

                return [
                    {
                        "player": row.get("player", "Unknown"),
                        "goals": row.get("goals", 0),
                        "assists": row.get("assists", 0),
                        "importance": row.get("importance_score", 0),
                    }
                    for _, row in top_players.iterrows()
                ]

            return []
        except Exception:
            return []

    def _default_team_aggregates(self) -> Dict[str, float]:
        """Aggregati di default."""
        return {
            "squad_size": 25,
            "total_goals": 30,
            "total_assists": 25,
            "avg_age": 26,
            "top_scorer_goals": 10,
            "avg_minutes": 1000,
            "avg_xg_per_90": 0.3,
            "total_yellows": 50,
            "total_reds": 2,
        }
