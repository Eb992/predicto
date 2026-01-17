"""Preprocessamento dati partite."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MatchProcessor:
    """Preprocessa i dati delle partite."""

    def __init__(self):
        self.team_name_mapping = {}

    def process_schedule(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Processa il calendario delle partite."""
        df = schedule.copy()

        # Normalizza date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Estrai risultati da colonna 'score' se presente
        if "score" in df.columns and "home_goals" not in df.columns:
            df = self._extract_score(df)

        # Normalizza nomi squadre
        for col in ["home_team", "away_team"]:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_team_name)

        # Crea match_id
        df["match_id"] = df.apply(
            lambda x: f"{x. get('date', '')}_{x.get('home_team', '')}_{x.get('away_team', '')}",
            axis=1,
        )

        # Rimuovi duplicati
        df = df.drop_duplicates(subset=["match_id"])

        # Ordina per data
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def _extract_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estrae home_goals e away_goals dalla colonna score."""

        def parse_score(score):
            if pd.isna(score) or not isinstance(score, str):
                return None, None
            try:
                parts = score.replace("–", "-").replace("—", "-").split("-")
                if len(parts) == 2:
                    return int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, AttributeError):
                pass
            return None, None

        scores = df["score"].apply(parse_score)
        df["home_goals"] = scores.apply(lambda x: x[0])
        df["away_goals"] = scores.apply(lambda x: x[1])

        return df

    def _normalize_team_name(self, name: str) -> str:
        """Normalizza il nome della squadra."""
        if not isinstance(name, str):
            return str(name)

        # Mapping personalizzabile
        if name in self.team_name_mapping:
            return self.team_name_mapping[name]

        # Pulizia base
        name = name.strip()

        return name

    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge le variabili target per i modelli."""
        df = df.copy()

        # Result (1X2)
        def get_result(row):
            hg = row.get("home_goals")
            ag = row.get("away_goals")
            if pd.isna(hg) or pd.isna(ag):
                return None
            if hg > ag:
                return "H"
            elif hg < ag:
                return "A"
            return "D"

        df["result"] = df.apply(get_result, axis=1)

        # Exact score
        df["exact_score"] = df.apply(
            lambda x: (
                f"{int(x['home_goals'])}-{int(x['away_goals'])}"
                if pd.notna(x.get("home_goals")) and pd.notna(x.get("away_goals"))
                else None
            ),
            axis=1,
        )

        # BTTS
        df["btts"] = df.apply(
            lambda x: (
                x.get("home_goals", 0) > 0 and x.get("away_goals", 0) > 0
                if pd.notna(x.get("home_goals")) and pd.notna(x.get("away_goals"))
                else None
            ),
            axis=1,
        )

        # Over/Under
        def calculate_over_under(row, threshold=2.5):
            hg = row.get("home_goals")
            ag = row.get("away_goals")
            if pd.isna(hg) or pd.isna(ag):
                return None
            return (hg + ag) > threshold

        df["over_2.5"] = df.apply(lambda row: calculate_over_under(row, 2.5), axis=1)
        df["over_1.5"] = df.apply(lambda row: calculate_over_under(row, 1.5), axis=1)
        df["over_3.5"] = df.apply(lambda row: calculate_over_under(row, 3.5), axis=1)

        return df
