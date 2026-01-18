"""Preprocessamento dati partite."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging. getLogger(__name__)


class MatchProcessor:
    """Preprocessa i dati delle partite."""

    def __init__(self):
        self.team_name_mapping = {}

    def process_schedule(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Processa il calendario delle partite."""
        df = schedule.copy()

        # Normalizza date
        if 'date' in df. columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Estrai risultati da colonna 'score' se presente
        if 'score' in df.columns and 'home_goals' not in df. columns:
            df = self._extract_score(df)

        # Normalizza nomi squadre
        for col in ['home_team', 'away_team']:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_team_name)

        # Crea match_id
        df['match_id'] = df.apply(
            lambda x: f"{x. get('date', '')}_{x.get('home_team', '')}_{x.get('away_team', '')}",
            axis=1
        )

        # Rimuovi duplicati
        df = df.drop_duplicates(subset=['match_id'])

        # Ordina per data
        df = df.sort_values('date').reset_index(drop=True)

        return df

    def _extract_score(self, df:  pd.DataFrame) -> pd.DataFrame:
        """Estrae home_goals e away_goals dalla colonna score."""
        def parse_score(score):
            if pd.isna(score) or not isinstance(score, str):
                return None, None
            try:
                parts = score. replace('–', '-').replace('—', '-').split('-')
                if len(parts) == 2:
                    return int(parts[0]. strip()), int(parts[1].strip())
            except (ValueError, AttributeError):
                pass
            return None, None

        scores = df['score']. apply(parse_score)
        df['home_goals'] = scores.apply(lambda x: x[0])
        df['away_goals'] = scores.apply(lambda x: x[1])

        return df

    def _normalize_team_name(self, name: str) -> str:
        """Normalizza il nome della squadra."""
        if not isinstance(name, str):
            return str(name)

        if name in self.team_name_mapping:
            return self. team_name_mapping[name]

        return name. strip()

    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge le variabili target per i modelli."""
        df = df.copy()

        # Result (1X2)
        def get_result(row):
            hg = row.get('home_goals')
            ag = row. get('away_goals')
            if pd.isna(hg) or pd.isna(ag):
                return None
            if hg > ag:
                return 'H'
            elif hg < ag:
                return 'A'
            return 'D'

        df['result'] = df.apply(get_result, axis=1)

        # Exact score
        df['exact_score'] = df.apply(
            lambda x: f"{int(x['home_goals'])}-{int(x['away_goals'])}"
            if pd.notna(x.get('home_goals')) and pd.notna(x. get('away_goals'))
            else None,
            axis=1
        )

        # BTTS
        df['btts'] = df.apply(
            lambda x: x.get('home_goals', 0) > 0 and x.get('away_goals', 0) > 0
            if pd.notna(x.get('home_goals')) and pd.notna(x.get('away_goals'))
            else None,
            axis=1
        )

        # Over/Under
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['over_1. 5'] = df['total_goals'] > 1.5
        df['over_2.5'] = df['total_goals'] > 2.5
        df['over_3.5'] = df['total_goals'] > 3.5

        # Clean sheet
        df['home_clean_sheet'] = df['away_goals'] == 0
        df['away_clean_sheet'] = df['home_goals'] == 0

        return df

    def merge_with_odds(self, matches: pd.DataFrame,
                        odds: pd.DataFrame) -> pd.DataFrame:
        """Unisce i dati delle partite con le quote."""
        if odds is None or len(odds) == 0:
            return matches

        # Normalizza date in odds
        if 'date' in odds.columns:
            odds['date'] = pd.to_datetime(odds['date'], errors='coerce')

        # Normalizza nomi squadre in odds
        for col in ['home_team', 'away_team', 'HomeTeam', 'AwayTeam']:
            if col in odds.columns:
                odds[col] = odds[col].apply(self._normalize_team_name)

        # Rinomina colonne standard
        odds = odds.rename(columns={
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'Date': 'date'
        })

        # Merge
        merged = pd.merge(
            matches,
            odds,
            on=['date', 'home_team', 'away_team'],
            how='left',
            suffixes=('', '_odds')
        )

        return merged