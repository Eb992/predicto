"""Modello per predizione Both Teams To Score."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from .base_model import BasePredictionModel


class BTTSModel(BasePredictionModel):
    """
    Modello per la predizione Both Teams To Score (BTTS).

    Predice se entrambe le squadre segneranno nella partita.
    """

    def __init__(self, min_accuracy: float = 0.60, random_state: int = 42):
        """
        Inizializza il modello BTTS.

        Args:
            min_accuracy:  Accuracy minima target
            random_state: Seed per riproducibilità
        """
        super().__init__(model_name="BTTS", min_accuracy=min_accuracy, random_state=random_state)

    def _create_model(self):
        """Crea l'ensemble per BTTS."""
        return VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_split=10,
                        random_state=self.random_state,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=self.random_state,
                    ),
                ),
            ],
            voting="soft",
        )

    def get_target_column(self) -> str:
        """Ritorna il nome della colonna target."""
        return "btts"

    def get_key_features_for_btts(self) -> List[str]:
        """
        Ritorna le features più importanti per BTTS.

        Returns:
            Lista delle features chiave
        """
        return [
            "home_form_goals_scored_avg",
            "away_form_goals_scored_avg",
            "home_form_goals_conceded_avg",
            "away_form_goals_conceded_avg",
            "h2h_btts_rate",
            "home_failed_to_score_rate",
            "away_failed_to_score_rate",
            "home_clean_sheet_rate",
            "away_clean_sheet_rate",
        ]

    def predict_btts_probability(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predice la probabilità di BTTS.

        Args:
            X:  Features

        Returns:
            DataFrame con probabilità Yes/No
        """
        proba = self.predict_proba(X)
        labels = self.get_class_labels()

        # Trova indici per True/False
        true_idx = list(labels).index(True) if True in labels else 1
        false_idx = list(labels).index(False) if False in labels else 0

        return pd.DataFrame(
            {
                "btts_yes_prob": proba[:, true_idx],
                "btts_no_prob": proba[:, false_idx],
                "prediction": self.predict(X),
                "confidence": np.abs(proba[:, true_idx] - 0.5) * 2,  # 0-1 scale
            }
        )
