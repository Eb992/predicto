"""Modello per predizione Over/Under goals."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from .base_model import BasePredictionModel


class OverUnderModel(BasePredictionModel):
    """
    Modello per la predizione Over/Under goals.

    Supporta diversi threshold (1.5, 2.5, 3.5, etc.)
    """

    SUPPORTED_THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    def __init__(self, threshold: float = 2.5, min_accuracy: float = 0.60, random_state: int = 42):
        """
        Inizializza il modello Over/Under.

        Args:
            threshold: Soglia gol (es. 2.5)
            min_accuracy: Accuracy minima target
            random_state: Seed per riproducibilità
        """
        if threshold not in self.SUPPORTED_THRESHOLDS:
            raise ValueError(f"Threshold deve essere uno di {self.SUPPORTED_THRESHOLDS}")

        self.threshold = threshold

        super().__init__(
            model_name=f"Over_Under_{threshold}",
            min_accuracy=min_accuracy,
            random_state=random_state,
        )

    def _create_model(self):
        """Crea l'ensemble per Over/Under."""
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
        return f"over_{self.threshold}"

    def get_key_features_for_over_under(self) -> List[str]:
        """
        Ritorna le features più importanti per Over/Under.

        Returns:
            Lista delle features chiave
        """
        return [
            "home_form_goals_scored_avg",
            "away_form_goals_scored_avg",
            "home_form_goals_conceded_avg",
            "away_form_goals_conceded_avg",
            "h2h_total_goals_avg",
            "h2h_over_2_5_rate",
            "home_xg_for",
            "away_xg_for",
            "referee_avg_total_goals",
        ]

    def predict_over_under(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predice Over/Under con probabilità.

        Args:
            X: Features

        Returns:
            DataFrame con predizioni e probabilità
        """
        proba = self.predict_proba(X)
        predictions = self.predict(X)
        labels = self.get_class_labels()

        # Trova indici
        over_idx = list(labels).index(True) if True in labels else 1
        under_idx = list(labels).index(False) if False in labels else 0

        return pd.DataFrame(
            {
                f"over_{self. threshold}_prob": proba[:, over_idx],
                f"under_{self. threshold}_prob": proba[:, under_idx],
                "prediction": ["Over" if p else "Under" for p in predictions],
                "threshold": self.threshold,
                "confidence": np.abs(proba[:, over_idx] - 0.5) * 2,
            }
        )

    @classmethod
    def create_multi_threshold_models(
        cls, thresholds: List[float] = None, **kwargs
    ) -> Dict[float, "OverUnderModel"]:
        """
        Crea modelli per multiple soglie.

        Args:
            thresholds: Lista soglie (default: [1.5, 2.5, 3.5])
            **kwargs: Argomenti per ogni modello

        Returns:
            Dict threshold -> modello
        """
        if thresholds is None:
            thresholds = [1.5, 2.5, 3.5]

        return {threshold: cls(threshold=threshold, **kwargs) for threshold in thresholds}
