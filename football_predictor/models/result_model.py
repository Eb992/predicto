"""Modello per predizione risultato 1X2."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression

from .base_model import BasePredictionModel


class ResultModel(BasePredictionModel):
    """
    Modello specializzato per la predizione del risultato finale (1X2).

    Utilizza un ensemble di classificatori ottimizzato per la predizione
    del risultato delle partite di calcio.
    """

    def __init__(
        self, min_accuracy: float = 0.60, random_state: int = 42, use_ensemble: bool = True
    ):
        """
        Inizializza il modello.

        Args:
            min_accuracy:  Accuracy minima target
            random_state: Seed per riproducibilità
            use_ensemble:  Se usare ensemble o singolo modello
        """
        super().__init__(
            model_name="Match_Result_1X2", min_accuracy=min_accuracy, random_state=random_state
        )
        self.use_ensemble = use_ensemble

    def _create_model(self):
        """Crea l'ensemble di modelli per 1X2."""
        if self.use_ensemble:
            models = [
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=self.random_state,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
                (
                    "gb",
                    HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.1,
                        max_iter=200,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=self.random_state,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=self.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
            return VotingClassifier(estimators=models, voting="soft")
        else:
            return HistGradientBoostingClassifier(
                max_depth=10, learning_rate=0.1, max_iter=300, random_state=self.random_state
            )

    def get_target_column(self) -> str:
        """Ritorna il nome della colonna target."""
        return "result"

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Ottieni l'importanza delle features.

        Returns:
            Dict con nome feature -> importanza
        """
        if not self.is_fitted:
            return {}

        try:
            # Per ensemble, usa il primo modello RF
            if hasattr(self.model, "estimators_"):
                rf_model = self.model.named_estimators_.get("rf")
                if rf_model and hasattr(rf_model, "feature_importances_"):
                    return dict(zip(self.feature_names, rf_model.feature_importances_))

            # Per singolo modello
            if hasattr(self.model, "feature_importances_"):
                return dict(zip(self.feature_names, self.model.feature_importances_))
        except Exception:
            pass

        return {}

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predice con livelli di confidenza.

        Args:
            X:  Features

        Returns:
            DataFrame con predizioni e confidenza
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        labels = self.get_class_labels()

        results = []
        for i in range(len(X)):
            max_prob = probabilities[i].max()
            pred = predictions[i]

            # Determina confidenza
            if max_prob >= 0.60:
                confidence = "high"
            elif max_prob >= 0.45:
                confidence = "medium"
            else:
                confidence = "low"

            results.append(
                {
                    "prediction": pred,
                    "probability": max_prob,
                    "confidence": confidence,
                    "prob_H": probabilities[i][list(labels).index("H")] if "H" in labels else 0,
                    "prob_D": probabilities[i][list(labels).index("D")] if "D" in labels else 0,
                    "prob_A": probabilities[i][list(labels).index("A")] if "A" in labels else 0,
                }
            )

        return pd.DataFrame(results)
