"""Test per i modelli predittivi."""

import numpy as np
import pandas as pd
import pytest

from football_predictor.models import BTTSModel, MatchResultModel, OverUnderModel


class TestMatchResultModel:
    """Test per MatchResultModel."""

    def setup_method(self):
        """Setup per ogni test."""
        self.model = MatchResultModel(min_accuracy=0.50, random_state=42)

        # Genera dati sintetici
        np.random.seed(42)
        n_samples = 500

        self.X = pd.DataFrame(
            {
                "home_form_ppg": np.random.uniform(0.5, 2.5, n_samples),
                "away_form_ppg": np.random.uniform(0.5, 2.5, n_samples),
                "h2h_home_win_rate": np.random.uniform(0.2, 0.6, n_samples),
                "home_rest_days": np.random.randint(3, 10, n_samples),
                "away_rest_days": np.random.randint(3, 10, n_samples),
            }
        )

        # Target basato sulle features
        self.y = pd.Series(
            [
                (
                    "H"
                    if row["home_form_ppg"] > row["away_form_ppg"] + 0.5
                    else "A" if row["away_form_ppg"] > row["home_form_ppg"] + 0.5 else "D"
                )
                for _, row in self.X.iterrows()
            ]
        )

    def test_fit(self):
        """Test training del modello."""
        self.model.fit(self.X, self.y)

        assert self.model.is_fitted == True
        assert "accuracy_mean" in self.model.metrics

    def test_predict(self):
        """Test predizioni."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X[:10])

        assert len(predictions) == 10
        assert all(p in ["H", "D", "A"] for p in predictions)

    def test_predict_proba(self):
        """Test probabilità."""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X[:10])

        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_get_class_labels(self):
        """Test etichette classi."""
        self.model.fit(self.X, self.y)
        labels = self.model.get_class_labels()

        assert set(labels) == {"H", "D", "A"}


class TestBTTSModel:
    """Test per BTTSModel."""

    def setup_method(self):
        """Setup per ogni test."""
        self.model = BTTSModel(min_accuracy=0.50, random_state=42)

        np.random.seed(42)
        n_samples = 500

        self.X = pd.DataFrame(
            {
                "home_goals_avg": np.random.uniform(0.5, 2.5, n_samples),
                "away_goals_avg": np.random.uniform(0.5, 2.5, n_samples),
                "h2h_btts_rate": np.random.uniform(0.3, 0.7, n_samples),
            }
        )

        self.y = pd.Series(
            [
                row["home_goals_avg"] > 1.0 and row["away_goals_avg"] > 1.0
                for _, row in self.X.iterrows()
            ]
        )

    def test_fit_predict(self):
        """Test training e predizione."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X[:10])

        assert self.model.is_fitted == True
        assert len(predictions) == 10


class TestOverUnderModel:
    """Test per OverUnderModel."""

    def test_initialization(self):
        """Test inizializzazione con threshold."""
        model = OverUnderModel(threshold=2.5)
        assert model.threshold == 2.5
        assert model.model_name == "Over_Under_2.5"

    def test_different_thresholds(self):
        """Test diversi threshold."""
        for threshold in [1.5, 2.5, 3.5]:
            model = OverUnderModel(threshold=threshold)
            assert model.get_target_column() == f"over_{threshold}"
