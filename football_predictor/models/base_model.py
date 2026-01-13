"""Modelli predittivi base."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn. pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn. impute import SimpleImputer
import joblib
import logging

logger = logging.getLogger(__name__)


class BasePredictionModel(ABC):
    """Classe base per tutti i modelli predittivi."""
    
    def __init__(self, 
                 model_name: str,
                 min_accuracy:  float = 0.60,
                 random_state: int = 42):
        self.model_name = model_name
        self.min_accuracy = min_accuracy
        self.random_state = random_state
        self. model:  Optional[BaseEstimator] = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator: 
        """Crea il modello specifico."""
        pass
    
    @abstractmethod
    def get_target_column(self) -> str:
        """Ritorna il nome della colonna target."""
        pass
    
    def prepare_features(self, X:  pd.DataFrame) -> np.ndarray:
        """Prepara le features per il training/prediction."""
        # Gestione valori mancanti
        X_filled = X.fillna(X.median())
        return X_filled.values
    
    def fit(self, X:  pd.DataFrame, y: pd.Series) -> "BasePredictionModel":
        """Addestra il modello."""
        self.feature_names = list(X.columns)
        
        # Prepara features
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler.fit_transform(X_prepared)
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Crea e addestra modello
        self. model = self._create_model()
        
        # Calibrazione probabilità
        calibrated_model = CalibratedClassifierCV(
            self.model, cv=3, method='isotonic'
        )
        calibrated_model.fit(X_scaled, y_encoded)
        self.model = calibrated_model
        
        # Valutazione con cross-validation temporale
        cv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        self.metrics = {
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores. std(),
            'cv_scores': scores. tolist()
        }
        
        self.is_fitted = True
        
        if scores.mean() < self.min_accuracy:
            logger.warning(
                f"Modello {self.model_name}:  accuracy {scores.mean():.2%} "
                f"sotto la soglia minima {self.min_accuracy:.2%}"
            )
        
        return self
    
    def predict(self, X:  pd.DataFrame) -> np.ndarray:
        """Predice le classi."""
        if not self.is_fitted:
            raise ValueError("Il modello non è stato addestrato")
        
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler. transform(X_prepared)
        y_pred_encoded = self.model. predict(X_scaled)
        return self.label_encoder. inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: pd. DataFrame) -> np.ndarray:
        """Predice le probabilità per ogni classe."""
        if not self.is_fitted:
            raise ValueError("Il modello non è stato addestrato")
        
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        return self.model.predict_proba(X_scaled)
    
    def get_class_labels(self) -> np.ndarray:
        """Ritorna le etichette delle classi."""
        return self.label_encoder.classes_
    
    def save(self, path: str):
        """Salva il modello su disco."""
        joblib.dump({
            'model':  self.model,
            'scaler':  self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names':  self.feature_names,
            'metrics': self.metrics
        }, path)
    
    def load(self, path:  str):
        """Carica il modello da disco."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.is_fitted = True


class MatchResultModel(BasePredictionModel):
    """Modello per predizione 1X2."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="1X2_Result", **kwargs)
    
    def _create_model(self) -> BaseEstimator:
        """Crea ensemble per 1X2."""
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=10,
                random_state=self. random_state,
                n_jobs=-1
            )),
            ('gb', HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.1,
                max_iter=200,
                random_state=self. random_state
            )),
            ('lr', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            ))
        ]
        return VotingClassifier(estimators=models, voting='soft')
    
    def get_target_column(self) -> str:
        return 'result'  # 'H', 'D', 'A'


class ExactScoreModel(BasePredictionModel):
    """Modello per predizione risultato esatto."""
    
    def __init__(self, top_n_scores: int = 20, **kwargs):
        super().__init__(model_name="Exact_Score", **kwargs)
        self.top_n_scores = top_n_scores
    
    def _create_model(self) -> BaseEstimator: 
        """Crea modello per risultato esatto."""
        # Usiamo un modello più potente per la classificazione multi-classe
        return HistGradientBoostingClassifier(
            max_depth=12,
            learning_rate=0.05,
            max_iter=500,
            random_state=self.random_state
        )
    
    def get_target_column(self) -> str:
        return 'exact_score'  # Es:  '2-1', '1-1', etc. 
    
    def get_top_predictions(self, X: pd.DataFrame, n:  int = 2) -> List[Tuple[str, float]]: 
        """Ritorna le top N predizioni per ogni partita."""
        proba = self.predict_proba(X)
        classes = self.get_class_labels()
        
        results = []
        for i in range(len(X)):
            sorted_idx = np.argsort(proba[i])[::-1][: n]
            predictions = [(classes[idx], proba[i][idx]) for idx in sorted_idx]
            results.append(predictions)
        
        return results


class BTTSModel(BasePredictionModel):
    """Modello per Both Teams To Score."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="BTTS", **kwargs)
    
    def _create_model(self) -> BaseEstimator:
        return VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=self.random_state))
            ],
            voting='soft'
        )
    
    def get_target_column(self) -> str:
        return 'btts'  # True/False


class OverUnderModel(BasePredictionModel):
    """Modello per Over/Under goals."""
    
    def __init__(self, threshold: float = 2.5, **kwargs):
        super().__init__(model_name=f"Over_Under_{threshold}", **kwargs)
        self.threshold = threshold
    
    def _create_model(self) -> BaseEstimator:
        return VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self. random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=self.random_state))
            ],
            voting='soft'
        )
    
    def get_target_column(self) -> str:
        return f'over_{self.threshold}'


class PlayerStatsModel(BasePredictionModel):
    """Modello per statistiche giocatori (tiri, falli, etc.)."""
    
    def __init__(self, stat_type: str = 'shots_on_target', **kwargs):
        super().__init__(model_name=f"Player_{stat_type}", **kwargs)
        self.stat_type = stat_type
    
    def _create_model(self) -> BaseEstimator:
        # Per regressione potremmo usare un regressore, ma per classificazione
        # (es. over/under X tiri) usiamo classificatore
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.1,
            max_iter=150,
            random_state=self.random_state
        )
    
    def get_target_column(self) -> str:
        return self.stat_type