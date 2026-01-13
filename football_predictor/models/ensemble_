"""Ensemble di modelli per predizioni combinate."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator
from . base_model import (
    BasePredictionModel,
    MatchResultModel,
    ExactScoreModel,
    BTTSModel,
    OverUnderModel
)
import logging

logger = logging. getLogger(__name__)


class EnsemblePredictor: 
    """Combina predizioni da multipli modelli."""
    
    def __init__(self, models: Dict[str, BasePredictionModel] = None):
        self.models = models or {}
        self.weights = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model:  BasePredictionModel, weight: float = 1.0):
        """Aggiunge un modello all'ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def fit(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]):
        """
        Addestra tutti i modelli dell'ensemble. 
        
        Args: 
            X: Features
            y_dict: Dizionario {model_name: target_series}
        """
        for name, model in self.models. items():
            if name in y_dict:
                target_col = model.get_target_column()
                y = y_dict. get(target_col, y_dict. get(name))
                
                if y is not None and len(y. dropna()) > 50: 
                    valid_idx = ~y.isna()
                    try:
                        model.fit(X[valid_idx], y[valid_idx])
                        logger.info(f"Modello {name} addestrato - Accuracy: {model.metrics. get('accuracy_mean', 0):.2%}")
                    except Exception as e: 
                        logger.warning(f"Errore training {name}: {e}")
        
        self.is_fitted = True
        return self
    
    def predict_all(self, X:  pd.DataFrame) -> Dict[str, np.ndarray]: 
        """Genera predizioni da tutti i modelli."""
        predictions = {}
        
        for name, model in self.models. items():
            if model.is_fitted:
                try: 
                    predictions[name] = {
                        'classes': model.predict(X),
                        'probabilities': model.predict_proba(X),
                        'labels': model.get_class_labels()
                    }
                except Exception as e: 
                    logger.warning(f"Errore predizione {name}: {e}")
        
        return predictions
    
    def get_combined_prediction(self, X: pd.DataFrame, 
                                market:  str) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Combina predizioni di modelli simili usando weighted voting.
        
        Args:
            X: Features
            market: Mercato da predire ('1x2', 'btts', etc.)
        
        Returns:
            (predictions, probabilities)
        """
        relevant_models = [
            (name, model) for name, model in self.models.items()
            if market. lower() in name.lower() and model.is_fitted
        ]
        
        if not relevant_models: 
            raise ValueError(f"Nessun modello trovato per il mercato {market}")
        
        # Se un solo modello, ritorna direttamente
        if len(relevant_models) == 1:
            name, model = relevant_models[0]
            return model.predict(X), model.predict_proba(X)
        
        # Weighted average delle probabilità
        all_proba = []
        all_weights = []
        
        for name, model in relevant_models: 
            proba = model.predict_proba(X)
            weight = self.weights.get(name, 1.0)
            all_proba. append(proba * weight)
            all_weights.append(weight)
        
        combined_proba = np.sum(all_proba, axis=0) / sum(all_weights)
        
        # Predizione = classe con probabilità massima
        labels = relevant_models[0][1].get_class_labels()
        predictions = labels[np.argmax(combined_proba, axis=1)]
        
        return predictions, combined_proba
    
    def get_confidence_score(self, probabilities: np.ndarray) -> np.ndarray:
        """Calcola un punteggio di confidenza per ogni predizione."""
        # Confidenza = max probabilità - seconda probabilità più alta
        sorted_proba = np.sort(probabilities, axis=1)[:, ::-1]
        
        if sorted_proba.shape[1] >= 2:
            confidence = sorted_proba[:, 0] - sorted_proba[:, 1]
        else:
            confidence = sorted_proba[:, 0]
        
        return confidence
    
    def evaluate_ensemble(self, X:  pd.DataFrame, 
                          y_true: Dict[str, pd. Series]) -> Dict[str, float]:
        """Valuta le performance dell'ensemble."""
        results = {}
        
        for name, model in self.models. items():
            if model.is_fitted and name in y_true:
                try: 
                    predictions = model.predict(X)
                    accuracy = np.mean(predictions == y_true[name]. values)
                    results[name] = {
                        'accuracy': accuracy,
                        'above_threshold': accuracy >= model.min_accuracy
                    }
                except Exception as e:
                    logger.warning(f"Errore valutazione {name}:  {e}")
        
        return results


class StackedEnsemble(EnsemblePredictor):
    """Ensemble con meta-learner per combinare predizioni."""
    
    def __init__(self, base_models: Dict[str, BasePredictionModel] = None,
                 meta_learner: BaseEstimator = None):
        super().__init__(base_models)
        from sklearn.linear_model import LogisticRegression
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)
        self.meta_features_names = []
    
    def fit(self, X:  pd.DataFrame, y:  pd.Series,
            y_dict: Dict[str, pd.Series] = None):
        """
        Addestra l'ensemble stacked in due fasi: 
        1. Addestra base models
        2. Addestra meta-learner sulle predizioni dei base models
        """
        # Fase 1: addestra base models
        if y_dict: 
            super().fit(X, y_dict)
        
        # Fase 2: genera meta-features
        meta_features = self._generate_meta_features(X)
        
        if len(meta_features. columns) > 0:
            self.meta_features_names = list(meta_features. columns)
            self.meta_learner.fit(meta_features, y)
        
        return self
    
    def _generate_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera features dalle predizioni dei base models."""
        meta_features = {}
        
        for name, model in self.models.items():
            if model. is_fitted: 
                try:
                    proba = model.predict_proba(X)
                    for i, label in enumerate(model. get_class_labels()):
                        meta_features[f"{name}_{label}_proba"] = proba[:, i]
                except Exception: 
                    pass
        
        return pd.DataFrame(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice usando il meta-learner."""
        meta_features = self._generate_meta_features(X)
        
        if len(meta_features.columns) == 0:
            raise ValueError("Nessun base model addestrato")
        
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: pd. DataFrame) -> np.ndarray:
        """Predice probabilità usando il meta-learner."""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)