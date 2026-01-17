"""Modello per predizione risultato esatto."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn. ensemble import HistGradientBoostingClassifier
from collections import Counter

from . base_model import BasePredictionModel


class ExactScoreModel(BasePredictionModel):
    """
    Modello per la predizione del risultato esatto. 
    
    Predice i risultati esatti più probabili (es. 1-0, 2-1, 1-1).
    Ritorna almeno 2 predizioni per partita come richiesto.
    """
    
    # Risultati più comuni nel calcio
    COMMON_SCORES = [
        '1-0', '0-0', '1-1', '2-1', '2-0', 
        '0-1', '1-2', '0-2', '2-2', '3-1',
        '3-0', '1-3', '0-3', '3-2', '2-3',
        '4-0', '4-1', '4-2', '0-4', '1-4'
    ]
    
    def __init__(self,
                 min_accuracy: float = 0.12,  # Target più basso per exact score
                 random_state: int = 42,
                 top_n_predictions: int = 2,
                 max_scores:  int = 20):
        """
        Inizializza il modello. 
        
        Args: 
            min_accuracy: Accuracy minima target (più bassa per exact score)
            random_state:  Seed per riproducibilità
            top_n_predictions:  Numero di predizioni per partita
            max_scores: Numero massimo di score da considerare
        """
        super().__init__(
            model_name="Exact_Score",
            min_accuracy=min_accuracy,
            random_state=random_state
        )
        self.top_n_predictions = top_n_predictions
        self.max_scores = max_scores
        self.score_to_idx = {}
        self.idx_to_score = {}
    
    def _create_model(self):
        """Crea il modello per risultato esatto."""
        return HistGradientBoostingClassifier(
            max_depth=12,
            learning_rate=0.05,
            max_iter=500,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=self.random_state
        )
    
    def get_target_column(self) -> str:
        """Ritorna il nome della colonna target."""
        return 'exact_score'
    
    def fit(self, X:  pd.DataFrame, y: pd.Series):
        """
        Addestra il modello. 
        
        Limita automaticamente ai risultati più comuni per gestire
        la dimensionalità del problema.
        """
        # Filtra ai risultati più comuni
        score_counts = y.value_counts()
        top_scores = score_counts. head(self.max_scores).index.tolist()
        
        # Crea mapping
        self.score_to_idx = {score:  i for i, score in enumerate(top_scores)}
        self.idx_to_score = {i: score for score, i in self. score_to_idx.items()}
        
        # Filtra dati
        mask = y.isin(top_scores)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Chiama il fit della classe base
        return super().fit(X_filtered, y_filtered)
    
    def get_top_predictions(self, X: pd.DataFrame, n:  int = None) -> List[List[Tuple[str, float]]]:
        """
        Ritorna le top N predizioni per ogni partita.
        
        Args: 
            X: Features
            n:  Numero di predizioni (default: self.top_n_predictions)
        
        Returns:
            Lista di liste di tuple (score, probabilità)
        """
        if n is None: 
            n = self. top_n_predictions
        
        probabilities = self.predict_proba(X)
        labels = self.get_class_labels()
        
        results = []
        for i in range(len(X)):
            proba = probabilities[i]
            # Ordina per probabilità decrescente
            sorted_indices = np. argsort(proba)[::-1][: n]
            
            predictions = [
                (str(labels[idx]), float(proba[idx]))
                for idx in sorted_indices
            ]
            results.append(predictions)
        
        return results
    
    def predict_with_alternatives(self, X:  pd.DataFrame) -> pd.DataFrame:
        """
        Predice con risultati alternativi. 
        
        Args:
            X:  Features
        
        Returns:
            DataFrame con predizione principale e alternative
        """
        top_preds = self.get_top_predictions(X, n=3)
        
        results = []
        for preds in top_preds:
            row = {
                'prediction_1': preds[0][0] if len(preds) > 0 else None,
                'probability_1': preds[0][1] if len(preds) > 0 else 0,
                'prediction_2': preds[1][0] if len(preds) > 1 else None,
                'probability_2': preds[1][1] if len(preds) > 1 else 0,
                'prediction_3': preds[2][0] if len(preds) > 2 else None,
                'probability_3': preds[2][1] if len(preds) > 2 else 0,
            }
            results.append(row)
        
        return pd. DataFrame(results)
    
    def get_score_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ottieni la distribuzione completa delle probabilità per ogni score.
        
        Args:
            X: Features
        
        Returns:
            DataFrame con probabilità per ogni score
        """
        probabilities = self.predict_proba(X)
        labels = self.get_class_labels()
        
        return pd.DataFrame(probabilities, columns=labels)