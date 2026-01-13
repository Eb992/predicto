"""Modello per predizione statistiche giocatori."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from .base_model import BasePredictionModel


class PlayerStatsModel(BasePredictionModel):
    """
    Modello per la predizione di statistiche individuali dei giocatori. 
    
    Supporta: 
    - Tiri in porta
    - Falli commessi
    - Cartellini
    - Altre statistiche
    """
    
    SUPPORTED_STATS = [
        'shots_on_target',
        'shots',
        'fouls_committed',
        'fouls_drawn',
        'yellow_cards',
        'tackles',
        'interceptions',
        'passes',
        'key_passes'
    ]
    
    def __init__(self,
                 stat_type: str = 'shots_on_target',
                 threshold: float = 0.5,
                 min_accuracy:  float = 0.55,
                 random_state: int = 42,
                 use_regression: bool = False):
        """
        Inizializza il modello per statistiche giocatori.
        
        Args: 
            stat_type:  Tipo di statistica da predire
            threshold:  Soglia per classificazione over/under
            min_accuracy: Accuracy minima target
            random_state:  Seed per riproducibilità
            use_regression: Se usare regressione invece di classificazione
        """
        if stat_type not in self.SUPPORTED_STATS:
            raise ValueError(f"stat_type deve essere uno di {self. SUPPORTED_STATS}")
        
        self.stat_type = stat_type
        self.threshold = threshold
        self.use_regression = use_regression
        
        super().__init__(
            model_name=f"Player_{stat_type}",
            min_accuracy=min_accuracy,
            random_state=random_state
        )
    
    def _create_model(self):
        """Crea il modello per statistiche giocatori."""
        if self.use_regression:
            return HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.1,
                max_iter=150,
                random_state=self. random_state
            )
        else:
            return HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.1,
                max_iter=150,
                random_state=self.random_state
            )
    
    def get_target_column(self) -> str:
        """Ritorna il nome della colonna target."""
        if self.use_regression:
            return self.stat_type
        return f'{self.stat_type}_over_{self.threshold}'
    
    def prepare_target(self, y: pd.Series) -> pd.Series:
        """
        Prepara il target per la classificazione.
        
        Args:
            y: Serie con valori numerici
        
        Returns:
            Serie booleana (over threshold)
        """
        if self.use_regression:
            return y
        return y > self.threshold
    
    def get_player_features(self) -> List[str]:
        """
        Ritorna le features rilevanti per le statistiche giocatori.
        
        Returns:
            Lista delle features
        """
        base_features = [
            'player_avg_minutes',
            'player_form_score',
            'player_position_group',
            'team_possession_avg',
            'opponent_fouls_avg',
        ]
        
        stat_specific = {
            'shots_on_target':  [
                'player_shots_per_90',
                'player_xg_per_90',
                'opponent_shots_conceded_avg'
            ],
            'fouls_committed': [
                'player_fouls_per_90',
                'referee_avg_fouls',
                'match_importance'
            ],
            'yellow_cards':  [
                'player_yellows_per_90',
                'referee_avg_yellows',
                'player_fouls_per_90'
            ]
        }
        
        return base_features + stat_specific. get(self.stat_type, [])
    
    def predict_player_stat(self, X:  pd.DataFrame, 
                            player_name: str = None) -> pd.DataFrame:
        """
        Predice la statistica per i giocatori. 
        
        Args: 
            X: Features
            player_name:  Nome giocatore (opzionale, per logging)
        
        Returns:
            DataFrame con predizioni
        """
        if self.use_regression:
            predictions = self.model.predict(self.scaler.transform(self.prepare_features(X)))
            return pd.DataFrame({
                'player':  player_name,
                'stat_type': self.stat_type,
                'predicted_value': predictions,
                'threshold':  self.threshold,
                'over_threshold':  predictions > self.threshold
            })
        else:
            proba = self.predict_proba(X)
            predictions = self.predict(X)
            
            return pd.DataFrame({
                'player': player_name,
                'stat_type':  self.stat_type,
                'threshold': self.threshold,
                'prediction': predictions,
                'over_probability': proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
                'confidence': np. abs(proba. max(axis=1) - 0.5) * 2
            })
    
    @classmethod
    def create_multi_stat_models(cls,
                                  stats: List[str] = None,
                                  **kwargs) -> Dict[str, 'PlayerStatsModel']:
        """
        Crea modelli per multiple statistiche.
        
        Args:
            stats: Lista statistiche (default: shots_on_target, fouls_committed)
            **kwargs:  Argomenti per ogni modello
        
        Returns: 
            Dict stat_type -> modello
        """
        if stats is None: 
            stats = ['shots_on_target', 'fouls_committed', 'yellow_cards']
        
        return {
            stat: cls(stat_type=stat, **kwargs)
            for stat in stats
        }