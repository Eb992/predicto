"""Engine di backtesting per il sistema predittivo."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

from .. models.base_model import (
    BasePredictionModel, 
    MatchResultModel, 
    ExactScoreModel,
    BTTSModel,
    OverUnderModel
)
from ..betting.kelly_criterion import KellyCriterion, BankrollManager, Bet
from .. features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult: 
    """Risultati del backtesting."""
    start_date: datetime
    end_date:  datetime
    total_matches: int
    total_bets:  int
    wins:  int
    losses:  int
    win_rate: float
    initial_bankroll: float
    final_bankroll: float
    total_profit: float
    roi: float
    max_drawdown: float
    sharpe_ratio: float
    model_accuracies: Dict[str, float]
    daily_returns: List[float]
    bankroll_history: List[float]


class Backtester: 
    """Engine di backtesting per valutare le strategie."""
    
    def __init__(self, 
                 initial_bankroll: float = 1000.0,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.05):
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self. min_edge = min_edge
        
        # Modelli
        self.models: Dict[str, BasePredictionModel] = {}
        
        # Feature engineer
        self. feature_engineer = FeatureEngineer()
        
        # Bankroll manager
        self.bankroll_manager = BankrollManager(initial_bankroll)
    
    def add_model(self, name: str, model:  BasePredictionModel):
        """Aggiunge un modello al backtester."""
        self. models[name] = model
    
    def run_backtest(self, 
                     data: pd.DataFrame,
                     odds_data: pd.DataFrame,
                     train_window_days: int = 365,
                     test_window_days:  int = 30,
                     min_train_matches: int = 200) -> BacktestResult:
        """
        Esegue il backtesting walk-forward.
        
        Args: 
            data: DataFrame con partite e features
            odds_data: DataFrame con le quote
            train_window_days: Finestra di training
            test_window_days: Finestra di test
            min_train_matches:  Numero minimo di partite per training
        
        Returns:
            BacktestResult con tutti i risultati
        """
        # Prepara dati
        data = data.sort_values('date').reset_index(drop=True)
        
        # Merge con odds
        if odds_data is not None and len(odds_data) > 0:
            data = self._merge_odds(data, odds_data)
        
        # Inizializza tracking
        bankroll_history = [self.initial_bankroll]
        daily_returns = []
        all_bets = []
        model_predictions = {name: [] for name in self. models}
        
        # Walk-forward
        unique_dates = sorted(data['date']. unique())
        
        logger.info(f"Backtesting dal {unique_dates[0]} al {unique_dates[-1]}")
        
        for i, test_date in enumerate(tqdm(unique_dates[min_train_matches: ], desc="Backtesting")):
            # Training set:  partite prima della data di test
            train_mask = data['date'] < test_date
            train_data = data[train_mask]. tail(min_train_matches * 10)  # Limita training
            
            if len(train_data) < min_train_matches:
                continue
            
            # Test set: partite del giorno di test
            test_mask = data['date'] == test_date
            test_data = data[test_mask]
            
            if len(test_data) == 0:
                continue
            
            # Train e predict per ogni modello
            day_bets = []
            
            for model_name, model in self.models.items():
                try:
                    # Prepara features e target
                    feature_cols = [c for c in train_data.columns 
                                   if c not in ['date', 'home_team', 'away_team', 'result', 
                                               'home_goals', 'away_goals', 'exact_score',
                                               'btts', 'over_2. 5', 'match_id']]
                    
                    X_train = train_data[feature_cols].copy()
                    target_col = model.get_target_column()
                    
                    if target_col not in train_data.columns:
                        continue
                    
                    y_train = train_data[target_col]
                    
                    # Rimuovi righe con target NaN
                    valid_idx = ~y_train.isna()
                    X_train = X_train[valid_idx]
                    y_train = y_train[valid_idx]
                    
                    if len(X_train) < 50:
                        continue
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    X_test = test_data[feature_cols].copy()
                    predictions = model. predict(X_test)
                    probabilities = model.predict_proba(X_test)
                    
                    # Valuta scommesse
                    for j, (idx, row) in enumerate(test_data.iterrows()):
                        pred = predictions[j]
                        proba = probabilities[j]
                        
                        # Trova odds corrispondenti
                        odds = self._get_odds(row, model_name, pred)
                        if odds is None or odds < 1. 1: 
                            continue
                        
                        # Probabilità della selezione
                        pred_idx = list(model. get_class_labels()).index(pred)
                        pred_proba = proba[pred_idx]
                        
                        # Valuta con Kelly
                        bet = self.bankroll_manager.kelly.evaluate_bet(
                            probability=pred_proba,
                            odds=odds,
                            match_id=row. get('match_id', str(idx)),
                            market=model_name,
                            selection=str(pred)
                        )
                        
                        if bet.is_value_bet:
                            bet.actual_outcome = row. get(target_col)
                            day_bets.append(bet)
                    
                    # Track accuracy
                    actual = test_data[target_col].values
                    model_predictions[model_name].extend(list(zip(predictions, actual)))
                    
                except Exception as e:
                    logger.warning(f"Errore modello {model_name}: {e}")
                    continue
            
            # Ottimizza stakes per correlazione
            day_bets = self. bankroll_manager.calculate_optimal_stakes(day_bets)
            
            # Simula risultati
            day_pnl = 0
            for bet in day_bets: 
                won = str(bet.actual_outcome) == bet.selection
                old_bankroll = self.bankroll_manager.current_bankroll
                self.bankroll_manager.kelly.place_bet(bet, won)
                self.bankroll_manager.current_bankroll = self.bankroll_manager.kelly.current_bankroll
                
                day_pnl += self.bankroll_manager.current_bankroll - old_bankroll
                all_bets.append({
                    'date': test_date,
                    'match_id': bet. match_id,
                    'market':  bet.market,
                    'selection': bet.selection,
                    'stake': bet.stake,
                    'odds': bet.odds,
                    'won': won,
                    'pnl': self.bankroll_manager.current_bankroll - old_bankroll
                })
            
            bankroll_history.append(self.bankroll_manager.current_bankroll)
            if len(bankroll_history) > 1:
                daily_returns.append(
                    (bankroll_history[-1] - bankroll_history[-2]) / bankroll_history[-2]
                )
            
            # Check stop conditions
            should_stop, reason = self.bankroll_manager.should_stop()
            if should_stop: 
                logger.info(f"Backtesting fermato: {reason}")
                break
        
        # Calcola metriche finali
        return self._calculate_results(
            all_bets=all_bets,
            bankroll_history=bankroll_history,
            daily_returns=daily_returns,
            model_predictions=model_predictions,
            start_date=unique_dates[0],
            end_date=unique_dates[-1]
        )
    
    def _merge_odds(self, data: pd. DataFrame, odds_data: pd.DataFrame) -> pd.DataFrame:
        """Merge partite con quote."""
        # Semplice merge su data e squadre
        return pd.merge(
            data, 
            odds_data,
            on=['date', 'home_team', 'away_team'],
            how='left'
        )
    
    def _get_odds(self, row: pd.Series, market:  str, selection: str) -> Optional[float]:
        """Recupera le quote per una selezione."""
        # Mappatura colonne odds
        odds_mapping = {
            '1X2_Result': {
                'H': 'odds_home_win',
                'D': 'odds_draw',
                'A': 'odds_away_win'
            },
            'BTTS': {
                'True': 'odds_btts_yes',
                'False': 'odds_btts_no',
                True: 'odds_btts_yes',
                False: 'odds_btts_no'
            },
            'Over_Under_2.5': {
                'True': 'odds_over_2.5',
                'False':  'odds_under_2.5',
                True: 'odds_over_2.5',
                False: 'odds_under_2.5'
            }
        }
        
        if market in odds_mapping and selection in odds_mapping[market]:
            col = odds_mapping[market][selection]
            if col in row:
                return row[col]
        
        # Fallback:  usa quote generiche se disponibili
        if 'B365H' in row and market == '1X2_Result':
            if selection == 'H':
                return row. get('B365H', row.get('odds_home_win'))
            elif selection == 'D': 
                return row.get('B365D', row.get('odds_draw'))
            elif selection == 'A':
                return row.get('B365A', row. get('odds_away_win'))
        
        return None
    
    def _calculate_results(self, 
                          all_bets: List[Dict],
                          bankroll_history: List[float],
                          daily_returns: List[float],
                          model_predictions:  Dict[str, List],
                          start_date: datetime,
                          end_date: datetime) -> BacktestResult: 
        """Calcola i risultati finali del backtest."""
        
        bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
        
        wins = len(bets_df[bets_df['won'] == True]) if len(bets_df) > 0 else 0
        losses = len(bets_df) - wins
        
        # Max drawdown
        peak = bankroll_history[0]
        max_dd = 0
        for val in bankroll_history:
            if val > peak: 
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio (annualizzato)
        if len(daily_returns) > 0:
            avg_return = np. mean(daily_returns)
            std_return = np. std(daily_returns)
            sharpe = (avg_return * 365) / (std_return * np.sqrt(365)) if std_return > 0 else 0
        else: 
            sharpe = 0
        
        # Model accuracies
        model_accuracies = {}
        for name, preds in model_predictions.items():
            if len(preds) > 0:
                correct = sum(1 for p, a in preds if str(p) == str(a))
                model_accuracies[name] = correct / len(preds)
        
        final_bankroll = bankroll_history[-1] if bankroll_history else self.initial_bankroll
        total_profit = final_bankroll - self. initial_bankroll
        total_staked = bets_df['stake'].sum() if len(bets_df) > 0 else 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_matches=len(bets_df),
            total_bets=len(bets_df),
            wins=wins,
            losses=losses,
            win_rate=wins / len(bets_df) if len(bets_df) > 0 else 0,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=final_bankroll,
            total_profit=total_profit,
            roi=total_profit / total_staked if total_staked > 0 else 0,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            model_accuracies=model_accuracies,
            daily_returns=daily_returns,
            bankroll_history=bankroll_history
        )