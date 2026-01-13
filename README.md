# ⚽ Football Prediction System

Sistema predittivo per scommesse calcistiche basato su Machine Learning, con gestione del bankroll tramite Kelly Criterion.

## 🎯 Funzionalità

- **Predizioni multiple**: 
  - Risultato finale (1X2)
  - Risultato esatto (top 2 per partita)
  - BTTS (Both Teams To Score)
  - Over/Under goals
  - Statistiche giocatori (tiri, falli)
  
- **Features avanzate**: 
  - Forma recente squadre
  - Head-to-head
  - Giorni di riposo
  - Impatto giocatori chiave
  - Statistiche arbitro
  
- **Gestione bankroll**: 
  - Kelly Criterion (frazionato)
  - Stop loss / Profit target
  - Value bet detection
  
- **Backtesting**: 
  - Walk-forward validation
  - Metriche complete (Sharpe, Drawdown, ROI)
  - Report HTML/Excel

## 📦 Installazione

```bash
# Clone repository
git clone https://github.com/yourusername/football-predictor.git
cd football-predictor

# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -e . 
```

## 🚀 Utilizzo

### CLI

```bash
# Backtesting
football-predictor backtest --bankroll 1000 --leagues "ENG-Premier League" "ITA-Serie A"

# Predizioni future (7 giorni)
football-predictor predict --days 7 --markets all --min-probability 0.55

# Trova value bets
football-predictor value-bets --bankroll 1000 --kelly-fraction 0.25

# Statistiche
football-predictor stats
```

### Python

```python
from football_predictor. data.scrapers import FBrefScraper
from football_predictor. features import FeatureEngineer
from football_predictor.models import MatchResultModel, BTTSModel
from football_predictor.betting import KellyCriterion
from football_predictor.backtesting import Backtester

# Setup
leagues = ["ENG-Premier League", "ITA-Serie A"]
seasons = [2023, 2024, 2025]

# Scarica dati
scraper = FBrefScraper(leagues, seasons)
schedule = scraper.fetch_schedule()
player_stats = scraper.fetch_player_stats()

# Feature engineering
fe = FeatureEngineer(lookback_games=10)
fe.set_historical_data(schedule, player_stats)

# Prepara dati
# ...  (vedi cli/main.py per esempio completo)

# Train modelli
model_1x2 = MatchResultModel(min_accuracy=0.60)
model_btts = BTTSModel(min_accuracy=0.60)

# Backtesting
backtester = Backtester(initial_bankroll=1000)
backtester. add_model('1X2', model_1x2)
backtester.add_model('BTTS', model_btts)

results = backtester. run_backtest(data, odds_data)
print(f"ROI: {results. roi:. 2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

## 📊 Struttura Progetto

```
football_predictor/
├── config/           # Configurazioni
├── data/
│   ├── scrapers/     # Download dati (soccerdata)
│   └── processors/   # Preprocessing
├── features/         # Feature engineering
├── models/           # Modelli ML
├── betting/          # Kelly criterion, bankroll
├── backtesting/      # Engine backtesting
└── cli/              # Interfaccia CLI
```

## 🎯 Target Accuracy

| Mercato | Target | Note |
|---------|--------|------|
| 1X2 | ≥60% | Risultato finale |
| BTTS | ≥60% | Entrambe segnano |
| Over/Under 2.5 | ≥60% | Totale gol |
| Exact Score | ≥12% | Top 2 predizioni |

## 📈 Metriche Backtesting

- **ROI**: Return on Investment
- **Win Rate**: Percentuale vittorie
- **Sharpe Ratio**:  Risk-adjusted return
- **Max Drawdown**:  Perdita massima dal picco
- **Profit Factor**: Gross profit / Gross loss

## ⚠️ Disclaimer

Questo software è fornito solo a scopo educativo e di ricerca. 
Le scommesse comportano rischi finanziari.  Non garantiamo profitti. 
Gioca responsabilmente. 

## 📄 Licenza

MIT License