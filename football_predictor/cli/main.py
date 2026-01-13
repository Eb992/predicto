"""CLI principale per il sistema predittivo."""

import click
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from ..config.settings import Config
from ..data.scrapers.base_scraper import FBrefScraper, MatchHistoryScraper, UnderstatScraper
from ..features.feature_engineering import FeatureEngineer
from ..models.base_model import (
    MatchResultModel, 
    ExactScoreModel, 
    BTTSModel, 
    OverUnderModel,
    PlayerStatsModel
)
from ..backtesting.backtester import Backtester
from ..betting.kelly_criterion import KellyCriterion, BankrollManager

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Path al file di configurazione')
@click.pass_context
def cli(ctx, config):
    """⚽ Sistema Predittivo per Scommesse Calcistiche."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config.load(config)


@cli.command()
@click.option('--start-date', '-s', default=None, help='Data inizio backtest (YYYY-MM-DD)')
@click.option('--end-date', '-e', default=None, help='Data fine backtest (YYYY-MM-DD)')
@click.option('--bankroll', '-b', default=1000.0, help='Bankroll iniziale')
@click.option('--kelly-fraction', '-k', default=0.25, help='Frazione Kelly (0.25 = quarter Kelly)')
@click.option('--leagues', '-l', multiple=True, help='Leghe da includere')
@click.option('--output', '-o', default='backtest_results.json', help='File output risultati')
@click.pass_context
def backtest(ctx, start_date, end_date, bankroll, kelly_fraction, leagues, output):
    """📊 Esegue il backtesting sui dati storici."""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold blue]⚽ Football Prediction System - Backtesting[/bold blue]",
        border_style="blue"
    ))
    
    leagues_list = list(leagues) if leagues else config.data.leagues
    
    console.print(f"[cyan]Leghe selezionate:[/cyan] {', '.join(leagues_list)}")
    console.print(f"[cyan]Bankroll iniziale:[/cyan] €{bankroll:.2f}")
    console.print(f"[cyan]Kelly fraction:[/cyan] {kelly_fraction}")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Preparazione dati...", total=100)
        
        # Scarica dati
        try:
            fbref_scraper = FBrefScraper(leagues_list, config.data.seasons)
            odds_scraper = MatchHistoryScraper(leagues_list, config.data.seasons)
            
            progress.update(task, advance=20, description="[cyan]Scaricando calendario...")
            schedule = fbref_scraper.fetch_schedule()
            
            progress.update(task, advance=20, description="[cyan]Scaricando statistiche squadre...")
            team_stats = fbref_scraper.fetch_team_stats()
            
            progress.update(task, advance=20, description="[cyan]Scaricando statistiche giocatori...")
            player_stats = fbref_scraper.fetch_player_stats()
            
            progress.update(task, advance=20, description="[cyan]Scaricando quote storiche...")
            odds_data = odds_scraper.fetch_historical_odds()
            
        except Exception as e: 
            console.print(f"[red]Errore nel download dati: {e}[/red]")
            return
        
        progress.update(task, advance=10, description="[cyan]Feature engineering...")
        
        # Prepara dati per backtest
        feature_engineer = FeatureEngineer(lookback_games=config.data.lookback_games)
        feature_engineer.set_historical_data(schedule, player_stats)
        
        # Prepara dataset con features
        processed_data = prepare_backtest_data(schedule, feature_engineer, odds_data)
        
        progress.update(task, advance=10, description="[cyan]Esecuzione backtest...")
    
    # Configura backtester
    backtester = Backtester(
        initial_bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=config.betting.min_edge
    )
    
    # Aggiungi modelli
    backtester.add_model('1X2_Result', MatchResultModel(min_accuracy=0.60))
    backtester.add_model('Exact_Score', ExactScoreModel(min_accuracy=0.15))  # Più basso per exact score
    backtester.add_model('BTTS', BTTSModel(min_accuracy=0.60))
    backtester.add_model('Over_Under_2.5', OverUnderModel(threshold=2.5, min_accuracy=0.60))
    
    # Esegui backtest
    console.print("\n[bold yellow]Esecuzione backtesting...[/bold yellow]\n")
    
    results = backtester.run_backtest(
        data=processed_data,
        odds_data=odds_data,
        train_window_days=365,
        test_window_days=30
    )
    
    # Mostra risultati
    display_backtest_results(results)
    
    # Salva risultati
    save_results(results, output)
    console.print(f"\n[green]✅ Risultati salvati in {output}[/green]")


@cli.command()
@click.option('--days', '-d', default=7, help='Numero di giorni da predire')
@click.option('--leagues', '-l', multiple=True, help='Leghe da includere')
@click.option('--markets', '-m', multiple=True, 
              type=click.Choice(['1x2', 'exact', 'btts', 'over_under', 'all']),
              default=['all'], help='Mercati da predire')
@click.option('--min-probability', '-p', default=0.55, help='Probabilità minima per mostrare predizione')
@click.option('--output', '-o', default='predictions.json', help='File output predizioni')
@click.pass_context
def predict(ctx, days, leagues, markets, min_probability, output):
    """🔮 Genera predizioni per le partite future."""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold green]⚽ Football Prediction System - Predizioni[/bold green]",
        border_style="green"
    ))
    
    leagues_list = list(leagues) if leagues else config.data.leagues
    markets_list = list(markets) if 'all' not in markets else ['1x2', 'exact', 'btts', 'over_under']
    
    console.print(f"[cyan]Leghe:[/cyan] {', '.join(leagues_list)}")
    console.print(f"[cyan]Giorni da predire:[/cyan] {days}")
    console.print(f"[cyan]Mercati:[/cyan] {', '.join(markets_list)}")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Caricamento dati...", total=100)
        
        # Scarica dati storici per training
        fbref_scraper = FBrefScraper(leagues_list, config.data.seasons)
        
        progress.update(task, advance=30, description="[cyan]Scaricando dati storici...")
        historical_data = fbref_scraper.fetch_schedule()
        team_stats = fbref_scraper.fetch_team_stats()
        player_stats = fbref_scraper.fetch_player_stats()
        
        progress.update(task, advance=20, description="[cyan]Preparando features...")
        
        # Feature engineering
        feature_engineer = FeatureEngineer(lookback_games=config.data.lookback_games)
        feature_engineer.set_historical_data(historical_data, player_stats)
        
        # Prepara dati training
        train_data = prepare_training_data(historical_data, feature_engineer)
        
        progress.update(task, advance=20, description="[cyan]Training modelli...")
        
        # Train modelli
        models = train_models(train_data, markets_list, config)
        
        progress.update(task, advance=20, description="[cyan]Generando predizioni...")
        
        # Ottieni fixtures future
        fixtures = get_upcoming_fixtures(fbref_scraper, days)
        
        # Genera predizioni
        predictions = generate_predictions(
            fixtures, models, feature_engineer, min_probability
        )
        
        progress.update(task, advance=10)
    
    # Mostra predizioni
    display_predictions(predictions, markets_list)
    
    # Salva predizioni
    with open(output, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    console.print(f"\n[green]✅ Predizioni salvate in {output}[/green]")


@cli.command()
@click.option('--bankroll', '-b', default=1000.0, help='Bankroll iniziale')
@click.option('--kelly-fraction', '-k', default=0.25, help='Frazione Kelly')
@click.option('--predictions-file', '-p', default='predictions.json', help='File predizioni')
@click.option('--odds-file', '-o', default=None, help='File quote (opzionale)')
@click.pass_context
def value_bets(ctx, bankroll, kelly_fraction, predictions_file, odds_file):
    """💰 Identifica le value bets dalle predizioni."""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold yellow]💰 Value Bets Finder[/bold yellow]",
        border_style="yellow"
    ))
    
    # Carica predizioni
    try:
        with open(predictions_file) as f:
            predictions = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File {predictions_file} non trovato. Esegui prima 'predict'.[/red]")
        return
    
    # Carica quote se fornite
    odds_data = None
    if odds_file:
        try:
            odds_data = pd.read_csv(odds_file)
        except FileNotFoundError:
            console.print(f"[yellow]File quote non trovato, usando quote simulate.[/yellow]")
    
    # Inizializza Kelly
    kelly = KellyCriterion(
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=config.betting.min_edge
    )
    
    # Trova value bets
    value_bets_list = find_value_bets(predictions, odds_data, kelly)
    
    # Mostra risultati
    display_value_bets(value_bets_list, kelly)


@cli.command()
@click.option('--file', '-f', default='bet_history.json', help='File storico scommesse')
@click.pass_context
def stats(ctx, file):
    """📈 Mostra statistiche delle scommesse."""
    console.print(Panel.fit(
        "[bold magenta]📈 Statistiche Scommesse[/bold magenta]",
        border_style="magenta"
    ))
    
    try:
        with open(file) as f:
            history = json.load(f)
    except FileNotFoundError: 
        console.print(f"[yellow]Nessuno storico trovato in {file}[/yellow]")
        return
    
    display_betting_stats(history)


# ============ Funzioni Helper ============

def prepare_backtest_data(schedule:  pd.DataFrame, 
                          feature_engineer: FeatureEngineer,
                          odds_data: pd.DataFrame) -> pd.DataFrame:
    """Prepara i dati per il backtesting con tutte le features."""
    processed_rows = []
    
    for idx, row in schedule.iterrows():
        try:
            # Calcola features
            features = feature_engineer.compute_all_features(
                home_team=row.get('home_team', ''),
                away_team=row.get('away_team', ''),
                match_date=row.get('date', datetime.now()),
                referee=row.get('referee'),
                season=row.get('season', 2024)
            )
            
            # Aggiungi info base
            features['date'] = row.get('date')
            features['home_team'] = row.get('home_team')
            features['away_team'] = row.get('away_team')
            features['match_id'] = f"{row.get('date')}_{row.get('home_team')}_{row.get('away_team')}"
            
            # Target variables
            home_goals = row.get('home_goals', row.get('score', '0-0').split('-')[0] if isinstance(row.get('score'), str) else 0)
            away_goals = row.get('away_goals', row.get('score', '0-0').split('-')[1] if isinstance(row.get('score'), str) else 0)
            
            try:
                home_goals = int(home_goals) if pd.notna(home_goals) else None
                away_goals = int(away_goals) if pd.notna(away_goals) else None
            except (ValueError, TypeError):
                home_goals, away_goals = None, None
            
            if home_goals is not None and away_goals is not None: 
                features['home_goals'] = home_goals
                features['away_goals'] = away_goals
                
                # Result (1X2)
                if home_goals > away_goals:
                    features['result'] = 'H'
                elif home_goals < away_goals:
                    features['result'] = 'A'
                else:
                    features['result'] = 'D'
                
                # Exact score
                features['exact_score'] = f"{home_goals}-{away_goals}"
                
                # BTTS
                features['btts'] = home_goals > 0 and away_goals > 0
                
                # Over/Under
                total_goals = home_goals + away_goals
                features['over_2.5'] = total_goals > 2.5
                features['over_1.5'] = total_goals > 1.5
                features['over_3.5'] = total_goals > 3.5
            
            processed_rows.append(features)
            
        except Exception as e:
            logger.debug(f"Errore processing row {idx}: {e}")
            continue
    
    return pd.DataFrame(processed_rows)


def prepare_training_data(historical_data:  pd.DataFrame,
                          feature_engineer: FeatureEngineer) -> pd.DataFrame:
    """Prepara i dati di training."""
    return prepare_backtest_data(historical_data, feature_engineer, None)


def train_models(train_data: pd.DataFrame, 
                 markets: list,
                 config:  Config) -> dict:
    """Addestra tutti i modelli richiesti."""
    models = {}
    
    feature_cols = [c for c in train_data.columns 
                   if c not in ['date', 'home_team', 'away_team', 'result', 
                               'home_goals', 'away_goals', 'exact_score',
                               'btts', 'over_2.5', 'over_1.5', 'over_3.5', 'match_id']]
    
    X = train_data[feature_cols].copy()
    
    if '1x2' in markets:
        model = MatchResultModel(min_accuracy=0.60)
        y = train_data['result'].dropna()
        X_valid = X.loc[y.index]
        if len(y) > 100:
            model.fit(X_valid, y)
            models['1x2'] = model
            console.print(f"[green]✓ Modello 1X2 - Accuracy: {model.metrics.get('accuracy_mean', 0):.2%}[/green]")
    
    if 'exact' in markets:
        model = ExactScoreModel(min_accuracy=0.10)
        y = train_data['exact_score'].dropna()
        X_valid = X.loc[y.index]
        if len(y) > 100:
            # Limita a score più comuni
            top_scores = y.value_counts().head(20).index
            mask = y.isin(top_scores)
            model.fit(X_valid[mask], y[mask])
            models['exact'] = model
            console.print(f"[green]✓ Modello Exact Score - Accuracy: {model.metrics.get('accuracy_mean', 0):.2%}[/green]")
    
    if 'btts' in markets: 
        model = BTTSModel(min_accuracy=0.60)
        y = train_data['btts'].dropna()
        X_valid = X.loc[y.index]
        if len(y) > 100:
            model.fit(X_valid, y)
            models['btts'] = model
            console.print(f"[green]✓ Modello BTTS - Accuracy: {model.metrics.get('accuracy_mean', 0):.2%}[/green]")
    
    if 'over_under' in markets:
        model = OverUnderModel(threshold=2.5, min_accuracy=0.60)
        y = train_data['over_2.5'].dropna()
        X_valid = X.loc[y.index]
        if len(y) > 100:
            model.fit(X_valid, y)
            models['over_under'] = model
            console.print(f"[green]✓ Modello Over/Under 2.5 - Accuracy: {model.metrics.get('accuracy_mean', 0):.2%}[/green]")
    
    return models


def get_upcoming_fixtures(scraper:  FBrefScraper, days: int) -> pd.DataFrame:
    """Ottiene le partite future."""
    schedule = scraper.fetch_schedule()
    today = datetime.now()
    end_date = today + timedelta(days=days)
    
    # Filtra partite future
    schedule['date'] = pd.to_datetime(schedule['date'])
    upcoming = schedule[
        (schedule['date'] >= today) & 
        (schedule['date'] <= end_date)
    ]
    
    return upcoming


def generate_predictions(fixtures:  pd.DataFrame,
                        models: dict,
                        feature_engineer: FeatureEngineer,
                        min_probability:  float) -> list:
    """Genera predizioni per le partite."""
    predictions = []
    
    for idx, row in fixtures.iterrows():
        try: 
            # Calcola features
            features = feature_engineer.compute_all_features(
                home_team=row.get('home_team', ''),
                away_team=row.get('away_team', ''),
                match_date=row.get('date', datetime.now()),
                referee=row.get('referee'),
                season=2025
            )
            
            X = pd.DataFrame([features])
            
            match_pred = {
                'date':  str(row.get('date')),
                'home_team':  row.get('home_team'),
                'away_team':  row.get('away_team'),
                'league': row.get('league'),
                'predictions': {}
            }
            
            # Predizioni per ogni modello
            for market, model in models.items():
                try: 
                    proba = model.predict_proba(X)[0]
                    classes = model.get_class_labels()
                    
                    market_preds = []
                    for i, cls in enumerate(classes):
                        if proba[i] >= min_probability: 
                            market_preds.append({
                                'selection': str(cls),
                                'probability': float(proba[i]),
                                'confidence': 'high' if proba[i] >= 0.65 else 'medium'
                            })
                    
                    if market_preds: 
                        # Ordina per probabilità
                        market_preds.sort(key=lambda x: x['probability'], reverse=True)
                        match_pred['predictions'][market] = market_preds
                        
                except Exception as e:
                    logger.debug(f"Errore predizione {market}: {e}")
                    continue
            
            if match_pred['predictions']:
                predictions.append(match_pred)
                
        except Exception as e:
            logger.debug(f"Errore processing fixture:  {e}")
            continue
    
    return predictions


def find_value_bets(predictions: list, 
                    odds_data: pd.DataFrame,
                    kelly:  KellyCriterion) -> list:
    """Trova le value bets dalle predizioni."""
    value_bets = []
    
    for match in predictions:
        for market, preds in match.get('predictions', {}).items():
            for pred in preds: 
                # Simula quote se non fornite (in produzione usare quote reali)
                implied_prob = 1 / pred['probability']
                simulated_odds = implied_prob * 0.95  # Margine bookmaker ~5%
                
                # Usa quote reali se disponibili
                actual_odds = get_actual_odds(match, market, pred['selection'], odds_data)
                if actual_odds: 
                    odds = actual_odds
                else:
                    odds = max(simulated_odds, 1.10)
                
                bet = kelly.evaluate_bet(
                    probability=pred['probability'],
                    odds=odds,
                    match_id=f"{match['date']}_{match['home_team']}_{match['away_team']}",
                    market=market,
                    selection=pred['selection']
                )
                
                if bet.is_value_bet: 
                    value_bets.append({
                        'match':  f"{match['home_team']} vs {match['away_team']}",
                        'date': match['date'],
                        'league': match.get('league', 'N/A'),
                        'market': market,
                        'selection': pred['selection'],
                        'probability': pred['probability'],
                        'odds': odds,
                        'edge': bet.expected_value,
                        'kelly_stake': bet.kelly_fraction,
                        'stake': bet.stake
                    })
    
    return sorted(value_bets, key=lambda x: x['edge'], reverse=True)


def get_actual_odds(match: dict, market: str, selection: str, 
                    odds_data: pd.DataFrame) -> float:
    """Recupera le quote reali se disponibili."""
    if odds_data is None or len(odds_data) == 0:
        return None
    # Implementazione lookup quote
    return None


def display_backtest_results(results):
    """Mostra i risultati del backtest."""
    console.print("\n")
    
    # Tabella risultati principali
    table = Table(title="📊 Risultati Backtesting", show_header=True, header_style="bold cyan")
    table.add_column("Metrica", style="cyan")
    table.add_column("Valore", style="green")
    
    table.add_row("Periodo", f"{results.start_date} → {results.end_date}")
    table.add_row("Totale Scommesse", str(results.total_bets))
    table.add_row("Vittorie / Perdite", f"{results.wins} / {results.losses}")
    table.add_row("Win Rate", f"{results.win_rate:.2%}")
    table.add_row("Bankroll Iniziale", f"€{results.initial_bankroll:.2f}")
    table.add_row("Bankroll Finale", f"€{results.final_bankroll:.2f}")
    table.add_row("Profitto Totale", f"€{results.total_profit:.2f}")
    table.add_row("ROI", f"{results.roi:.2%}")
    table.add_row("Max Drawdown", f"{results.max_drawdown:.2%}")
    table.add_row("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
    
    console.print(table)
    
    # Accuracy modelli
    if results.model_accuracies:
        console.print("\n")
        model_table = Table(title="🎯 Accuracy Modelli", show_header=True, header_style="bold magenta")
        model_table.add_column("Modello", style="magenta")
        model_table.add_column("Accuracy", style="yellow")
        model_table.add_column("Status", style="white")
        
        for model, acc in results.model_accuracies.items():
            status = "✅" if acc >= 0.60 else "⚠️"
            model_table.add_row(model, f"{acc:.2%}", status)
        
        console.print(model_table)


def display_predictions(predictions: list, markets:  list):
    """Mostra le predizioni."""
    console.print("\n")
    
    for pred in predictions[: 20]:  # Limita a 20 partite
        match_str = f"[bold]{pred['home_team']}[/bold] vs [bold]{pred['away_team']}[/bold]"
        console.print(Panel(match_str, title=f"📅 {pred['date']} | {pred.get('league', '')}"))
        
        for market, market_preds in pred.get('predictions', {}).items():
            console.print(f"  [cyan]{market.upper()}:[/cyan]")
            for p in market_preds[: 3]: 
                conf_color = "green" if p['confidence'] == 'high' else "yellow"
                console.print(f"    • {p['selection']}: [{conf_color}]{p['probability']:.2%}[/{conf_color}]")
        
        console.print("")


def display_value_bets(value_bets: list, kelly: KellyCriterion):
    """Mostra le value bets trovate."""
    if not value_bets:
        console.print("[yellow]Nessuna value bet trovata con i criteri attuali.[/yellow]")
        return
    
    table = Table(title="💰 Value Bets", show_header=True, header_style="bold yellow")
    table.add_column("Partita", style="white")
    table.add_column("Mercato", style="cyan")
    table.add_column("Selezione", style="green")
    table.add_column("Prob.", style="yellow")
    table.add_column("Quote", style="magenta")
    table.add_column("Edge", style="green")
    table.add_column("Stake", style="bold white")
    
    for bet in value_bets[: 15]: 
        table.add_row(
            bet['match'][:30],
            bet['market'],
            bet['selection'],
            f"{bet['probability']:.2%}",
            f"{bet['odds']:.2f}",
            f"{bet['edge']:.2%}",
            f"€{bet['stake']:.2f}"
        )
    
    console.print(table)
    
    # Riepilogo
    total_stake = sum(b['stake'] for b in value_bets)
    console.print(f"\n[bold]Totale stake consigliato:[/bold] €{total_stake:.2f}")
    console.print(f"[bold]Bankroll disponibile:[/bold] €{kelly.current_bankroll:.2f}")


def display_betting_stats(history: list):
    """Mostra statistiche scommesse."""
    if not history:
        console.print("[yellow]Nessuna scommessa registrata.[/yellow]")
        return
    
    df = pd.DataFrame(history)
    
    wins = len(df[df['won'] == True])
    losses = len(df) - wins
    total_stake = df['stake'].sum()
    total_profit = df['pnl'].sum()
    
    table = Table(title="📈 Statistiche Complessive", show_header=True)
    table.add_column("Metrica", style="cyan")
    table.add_column("Valore", style="green")
    
    table.add_row("Scommesse Totali", str(len(df)))
    table.add_row("Vinte / Perse", f"{wins} / {losses}")
    table.add_row("Win Rate", f"{wins/len(df):.2%}" if len(df) > 0 else "N/A")
    table.add_row("Stake Totale", f"€{total_stake:.2f}")
    table.add_row("Profitto/Perdita", f"€{total_profit:.2f}")
    table.add_row("ROI", f"{total_profit/total_stake:.2%}" if total_stake > 0 else "N/A")
    
    console.print(table)


def save_results(results, output:  str):
    """Salva i risultati su file."""
    data = {
        'start_date': str(results.start_date),
        'end_date': str(results.end_date),
        'total_bets':  results.total_bets,
        'wins': results.wins,
        'losses': results.losses,
        'win_rate': results.win_rate,
        'initial_bankroll': results.initial_bankroll,
        'final_bankroll': results.final_bankroll,
        'total_profit':  results.total_profit,
        'roi': results.roi,
        'max_drawdown': results.max_drawdown,
        'sharpe_ratio': results.sharpe_ratio,
        'model_accuracies': results.model_accuracies
    }
    
    with open(output, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()