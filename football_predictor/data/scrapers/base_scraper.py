"""Scraper base che utilizza soccerdata con supporto Tor."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Callable
import pandas as pd
import soccerdata as sd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Configurazione Tor
TOR_PROXY = "socks5://127.0.0.1:9150"  # Porta Tor Browser
TOR_DAEMON_PROXY = "socks5://127.0.0.1:9050"  # Porta Tor daemon (alternativa)


class BaseScraper(ABC):
    """Classe base per tutti gli scraper."""
    
    def __init__(self, 
                 leagues: List[str], 
                 seasons: List[int],
                 use_tor: bool = True,
                 tor_port: int = 9150):
        """
        Inizializza lo scraper.
        
        Args: 
            leagues: Lista delle leghe da scaricare
            seasons:  Lista delle stagioni
            use_tor:  Se usare Tor come proxy
            tor_port: Porta Tor (9150 per Tor Browser, 9050 per daemon)
        """
        self.leagues = leagues
        self. seasons = seasons
        self.use_tor = use_tor
        self.tor_port = tor_port
        self._cache = {}
        
        # Configura proxy
        if use_tor:
            self. proxy = f"socks5://127.0.0.1:{tor_port}"
            logger.info(f"Usando Tor proxy su porta {tor_port}")
        else: 
            self.proxy = None
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """Recupera i dati dalla fonte."""
        pass
    
    def _normalize_team_names(self, df: pd.DataFrame, col:  str) -> pd.DataFrame:
        """Normalizza i nomi delle squadre."""
        return df


class FBrefScraper(BaseScraper):
    """Scraper per FBref utilizzando soccerdata con Tor."""
    
    def __init__(self, 
                 leagues:  List[str], 
                 seasons:  List[int],
                 use_tor: bool = True,
                 tor_port: int = 9150):
        """
        Inizializza lo scraper FBref.
        
        Args: 
            leagues: Lista delle leghe
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy (default: True)
            tor_port:  Porta Tor (default: 9150 per Tor Browser)
        """
        super().__init__(leagues, seasons, use_tor, tor_port)
        
        self.scrapers = {}
        for league in leagues: 
            try:
                # FBref usa requests, quindi passiamo il proxy
                self. scrapers[league] = sd.FBref(
                    leagues=league,
                    seasons=seasons,
                    proxy=self.proxy if use_tor else None
                )
                logger.info(f"Inizializzato FBref scraper per {league}")
            except Exception as e:
                logger.warning(f"Errore inizializzazione FBref per {league}: {e}")
    
    def fetch_schedule(self, force_cache: bool = False) -> pd.DataFrame:
        """Recupera il calendario delle partite."""
        all_data = []
        for league, scraper in self. scrapers.items():
            try: 
                schedule = scraper.read_schedule(force_cache=force_cache)
                schedule['league'] = league
                all_data.append(schedule)
                logger.info(f"Scaricato calendario {league}:  {len(schedule)} partite")
            except Exception as e: 
                logger.warning(f"Errore scaricamento calendario {league}: {e}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def fetch_team_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """
        Recupera statistiche squadra per tipo.
        
        Args:
            stat_type: Tipo di statistiche.  Opzioni: 
                - 'standard'
                - 'shooting'
                - 'passing'
                - 'passing_types'
                - 'goal_shot_creation'
                - 'defense'
                - 'possession'
                - 'playing_time'
                - 'misc'
                - 'keeper'
                - 'keeper_adv'
        """
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                stats = scraper. read_team_season_stats(stat_type=stat_type)
                all_data.append(stats)
                logger.info(f"Scaricate stats {stat_type} per {league}")
            except Exception as e:
                logger.warning(f"Errore stats {stat_type} {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_player_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """
        Recupera statistiche giocatori.
        
        Args:
            stat_type: Tipo di statistiche (come fetch_team_stats)
        """
        all_data = []
        for league, scraper in self. scrapers.items():
            try: 
                stats = scraper.read_player_season_stats(stat_type=stat_type)
                all_data.append(stats)
                logger.info(f"Scaricate player stats {stat_type} per {league}")
            except Exception as e:
                logger.warning(f"Errore player stats {stat_type} {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_player_match_stats(self, stat_type: str = "summary",
                                  match_id: Optional[str] = None) -> pd.DataFrame:
        """
        Recupera statistiche giocatori per partita.
        
        Args:
            stat_type: Tipo di statistiche
            match_id: ID partita specifico (opzionale)
        """
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                stats = scraper.read_player_match_stats(
                    stat_type=stat_type,
                    match_id=match_id
                )
                all_data.append(stats)
            except Exception as e: 
                logger.warning(f"Errore player match stats {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_shooting_events(self, match_id: Optional[str] = None) -> pd.DataFrame:
        """Recupera eventi di tiro."""
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                shots = scraper.read_shot_events(match_id=match_id)
                all_data.append(shots)
            except Exception as e:
                logger.warning(f"Errore shot events {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_data(self) -> pd.DataFrame:
        """Implementazione metodo astratto."""
        return self.fetch_schedule()


class UnderstatScraper(BaseScraper):
    """Scraper per Understat (xG e statistiche avanzate) con Tor."""
    
    # Mapping leghe soccerdata -> Understat
    LEAGUE_MAPPING = {
        "ENG-Premier League": "ENG-Premier League",
        "ITA-Serie A": "ITA-Serie A",
        "ESP-La Liga": "ESP-La Liga",
        "GER-Bundesliga": "GER-Bundesliga",
        "FRA-Ligue 1":  "FRA-Ligue 1",
    }
    
    def __init__(self, 
                 leagues: List[str], 
                 seasons: List[int],
                 use_tor: bool = True,
                 tor_port: int = 9150):
        """
        Inizializza lo scraper Understat.
        
        Args:
            leagues: Lista delle leghe
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy
            tor_port: Porta Tor
        """
        super().__init__(leagues, seasons, use_tor, tor_port)
        
        self.scrapers = {}
        for league in leagues: 
            if league in self. LEAGUE_MAPPING: 
                try:
                    self.scrapers[league] = sd. Understat(
                        leagues=self.LEAGUE_MAPPING[league],
                        seasons=seasons,
                        proxy=self.proxy if use_tor else None
                    )
                    logger.info(f"Inizializzato Understat scraper per {league}")
                except Exception as e: 
                    logger.warning(f"Errore inizializzazione Understat per {league}: {e}")
    
    def fetch_team_stats(self) -> pd.DataFrame:
        """Recupera statistiche squadre con xG."""
        all_data = []
        for league, scraper in self.scrapers. items():
            try:
                stats = scraper.read_team_season_stats()
                stats['league'] = league
                all_data.append(stats)
            except Exception as e: 
                logger.warning(f"Errore Understat team stats {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_player_stats(self) -> pd.DataFrame:
        """Recupera statistiche giocatori con xG."""
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                stats = scraper. read_player_season_stats()
                stats['league'] = league
                all_data.append(stats)
            except Exception as e:
                logger.warning(f"Errore Understat player stats {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd. DataFrame()
    
    def fetch_match_stats(self) -> pd.DataFrame:
        """Recupera statistiche partite con xG."""
        all_data = []
        for league, scraper in self. scrapers.items():
            try: 
                stats = scraper.read_game_stats()
                stats['league'] = league
                all_data.append(stats)
            except Exception as e:
                logger.warning(f"Errore Understat match stats {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd. DataFrame()
    
    def fetch_data(self) -> pd.DataFrame:
        """Implementazione metodo astratto."""
        return self. fetch_team_stats()


class MatchHistoryScraper(BaseScraper):
    """Scraper per Football-Data. co.uk (quote storiche) con Tor."""
    
    def __init__(self, 
                 leagues:  List[str], 
                 seasons:  List[int],
                 use_tor: bool = True,
                 tor_port: int = 9150):
        """
        Inizializza lo scraper MatchHistory.
        
        Args:
            leagues: Lista delle leghe
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy
            tor_port: Porta Tor
        """
        super().__init__(leagues, seasons, use_tor, tor_port)
        
        self.scrapers = {}
        for league in leagues: 
            try: 
                self.scrapers[league] = sd.MatchHistory(
                    leagues=league,
                    seasons=seasons,
                    proxy=self.proxy if use_tor else None
                )
                logger. info(f"Inizializzato MatchHistory scraper per {league}")
            except Exception as e: 
                logger.warning(f"MatchHistory non disponibile per {league}:  {e}")
    
    def fetch_historical_odds(self) -> pd.DataFrame:
        """
        Recupera quote storiche e risultati. 
        
        Colonne quote disponibili (dipende dalla lega):
        - B365H, B365D, B365A:  Bet365
        - BWH, BWD, BWA: Betway
        - IWH, IWD, IWA:  Interwetten
        - PSH, PSD, PSA:  Pinnacle
        - WHH, WHD, WHA: William Hill
        - VCH, VCD, VCA: VC Bet
        - MaxH, MaxD, MaxA: Quote massime
        - AvgH, AvgD, AvgA: Quote medie
        
        Per Over/Under 2.5:
        - B365>2.5, B365<2.5
        - P>2.5, P<2.5
        
        Per Asian Handicap:
        - AHh, AHCh, AHCa
        """
        all_data = []
        for league, scraper in self.scrapers. items():
            try:
                history = scraper.read_games()
                history['league'] = league
                all_data.append(history)
                logger.info(f"Scaricate {len(history)} partite storiche per {league}")
            except Exception as e:
                logger.warning(f"Errore odds {league}: {e}")
        
        if not all_data: 
            return pd. DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Normalizza nomi colonne quote
        df = self._normalize_odds_columns(df)
        
        return df
    
    def _normalize_odds_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizza i nomi delle colonne delle quote."""
        column_mapping = {
            'B365H': 'odds_home_win_b365',
            'B365D': 'odds_draw_b365',
            'B365A': 'odds_away_win_b365',
            'PSH': 'odds_home_win_pinnacle',
            'PSD': 'odds_draw_pinnacle',
            'PSA': 'odds_away_win_pinnacle',
            'MaxH': 'odds_home_win_max',
            'MaxD': 'odds_draw_max',
            'MaxA': 'odds_away_win_max',
            'AvgH':  'odds_home_win_avg',
            'AvgD': 'odds_draw_avg',
            'AvgA': 'odds_away_win_avg',
            'B365>2.5': 'odds_over_2.5_b365',
            'B365<2.5': 'odds_under_2.5_b365',
            'P>2.5': 'odds_over_2.5_pinnacle',
            'P<2.5': 'odds_under_2.5_pinnacle',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        return df
    
    def fetch_data(self) -> pd.DataFrame:
        """Implementazione metodo astratto."""
        return self.fetch_historical_odds()


class WhoScoredScraper(BaseScraper):
    """
    Scraper per WhoScored (richiede Selenium + Tor).
    
    NOTA: WhoScored usa protezione Incapsula, richiede browser reale.
    """
    
    def __init__(self, 
                 leagues: List[str], 
                 seasons: List[int],
                 use_tor: bool = True,
                 tor_port: int = 9150,
                 headless: bool = True,
                 path_to_browser:  Optional[str] = None):
        """
        Inizializza lo scraper WhoScored.
        
        Args: 
            leagues: Lista delle leghe
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy
            tor_port:  Porta Tor
            headless: Se eseguire browser in modalità headless
            path_to_browser: Path al browser Chrome/Chromium
        """
        super().__init__(leagues, seasons, use_tor, tor_port)
        
        self.headless = headless
        self.path_to_browser = path_to_browser
        self.scrapers = {}
        
        for league in leagues:
            try: 
                # WhoScored richiede Selenium
                self.scrapers[league] = sd.WhoScored(
                    leagues=league,
                    seasons=seasons,
                    proxy=self.proxy if use_tor else None,
                    headless=headless,
                    path_to_browser=path_to_browser
                )
                logger. info(f"Inizializzato WhoScored scraper per {league}")
            except Exception as e:
                logger. warning(f"Errore inizializzazione WhoScored per {league}: {e}")
    
    def fetch_schedule(self) -> pd.DataFrame:
        """Recupera calendario partite."""
        all_data = []
        for league, scraper in self. scrapers.items():
            try: 
                schedule = scraper.read_schedule()
                all_data.append(schedule)
            except Exception as e: 
                logger.warning(f"Errore WhoScored schedule {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_match_events(self, match_id: Optional[int] = None) -> pd.DataFrame:
        """Recupera eventi partita (passaggi, tiri, etc.)."""
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                events = scraper. read_events(match_id=match_id)
                all_data. append(events)
            except Exception as e:
                logger. warning(f"Errore WhoScored events {league}:  {e}")
        
        return pd. concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_data(self) -> pd.DataFrame:
        """Implementazione metodo astratto."""
        return self. fetch_schedule()
    
    def close(self):
        """Chiude i browser Selenium."""
        for scraper in self. scrapers.values():
            try:
                if hasattr(scraper, '_driver'):
                    scraper._driver. quit()
            except Exception: 
                pass


class FotMobScraper(BaseScraper):
    """Scraper per FotMob con Tor."""
    
    def __init__(self, 
                 leagues: List[str], 
                 seasons: List[int],
                 use_tor:  bool = True,
                 tor_port: int = 9150):
        """
        Inizializza lo scraper FotMob. 
        
        Args: 
            leagues: Lista delle leghe
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy
            tor_port:  Porta Tor
        """
        super().__init__(leagues, seasons, use_tor, tor_port)
        
        self.scrapers = {}
        for league in leagues:
            try:
                self.scrapers[league] = sd.FotMob(
                    leagues=league,
                    seasons=seasons,
                    proxy=self. proxy if use_tor else None
                )
                logger.info(f"Inizializzato FotMob scraper per {league}")
            except Exception as e: 
                logger.warning(f"Errore inizializzazione FotMob per {league}: {e}")
    
    def fetch_schedule(self) -> pd.DataFrame:
        """Recupera calendario partite."""
        all_data = []
        for league, scraper in self. scrapers.items():
            try: 
                schedule = scraper.read_schedule()
                all_data.append(schedule)
            except Exception as e:
                logger.warning(f"Errore FotMob schedule {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_team_stats(self, stat_type: str = "Top stats") -> pd.DataFrame:
        """
        Recupera statistiche squadre.
        
        Args:
            stat_type:  Tipo stats.  Opzioni: 
                - 'Top stats'
                - 'Shots'
                - 'Expected goals (xG)'
                - 'Passes'
                - 'Defence'
                - 'Duels'
                - 'Discipline'
        """
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                stats = scraper.read_team_match_stats(stat_type=stat_type)
                all_data.append(stats)
            except Exception as e:
                logger.warning(f"Errore FotMob team stats {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def fetch_data(self) -> pd.DataFrame:
        """Implementazione metodo astratto."""
        return self. fetch_schedule()


class SofascoreScraper(BaseScraper):
    """Scraper per Sofascore con Tor."""
    
    def __init__(self, 
                 leagues: List[str], 
                 seasons: List[int],
                 use_tor: bool = True,
                 tor_port: int = 9150):
        """
        Inizializza lo scraper Sofascore.
        
        Args:
            leagues: Lista delle leghe
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy
            tor_port: Porta Tor
        """
        super().__init__(leagues, seasons, use_tor, tor_port)
        
        self.scrapers = {}
        for league in leagues: 
            try: 
                self.scrapers[league] = sd.Sofascore(
                    leagues=league,
                    seasons=seasons,
                    proxy=self.proxy if use_tor else None
                )
                logger.info(f"Inizializzato Sofascore scraper per {league}")
            except Exception as e:
                logger.warning(f"Errore inizializzazione Sofascore per {league}:  {e}")
    
    def fetch_schedule(self) -> pd.DataFrame:
        """Recupera calendario partite."""
        all_data = []
        for league, scraper in self.scrapers.items():
            try:
                schedule = scraper.read_schedule()
                all_data.append(schedule)
            except Exception as e:
                logger.warning(f"Errore Sofascore schedule {league}: {e}")
        
        return pd.concat(all_data) if all_data else pd. DataFrame()
    
    def fetch_data(self) -> pd.DataFrame:
        """Implementazione metodo astratto."""
        return self. fetch_schedule()


# Factory function per creare scraper con Tor configurato
def create_scraper(source: str,
                   leagues: List[str],
                   seasons: List[int],
                   use_tor: bool = True,
                   tor_port: int = 9150,
                   **kwargs) -> BaseScraper:
    """
    Factory per creare scraper configurati.
    
    Args:
        source: Nome della fonte dati ('fbref', 'understat', 'matchhistory', etc.)
        leagues: Lista delle leghe
        seasons: Lista delle stagioni
        use_tor: Se usare Tor
        tor_port:  Porta Tor (9150 per Tor Browser, 9050 per daemon)
        **kwargs:  Argomenti aggiuntivi per lo scraper specifico
    
    Returns:
        Istanza dello scraper configurato
    
    Example:
        >>> scraper = create_scraper(
        ...     source='fbref',
        ...     leagues=['ENG-Premier League', 'ITA-Serie A'],
        ...     seasons=[2023, 2024],
        ...     use_tor=True,
        ...     tor_port=9150
        ... )
        >>> schedule = scraper.fetch_schedule()
    """
    scrapers = {
        'fbref': FBrefScraper,
        'understat': UnderstatScraper,
        'matchhistory':  MatchHistoryScraper,
        'whoscored': WhoScoredScraper,
        'fotmob': FotMobScraper,
        'sofascore': SofascoreScraper,
    }
    
    source = source.lower()
    if source not in scrapers:
        raise ValueError(f"Fonte non supportata: {source}. Opzioni: {list(scrapers.keys())}")
    
    return scrapers[source](
        leagues=leagues,
        seasons=seasons,
        use_tor=use_tor,
        tor_port=tor_port,
        **kwargs
    )