"""Scraper per SoFIFA (rating giocatori FIFA) con supporto Tor."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import soccerdata as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class SoFIFAScraper:
    """Scraper per dati SoFIFA (rating FIFA dei giocatori) con Tor."""

    # Mapping leghe -> ID SoFIFA
    LEAGUE_MAPPING = {
        "ENG-Premier League": 13,
        "ITA-Serie A": 31,
        "ESP-La Liga": 53,
        "GER-Bundesliga": 19,
        "FRA-Ligue 1": 16,
        "NED-Eredivisie": 10,
        "POR-Primeira Liga": 308,
        "ENG-Championship": 14,
    }

    def __init__(
        self, leagues: List[str], seasons: List[int], use_tor: bool = True, tor_port: int = 9150
    ):
        """
        Inizializza lo scraper SoFIFA.

        Args:
            leagues: Lista delle leghe da scaricare
            seasons: Lista delle stagioni
            use_tor: Se usare Tor come proxy
            tor_port:  Porta Tor (9150 per Tor Browser, 9050 per daemon)
        """
        self.leagues = leagues
        self.seasons = seasons
        self.use_tor = use_tor
        self.tor_port = tor_port
        self._cache = {}

        # Configura proxy
        if use_tor:
            self.proxy = f"socks5://127.0.0.1:{tor_port}"
            logger.info(f"SoFIFA:  usando Tor proxy su porta {tor_port}")
        else:
            self.proxy = None

        self.scrapers = {}
        if sd is not None:
            for league in leagues:
                if league in self.LEAGUE_MAPPING:
                    try:
                        self.scrapers[league] = sd.SoFIFA(
                            leagues=league, seasons=seasons, proxy=self.proxy
                        )
                        logger.info(f"Inizializzato SoFIFA scraper per {league}")
                    except Exception as e:
                        logger.warning(f"Impossibile inizializzare SoFIFA per {league}: {e}")

    def fetch_player_ratings(self) -> pd.DataFrame:
        """
        Scarica i rating FIFA dei giocatori.

        Returns:
            DataFrame con rating giocatori
        """
        all_data = []

        for league, scraper in self.scrapers.items():
            try:
                players = scraper.read_players()
                players["league"] = league
                all_data.append(players)
                logger.info(f"Scaricati {len(players)} giocatori da {league}")
            except Exception as e:
                logger.warning(f"Errore download SoFIFA {league}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def fetch_team_ratings(self) -> pd.DataFrame:
        """
        Scarica i rating FIFA delle squadre.

        Returns:
            DataFrame con rating squadre
        """
        all_data = []

        for league, scraper in self.scrapers.items():
            try:
                teams = scraper.read_teams()
                teams["league"] = league
                all_data.append(teams)
                logger.info(f"Scaricati {len(teams)} squadre da {league}")
            except Exception as e:
                logger.warning(f"Errore download team ratings {league}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def get_player_rating(self, player_name: str, team: str = None) -> Dict[str, float]:
        """
        Ottieni il rating di un giocatore specifico.

        Args:
            player_name: Nome del giocatore
            team: Nome della squadra (opzionale, per disambiguazione)

        Returns:
            Dict con rating e attributi del giocatore
        """
        cache_key = f"{player_name}_{team}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        players_df = self.fetch_player_ratings()

        if len(players_df) == 0:
            return self._default_player_rating()

        mask = players_df["player"].str.contains(player_name, case=False, na=False)

        if team:
            mask &= players_df["team"].str.contains(team, case=False, na=False)

        matches = players_df[mask]

        if len(matches) == 0:
            return self._default_player_rating()

        player = matches.iloc[0]

        result = {
            "overall_rating": player.get("overall", 70),
            "potential": player.get("potential", 75),
            "pace": player.get("pace", 70),
            "shooting": player.get("shooting", 65),
            "passing": player.get("passing", 65),
            "dribbling": player.get("dribbling", 65),
            "defending": player.get("defending", 50),
            "physical": player.get("physical", 70),
            "value": player.get("value", 0),
            "wage": player.get("wage", 0),
        }

        self._cache[cache_key] = result
        return result

    def get_team_overall_rating(self, team: str) -> Dict[str, float]:
        """
        Calcola il rating complessivo di una squadra.

        Args:
            team: Nome della squadra

        Returns:
            Dict con rating aggregati della squadra
        """
        teams_df = self.fetch_team_ratings()

        if len(teams_df) == 0:
            return self._default_team_rating()

        mask = teams_df["team"].str.contains(team, case=False, na=False)
        matches = teams_df[mask]

        if len(matches) == 0:
            return self._default_team_rating()

        team_data = matches.iloc[0]

        return {
            "overall": team_data.get("overall", 75),
            "attack": team_data.get("attack", 75),
            "midfield": team_data.get("midfield", 75),
            "defence": team_data.get("defence", 75),
            "transfer_budget": team_data.get("transfer_budget", 0),
            "club_worth": team_data.get("club_worth", 0),
        }

    def get_squad_depth_analysis(self, team: str) -> Dict[str, any]:
        """
        Analizza la profondità della rosa.

        Args:
            team:  Nome della squadra

        Returns:
            Dict con analisi della rosa
        """
        players_df = self.fetch_player_ratings()

        if len(players_df) == 0:
            return self._default_squad_analysis()

        mask = players_df["team"].str.contains(team, case=False, na=False)
        squad = players_df[mask]

        if len(squad) == 0:
            return self._default_squad_analysis()

        return {
            "squad_size": len(squad),
            "avg_rating": squad.get("overall", pd.Series([70])).mean(),
            "avg_age": squad.get("age", pd.Series([26])).mean(),
            "star_players": len(squad[squad.get("overall", 0) >= 85]),
            "young_talents": len(
                squad[(squad.get("age", 30) <= 21) & (squad.get("potential", 0) >= 80)]
            ),
            "max_rating": squad.get("overall", pd.Series([70])).max(),
            "min_rating": squad.get("overall", pd.Series([70])).min(),
        }

    def compare_teams(self, team1: str, team2: str) -> Dict[str, float]:
        """
        Confronta due squadre basandosi sui rating FIFA.

        Args:
            team1: Prima squadra
            team2: Seconda squadra

        Returns:
            Dict con confronto rating
        """
        rating1 = self.get_team_overall_rating(team1)
        rating2 = self.get_team_overall_rating(team2)

        return {
            "overall_diff": rating1["overall"] - rating2["overall"],
            "attack_diff": rating1["attack"] - rating2["attack"],
            "midfield_diff": rating1["midfield"] - rating2["midfield"],
            "defence_diff": rating1["defence"] - rating2["defence"],
            "team1_advantage": rating1["overall"] > rating2["overall"],
            "rating_gap": abs(rating1["overall"] - rating2["overall"]),
        }

    def _default_player_rating(self) -> Dict[str, float]:
        """Rating di default per giocatori non trovati."""
        return {
            "overall_rating": 70,
            "potential": 72,
            "pace": 70,
            "shooting": 65,
            "passing": 65,
            "dribbling": 65,
            "defending": 50,
            "physical": 70,
            "value": 0,
            "wage": 0,
        }

    def _default_team_rating(self) -> Dict[str, float]:
        """Rating di default per squadre non trovate."""
        return {
            "overall": 75,
            "attack": 75,
            "midfield": 75,
            "defence": 75,
            "transfer_budget": 0,
            "club_worth": 0,
        }

    def _default_squad_analysis(self) -> Dict[str, any]:
        """Analisi rosa di default."""
        return {
            "squad_size": 25,
            "avg_rating": 70,
            "avg_age": 26,
            "star_players": 0,
            "young_talents": 0,
            "max_rating": 75,
            "min_rating": 65,
        }
