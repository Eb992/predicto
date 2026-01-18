"""Modulo scrapers per download dati con supporto Tor."""

from . base_scraper import (
    BaseScraper,
    FBrefScraper,
    UnderstatScraper,
    MatchHistoryScraper,
    WhoScoredScraper,
    FotMobScraper,
    SofascoreScraper,
    create_scraper,
    TOR_PROXY,
    TOR_DAEMON_PROXY
)
from .sofifa_scraper import SoFIFAScraper

__all__ = [
    # Base
    "BaseScraper",
    "create_scraper",
    
    # Scrapers
    "FBrefScraper",
    "UnderstatScraper", 
    "MatchHistoryScraper",
    "WhoScoredScraper",
    "FotMobScraper",
    "SofascoreScraper",
    "SoFIFAScraper",
    
    # Costanti Tor
    "TOR_PROXY",
    "TOR_DAEMON_PROXY"
]