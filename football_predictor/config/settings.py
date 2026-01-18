"""Configurazioni globali del sistema."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TorConfig:
    """Configurazione Tor proxy."""

    enabled: bool = True
    port: int = 9150  # Porta Tor Browser (9050 per Tor daemon)
    host: str = "127.0.0.1"

    @property
    def proxy_url(self) -> str:
        """Ritorna l'URL del proxy SOCKS5."""
        return f"socks5://{self.host}:{self.port}"

    @property
    def is_browser(self) -> bool:
        """True se usa Tor Browser (porta 9150)."""
        return self.port == 9150


@dataclass
class ModelConfig:
    """Configurazione modelli."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    min_accuracy_threshold: float = 0.60  # 60% minimo richiesto


@dataclass
class BettingConfig:
    """Configurazione betting."""

    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.25  # Kelly frazionato (più conservativo)
    max_stake_pct: float = 0.05  # Max 5% del bankroll per scommessa
    min_edge: float = 0.05  # Edge minimo per piazzare scommessa
    stop_loss_pct: float = 0.50  # Stop se perdi 50%
    profit_target_pct: float = 1.00  # Target raddoppio


@dataclass
class DataConfig:
    """Configurazione dati."""

    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    leagues: List[str] = field(
        default_factory=lambda: [
            "ENG-Premier League",
            "ITA-Serie A",
            "ESP-La Liga",
            "GER-Bundesliga",
            "FRA-Ligue 1",
        ]
    )
    seasons: List[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    lookback_games: int = 10  # Partite precedenti per features forma


@dataclass
class Config:
    """Configurazione principale."""

    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tor: TorConfig = field(default_factory=TorConfig)

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Config":
        """Carica configurazione da file YAML."""
        config_path = Path(path)

        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            return cls(
                model=ModelConfig(**data.get("model", {})),
                betting=BettingConfig(**data.get("betting", {})),
                data=DataConfig(
                    **{
                        k: (Path(v) if k == "cache_dir" else v)
                        for k, v in data.get("data", {}).items()
                    }
                ),
                tor=TorConfig(**data.get("tor", {})),
            )

        return cls()

    def save(self, path: str = "config/config.yaml"):
        """Salva configurazione su file YAML."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": {
                "random_state": self.model.random_state,
                "test_size": self.model.test_size,
                "cv_folds": self.model.cv_folds,
                "min_accuracy_threshold": self.model.min_accuracy_threshold,
            },
            "betting": {
                "initial_bankroll": self.betting.initial_bankroll,
                "kelly_fraction": self.betting.kelly_fraction,
                "max_stake_pct": self.betting.max_stake_pct,
                "min_edge": self.betting.min_edge,
                "stop_loss_pct": self.betting.stop_loss_pct,
                "profit_target_pct": self.betting.profit_target_pct,
            },
            "data": {
                "cache_dir": str(self.data.cache_dir),
                "leagues": self.data.leagues,
                "seasons": self.data.seasons,
                "lookback_games": self.data.lookback_games,
            },
            "tor": {
                "enabled": self.tor.enabled,
                "port": self.tor.port,
                "host": self.tor.host,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
