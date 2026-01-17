"""Test per Kelly Criterion."""

import pytest

from football_predictor.betting import Bet, KellyCriterion


class TestKellyCriterion:
    """Test suite per KellyCriterion."""

    def setup_method(self):
        """Setup per ogni test."""
        self.kelly = KellyCriterion(
            bankroll=1000.0, kelly_fraction=0.25, max_stake_pct=0.05, min_edge=0.05
        )

    def test_calculate_edge_positive(self):
        """Test edge positivo."""
        edge = self.kelly.calculate_edge(probability=0.55, odds=2.0)
        assert edge == pytest.approx(0.10, rel=0.01)

    def test_calculate_edge_negative(self):
        """Test edge negativo."""
        edge = self.kelly.calculate_edge(probability=0.40, odds=2.0)
        assert edge == pytest.approx(-0.20, rel=0.01)

    def test_calculate_kelly_stake_value_bet(self):
        """Test stake Kelly per value bet."""
        stake = self.kelly.calculate_kelly_stake(probability=0.60, odds=2.0)
        assert stake > 0
        assert stake <= self.kelly.max_stake_pct

    def test_calculate_kelly_stake_no_value(self):
        """Test stake Kelly per non-value bet."""
        stake = self.kelly.calculate_kelly_stake(probability=0.40, odds=2.0)
        assert stake == 0.0

    def test_evaluate_bet_value(self):
        """Test valutazione value bet."""
        bet = self.kelly.evaluate_bet(
            probability=0.60, odds=2.0, match_id="test_match", market="1X2", selection="H"
        )
        assert bet.is_value_bet == True
        assert bet.stake > 0
        assert bet.expected_value > self.kelly.min_edge

    def test_evaluate_bet_no_value(self):
        """Test valutazione non-value bet."""
        bet = self.kelly.evaluate_bet(
            probability=0.45, odds=2.0, match_id="test_match", market="1X2", selection="H"
        )
        assert bet.is_value_bet == False
        assert bet.stake == 0

    def test_place_bet_win(self):
        """Test piazzamento scommessa vincente."""
        bet = Bet(
            match_id="test",
            market="1X2",
            selection="H",
            probability=0.60,
            odds=2.0,
            stake=50.0,
            is_value_bet=True,
            expected_value=0.10,
            kelly_fraction=0.05,
        )

        initial = self.kelly.current_bankroll
        self.kelly.place_bet(bet, won=True)

        assert self.kelly.current_bankroll == initial + 50.0  # stake * (odds - 1)

    def test_place_bet_loss(self):
        """Test piazzamento scommessa perdente."""
        bet = Bet(
            match_id="test",
            market="1X2",
            selection="H",
            probability=0.60,
            odds=2.0,
            stake=50.0,
            is_value_bet=True,
            expected_value=0.10,
            kelly_fraction=0.05,
        )

        initial = self.kelly.current_bankroll
        self.kelly.place_bet(bet, won=False)

        assert self.kelly.current_bankroll == initial - 50.0

    def test_get_statistics(self):
        """Test statistiche."""
        # Simula alcune scommesse
        for i in range(10):
            bet = Bet(
                match_id=f"match_{i}",
                market="1X2",
                selection="H",
                probability=0.60,
                odds=2.0,
                stake=50.0,
                is_value_bet=True,
                expected_value=0.10,
                kelly_fraction=0.05,
            )
            self.kelly.place_bet(bet, won=(i % 2 == 0))

        stats = self.kelly.get_statistics()

        assert stats["total_bets"] == 10
        assert stats["wins"] == 5
        assert stats["losses"] == 5
        assert stats["win_rate"] == 0.5


class TestBet:
    """Test per classe Bet."""

    def test_bet_creation(self):
        """Test creazione Bet."""
        bet = Bet(
            match_id="2024-01-15_TeamA_TeamB",
            market="1X2",
            selection="H",
            probability=0.55,
            odds=2.10,
            stake=25.0,
            is_value_bet=True,
            expected_value=0.155,
            kelly_fraction=0.03,
        )

        assert bet.match_id == "2024-01-15_TeamA_TeamB"
        assert bet.market == "1X2"
        assert bet.stake == 25.0
        assert bet.is_value_bet == True
