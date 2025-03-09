import pytest
import numpy as np
from src.analysis.risk_manager import RiskManager, PositionInfo, PositionType

class TestRiskManager:
    @pytest.fixture
    def config(self):
        return {
            'MAX_RISK_PER_TRADE': 0.02,
            'MAX_TOTAL_RISK': 0.06,
            'MIN_RISK_REWARD': 2.0,
            'POSITION_SIZING_ATR': 1.5,
            'TRAILING_STOP_ACTIVATION': 0.02,
            'TRAILING_STOP_DISTANCE': 0.01,
            'MIN_POSITION_SIZE': 0.001,
            'MAX_POSITION_SIZE': 1.0,
            'CHECK_CORRELATION': True
        }
        
    @pytest.fixture
    def risk_manager(self, config):
        return RiskManager(config)
        
    @pytest.fixture
    def sample_position(self):
        return PositionInfo(
            type=PositionType.LONG,
            entry_price=100.0,
            size=1.0,
            stop_loss=95.0,
            take_profit=110.0,
            risk_reward_ratio=2.0,
            max_drawdown=0.05
        )
        
    def test_calculate_position_size(self, risk_manager):
        """Test Position Size Berechnung."""
        # Test mit normalen Werten
        size = risk_manager.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss=95.0,
            volatility=0.02
        )
        
        assert size > 0
        assert size <= risk_manager.config['MAX_POSITION_SIZE']
        
        # Test mit hoher Volatilität
        high_vol_size = risk_manager.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss=95.0,
            volatility=0.05
        )
        
        assert high_vol_size < size  # Größe sollte kleiner sein bei höherer Volatilität
        
    def test_calculate_dynamic_stops(self, risk_manager):
        """Test Dynamic Stop Loss/Take Profit Berechnung."""
        # Test für Long Position
        stop_loss, take_profit = risk_manager.calculate_dynamic_stops(
            entry_price=100.0,
            position_type=PositionType.LONG,
            atr=2.0
        )
        
        assert stop_loss < 100.0
        assert take_profit > 100.0
        assert (take_profit - 100.0) >= (100.0 - stop_loss) * risk_manager.min_risk_reward
        
        # Test mit Support/Resistance Levels
        support_resistance = {
            'support': [98.0, 95.0, 92.0],
            'resistance': [102.0, 105.0, 108.0]
        }
        
        stop_loss, take_profit = risk_manager.calculate_dynamic_stops(
            entry_price=100.0,
            position_type=PositionType.LONG,
            atr=2.0,
            support_resistance=support_resistance
        )
        
        assert stop_loss >= 98.0  # Sollte nicht unter nächstem Support sein
        assert take_profit <= 102.0  # Sollte nicht über nächster Resistance sein
        
    def test_update_trailing_stop(self, risk_manager, sample_position):
        """Test Trailing Stop Updates."""
        # Test ohne Aktivierung
        new_stop = risk_manager.update_trailing_stop(
            sample_position,
            current_price=101.0,  # Kleiner Profit
            atr=2.0
        )
        
        assert new_stop == sample_position.stop_loss  # Sollte unverändert sein
        
        # Test mit Aktivierung
        new_stop = risk_manager.update_trailing_stop(
            sample_position,
            current_price=105.0,  # Größerer Profit
            atr=2.0
        )
        
        assert new_stop > sample_position.stop_loss  # Sollte angehoben sein
        assert new_stop < 105.0  # Sollte unter aktuellem Preis sein
        
    def test_calculate_risk_metrics(self, risk_manager, sample_position):
        """Test Risiko-Metrik Berechnung."""
        positions = {
            'BTC': sample_position,
            'ETH': PositionInfo(
                type=PositionType.SHORT,
                entry_price=2000.0,
                size=0.5,
                stop_loss=2100.0,
                take_profit=1800.0,
                risk_reward_ratio=2.0,
                max_drawdown=0.05
            )
        }
        
        metrics = risk_manager.calculate_risk_metrics(
            positions,
            account_balance=10000.0
        )
        
        assert 'total_risk' in metrics
        assert 'max_drawdown' in metrics
        assert 'avg_risk_reward' in metrics
        assert metrics['position_count'] == 2
        assert metrics['total_risk'] <= risk_manager.max_total_risk
        
    def test_validate_new_position(self, risk_manager, sample_position):
        """Test Position Validierung."""
        existing_positions = {
            'BTC': sample_position
        }
        
        # Test gültige Position
        valid_position = PositionInfo(
            type=PositionType.LONG,
            entry_price=2000.0,
            size=0.5,
            stop_loss=1900.0,
            take_profit=2200.0,
            risk_reward_ratio=2.0,
            max_drawdown=0.05
        )
        
        is_valid, message = risk_manager.validate_new_position(
            valid_position,
            existing_positions,
            account_balance=10000.0
        )
        
        assert is_valid
        
        # Test ungültige Position (zu niedriges Risk/Reward)
        invalid_position = PositionInfo(
            type=PositionType.LONG,
            entry_price=2000.0,
            size=0.5,
            stop_loss=1900.0,
            take_profit=2050.0,
            risk_reward_ratio=0.5,
            max_drawdown=0.05
        )
        
        is_valid, message = risk_manager.validate_new_position(
            invalid_position,
            existing_positions,
            account_balance=10000.0
        )
        
        assert not is_valid
        assert "Risk-reward ratio" in message
        
    def test_calculate_value_at_risk(self, risk_manager, sample_position):
        """Test VaR Berechnung."""
        positions = {
            'BTC': sample_position
        }
        
        # Generiere historische Preisdaten
        np.random.seed(42)
        price_history = {
            'BTC': np.random.normal(100.0, 2.0, 1000)  # 1000 Tage Historie
        }
        
        var = risk_manager.calculate_value_at_risk(
            positions,
            price_history,
            confidence_level=0.95
        )
        
        assert var > 0
        assert var < sample_position.entry_price * sample_position.size  # VaR sollte kleiner als Position sein
        
    def test_edge_cases(self, risk_manager):
        """Test Verhalten bei Grenzfällen."""
        # Test mit sehr kleinem Account
        size = risk_manager.calculate_position_size(
            account_balance=100.0,
            entry_price=100.0,
            stop_loss=95.0,
            volatility=0.02
        )
        
        assert size >= risk_manager.config['MIN_POSITION_SIZE']
        
        # Test mit sehr großem Stop Loss
        size = risk_manager.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss=50.0,
            volatility=0.02
        )
        
        assert size <= risk_manager.config['MAX_POSITION_SIZE']
        
        # Test mit leeren Positionen
        metrics = risk_manager.calculate_risk_metrics({}, 10000.0)
        assert metrics['total_risk'] == 0.0
        assert metrics['position_count'] == 0
        
    def test_performance(self, risk_manager):
        """Test Performance bei vielen Positionen."""
        # Generiere viele Positionen
        positions = {}
        for i in range(100):
            positions[f'COIN_{i}'] = PositionInfo(
                type=PositionType.LONG,
                entry_price=100.0,
                size=1.0,
                stop_loss=95.0,
                take_profit=110.0,
                risk_reward_ratio=2.0,
                max_drawdown=0.05
            )
            
        import time
        
        # Zeitmessung für Risiko-Berechnung
        start_time = time.time()
        metrics = risk_manager.calculate_risk_metrics(positions, 1000000.0)
        calc_time = time.time() - start_time
        
        assert calc_time < 1.0  # Sollte unter 1 Sekunde sein 