import pytest
import numpy as np
from src.analysis.indicators import TechnicalIndicators, TrendDirection, CandlestickPattern

class TestTechnicalIndicators:
    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()
        
    @pytest.fixture
    def sample_data(self):
        # Generiere Testdaten
        np.random.seed(42)
        return {
            'close': np.random.random(100) * 100,
            'high': np.random.random(100) * 110,
            'low': np.random.random(100) * 90,
            'volume': np.random.random(100) * 1000
        }
        
    def test_calculate_ma(self, indicators, sample_data):
        """Test Moving Average Berechnung."""
        ma = TechnicalIndicators.calculate_ma(sample_data['close'], period=20)
        
        assert len(ma) == len(sample_data['close'])
        assert np.isnan(ma[0])  # Erste Werte sollten NaN sein
        assert not np.isnan(ma[-1])  # Letzte Werte sollten berechnet sein
        
    def test_calculate_rsi(self, indicators, sample_data):
        """Test RSI Berechnung."""
        rsi = TechnicalIndicators.calculate_rsi(sample_data['close'], period=14)
        
        assert len(rsi) == len(sample_data['close'])
        assert np.all((rsi[~np.isnan(rsi)] >= 0) & (rsi[~np.isnan(rsi)] <= 100))
        
    def test_calculate_macd(self, indicators, sample_data):
        """Test MACD Berechnung."""
        macd, signal, hist = TechnicalIndicators.calculate_macd(
            sample_data['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        assert len(macd) == len(sample_data['close'])
        assert len(signal) == len(sample_data['close'])
        assert len(hist) == len(sample_data['close'])
        
    def test_calculate_bollinger_bands(self, indicators, sample_data):
        """Test Bollinger Bands Berechnung."""
        ma, upper, lower = TechnicalIndicators.calculate_bollinger_bands(
            sample_data['close'],
            period=20,
            num_std=2.0
        )
        
        assert len(ma) == len(sample_data['close'])
        assert len(upper) == len(sample_data['close'])
        assert len(lower) == len(sample_data['close'])
        assert np.all(upper[~np.isnan(upper)] >= ma[~np.isnan(ma)])
        assert np.all(lower[~np.isnan(lower)] <= ma[~np.isnan(ma)])
        
    def test_calculate_atr(self, indicators, sample_data):
        """Test ATR Berechnung."""
        atr = TechnicalIndicators.calculate_atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            period=14
        )
        
        assert len(atr) == len(sample_data['close'])
        assert np.all(atr[~np.isnan(atr)] >= 0)
        
    def test_detect_candlestick_patterns(self, indicators, sample_data):
        """Test Candlestick Pattern Erkennung."""
        patterns = TechnicalIndicators.detect_candlestick_patterns(
            sample_data['close'],  # Using close as open for simplicity
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )
        
        for pattern in patterns:
            assert isinstance(pattern, CandlestickPattern)
            assert isinstance(pattern.direction, TrendDirection)
            assert 0 <= pattern.strength <= 1
            
    def test_detect_divergences(self, indicators, sample_data):
        """Test Divergenz Erkennung."""
        # Erstelle künstliche Divergenz
        price = np.array([1, 2, 1.5, 2.5, 2, 3])
        indicator = np.array([1, 2, 1.8, 1.9, 1.7, 1.5])
        
        divergences = TechnicalIndicators.detect_divergences(price, indicator)
        
        assert len(divergences) > 0
        for div in divergences:
            assert 'type' in div
            assert 'strength' in div
            assert 'index' in div
            assert 0 <= div['strength'] <= 1
            
    def test_edge_cases(self, indicators):
        """Test Verhalten bei Grenzfällen."""
        # Leere Arrays
        empty = np.array([])
        with pytest.raises(Exception):
            TechnicalIndicators.calculate_ma(empty, period=20)
            
        # Einzelner Wert
        single = np.array([100])
        ma = TechnicalIndicators.calculate_ma(single, period=1)
        assert len(ma) == 1
        assert ma[0] == 100
        
        # NaN Werte
        data_with_nan = np.array([1, np.nan, 3, 4, 5])
        ma = TechnicalIndicators.calculate_ma(data_with_nan, period=2)
        assert np.isnan(ma[1])  # NaN Wert sollte erhalten bleiben
        
    def test_performance(self, indicators, sample_data):
        """Test Performance bei großen Datensätzen."""
        # Generiere großen Datensatz
        large_data = {
            'close': np.random.random(10000) * 100,
            'high': np.random.random(10000) * 110,
            'low': np.random.random(10000) * 90,
            'volume': np.random.random(10000) * 1000
        }
        
        import time
        
        # Zeitmessung für verschiedene Berechnungen
        start_time = time.time()
        TechnicalIndicators.calculate_ma(large_data['close'], period=20)
        ma_time = time.time() - start_time
        
        start_time = time.time()
        TechnicalIndicators.calculate_bollinger_bands(large_data['close'])
        bb_time = time.time() - start_time
        
        # Überprüfe, ob Berechnungen in akzeptabler Zeit durchgeführt wurden
        assert ma_time < 1.0  # Sollte unter 1 Sekunde sein
        assert bb_time < 1.0  # Sollte unter 1 Sekunde sein 