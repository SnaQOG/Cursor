import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class TrendDirection(Enum):
    """Enum for trend directions."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

@dataclass
class CandlestickPattern:
    """Data class for candlestick pattern detection results."""
    name: str
    strength: float  # 0-1 scale
    direction: TrendDirection
    description: str

class TechnicalIndicators:
    """Enhanced technical analysis indicators."""
    
    @staticmethod
    def calculate_ma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Moving Average."""
        return pd.Series(data).rolling(window=period).mean().values
        
    @staticmethod
    def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
        
    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = np.diff(data)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate(([np.nan], rsi))
        
    @staticmethod
    def calculate_macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = pd.Series(data).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(data).ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, histogram.values
        
    @staticmethod
    def calculate_bollinger_bands(data: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        ma = pd.Series(data).rolling(window=period).mean()
        std = pd.Series(data).rolling(window=period).std()
        
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        return ma.values, upper_band.values, lower_band.values
        
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        
        ranges = np.vstack([high_low, high_close, low_close])
        true_range = np.max(ranges, axis=0)
        
        return pd.Series(true_range).rolling(window=period).mean().values
        
    @staticmethod
    def calculate_stochastic_rsi(data: np.ndarray, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic RSI."""
        rsi = TechnicalIndicators.calculate_rsi(data, period)
        
        stoch_rsi = np.zeros_like(rsi)
        for i in range(period, len(rsi)):
            window = rsi[i-period+1:i+1]
            if not np.isnan(window).any():
                min_val = np.min(window)
                max_val = np.max(window)
                if max_val - min_val != 0:
                    stoch_rsi[i] = (rsi[i] - min_val) / (max_val - min_val) * 100
                    
        k = pd.Series(stoch_rsi).rolling(window=smooth_k).mean().values
        d = pd.Series(k).rolling(window=smooth_d).mean().values
        
        return k, d
        
    @staticmethod
    def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume."""
        price_change = np.diff(close)
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if price_change[i-1] > 0:
                obv[i] = obv[i-1] + volume[i]
            elif price_change[i-1] < 0:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        return obv
        
    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
        return vwap
        
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Average Directional Index."""
        # Calculate True Range
        tr = TechnicalIndicators.calculate_atr(high, low, close, 1)
        
        # Calculate +DM and -DM
        high_diff = np.diff(high)
        low_diff = np.diff(low)
        
        pos_dm = np.where((high_diff > 0) & (high_diff > -low_diff), high_diff, 0)
        neg_dm = np.where((low_diff < 0) & (-low_diff > high_diff), -low_diff, 0)
        
        # Calculate smoothed values
        tr_smooth = pd.Series(tr).rolling(window=period).mean()
        pos_dm_smooth = pd.Series(pos_dm).rolling(window=period).mean()
        neg_dm_smooth = pd.Series(neg_dm).rolling(window=period).mean()
        
        # Calculate +DI and -DI
        pos_di = 100 * pos_dm_smooth / tr_smooth
        neg_di = 100 * neg_dm_smooth / tr_smooth
        
        # Calculate ADX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = pd.Series(dx).rolling(window=period).mean()
        
        return adx.values, pos_di.values, neg_di.values
        
    @staticmethod
    def detect_candlestick_patterns(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[CandlestickPattern]:
        """Detect candlestick patterns in the data."""
        patterns = []
        
        # Calculate basic measures
        body = close - open_
        body_pct = body / open_ * 100
        upper_shadow = high - np.maximum(open_, close)
        lower_shadow = np.minimum(open_, close) - low
        
        # Doji pattern
        doji_idx = np.where(np.abs(body_pct) < 0.1)[0]
        if len(doji_idx) > 0:
            patterns.append(CandlestickPattern(
                name="Doji",
                strength=0.7,
                direction=TrendDirection.SIDEWAYS,
                description="Indecision in the market"
            ))
            
        # Hammer pattern
        hammer_idx = np.where(
            (body_pct > 0) &  # Bullish
            (lower_shadow > 2 * np.abs(body)) &  # Long lower shadow
            (upper_shadow < 0.1 * np.abs(body))  # Short upper shadow
        )[0]
        if len(hammer_idx) > 0:
            patterns.append(CandlestickPattern(
                name="Hammer",
                strength=0.8,
                direction=TrendDirection.UP,
                description="Potential trend reversal from downtrend"
            ))
            
        # Shooting Star pattern
        shooting_star_idx = np.where(
            (body_pct < 0) &  # Bearish
            (upper_shadow > 2 * np.abs(body)) &  # Long upper shadow
            (lower_shadow < 0.1 * np.abs(body))  # Short lower shadow
        )[0]
        if len(shooting_star_idx) > 0:
            patterns.append(CandlestickPattern(
                name="Shooting Star",
                strength=0.8,
                direction=TrendDirection.DOWN,
                description="Potential trend reversal from uptrend"
            ))
            
        # Engulfing patterns
        for i in range(1, len(close)):
            # Bullish engulfing
            if (body[i] > 0 and  # Current candle is bullish
                body[i-1] < 0 and  # Previous candle is bearish
                close[i] > open_[i-1] and  # Current close is higher than previous open
                open_[i] < close[i-1]):  # Current open is lower than previous close
                patterns.append(CandlestickPattern(
                    name="Bullish Engulfing",
                    strength=0.9,
                    direction=TrendDirection.UP,
                    description="Strong reversal signal from downtrend"
                ))
                
            # Bearish engulfing
            elif (body[i] < 0 and  # Current candle is bearish
                  body[i-1] > 0 and  # Previous candle is bullish
                  close[i] < open_[i-1] and  # Current close is lower than previous open
                  open_[i] > close[i-1]):  # Current open is higher than previous close
                patterns.append(CandlestickPattern(
                    name="Bearish Engulfing",
                    strength=0.9,
                    direction=TrendDirection.DOWN,
                    description="Strong reversal signal from uptrend"
                ))
                
        return patterns
        
    @staticmethod
    def detect_divergences(price: np.ndarray, indicator: np.ndarray, lookback: int = 10) -> List[Dict]:
        """Detect regular and hidden divergences between price and indicator."""
        divergences = []
        
        for i in range(lookback, len(price)):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            # Find local extrema
            price_highs = np.where((price_window[1:-1] > price_window[:-2]) & 
                                 (price_window[1:-1] > price_window[2:]))[0] + 1
            price_lows = np.where((price_window[1:-1] < price_window[:-2]) & 
                                (price_window[1:-1] < price_window[2:]))[0] + 1
            
            ind_highs = np.where((indicator_window[1:-1] > indicator_window[:-2]) & 
                               (indicator_window[1:-1] > indicator_window[2:]))[0] + 1
            ind_lows = np.where((indicator_window[1:-1] < indicator_window[:-2]) & 
                              (indicator_window[1:-1] < indicator_window[2:]))[0] + 1
            
            # Regular bullish divergence
            if (len(price_lows) >= 2 and len(ind_lows) >= 2 and
                price_window[price_lows[-1]] < price_window[price_lows[-2]] and
                indicator_window[ind_lows[-1]] > indicator_window[ind_lows[-2]]):
                divergences.append({
                    'type': 'regular_bullish',
                    'strength': 0.8,
                    'index': i
                })
                
            # Regular bearish divergence
            if (len(price_highs) >= 2 and len(ind_highs) >= 2 and
                price_window[price_highs[-1]] > price_window[price_highs[-2]] and
                indicator_window[ind_highs[-1]] < indicator_window[ind_highs[-2]]):
                divergences.append({
                    'type': 'regular_bearish',
                    'strength': 0.8,
                    'index': i
                })
                
            # Hidden bullish divergence
            if (len(price_lows) >= 2 and len(ind_lows) >= 2 and
                price_window[price_lows[-1]] > price_window[price_lows[-2]] and
                indicator_window[ind_lows[-1]] < indicator_window[ind_lows[-2]]):
                divergences.append({
                    'type': 'hidden_bullish',
                    'strength': 0.7,
                    'index': i
                })
                
            # Hidden bearish divergence
            if (len(price_highs) >= 2 and len(ind_highs) >= 2 and
                price_window[price_highs[-1]] < price_window[price_highs[-2]] and
                indicator_window[ind_highs[-1]] > indicator_window[ind_highs[-2]]):
                divergences.append({
                    'type': 'hidden_bearish',
                    'strength': 0.7,
                    'index': i
                })
                
        return divergences 