from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import talib
from src.strategies.base_strategy import BaseStrategy
from src.utils.logger import TradingLogger
from src.utils.config_manager import ConfigManager

class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-Timeframe Trading Strategie.
    Kombiniert Signale aus verschiedenen Zeitrahmen mit technischen Indikatoren.
    """
    
    def __init__(self, config: ConfigManager, logger: TradingLogger):
        super().__init__(config, logger)
        self.ta_config = config.get_section('TA')
        self.signal_history = {}
        self.market_regime = {}
        
    def _calculate_indicators(self) -> None:
        """Berechnet technische Indikatoren für alle Timeframes"""
        for timeframe, df in self.data.items():
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            opens = df['open'].values
            
            # Initialisiere Dictionary für diesen Timeframe
            self.indicators[timeframe] = {}
            
            # RSI mit dynamischen Grenzen basierend auf Volatilität
            rsi = talib.RSI(closes, timeperiod=self.ta_config['RSI']['PERIOD'])
            atr = talib.ATR(highs, lows, closes, timeperiod=self.ta_config['ATR']['PERIOD'])
            volatility_factor = atr[-1] / closes[-1]  # Normalisierte Volatilität
            
            # Dynamische RSI-Grenzen
            dynamic_oversold = max(20, 30 - (volatility_factor * 100))
            dynamic_overbought = min(80, 70 + (volatility_factor * 100))
            
            self.indicators[timeframe].update({
                'rsi': rsi,
                'rsi_bounds': {'oversold': dynamic_oversold, 'overbought': dynamic_overbought}
            })
            
            # MACD
            macd, signal, hist = talib.MACD(
                closes,
                fastperiod=self.ta_config['MACD']['FAST_PERIOD'],
                slowperiod=self.ta_config['MACD']['SLOW_PERIOD'],
                signalperiod=self.ta_config['MACD']['SIGNAL_PERIOD']
            )
            self.indicators[timeframe].update({
                'macd': macd,
                'macd_signal': signal,
                'macd_hist': hist
            })
            
            # Bollinger Bands mit dynamischer Standardabweichung
            std_dev = self.ta_config['BOLLINGER_BANDS']['STD_DEV'] * (1 + volatility_factor)
            upper, middle, lower = talib.BBANDS(
                closes,
                timeperiod=self.ta_config['BOLLINGER_BANDS']['PERIOD'],
                nbdevup=std_dev,
                nbdevdn=std_dev
            )
            self.indicators[timeframe].update({
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower
            })
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                highs,
                lows,
                closes,
                fastk_period=self.ta_config['STOCHASTIC']['K_PERIOD'],
                slowk_period=self.ta_config['STOCHASTIC']['D_PERIOD'],
                slowd_period=self.ta_config['STOCHASTIC']['SLOW_PERIOD']
            )
            self.indicators[timeframe].update({
                'stoch_k': slowk,
                'stoch_d': slowd
            })
            
            # ATR
            self.indicators[timeframe]['atr'] = atr
            
            # ADX
            self.indicators[timeframe]['adx'] = talib.ADX(
                highs,
                lows,
                closes,
                timeperiod=self.ta_config['ADX']['PERIOD']
            )
            
            # CCI
            self.indicators[timeframe]['cci'] = talib.CCI(
                highs,
                lows,
                closes,
                timeperiod=self.ta_config['CCI']['PERIOD']
            )
            
            # OBV mit EMA für Trendbestätigung
            obv = talib.OBV(closes, volumes)
            obv_ema = talib.EMA(obv, timeperiod=20)
            self.indicators[timeframe].update({
                'obv': obv,
                'obv_ema': obv_ema
            })
            
            # VWAP (Volume Weighted Average Price)
            typical_price = (highs + lows + closes) / 3
            vwap = np.cumsum(typical_price * volumes) / np.cumsum(volumes)
            self.indicators[timeframe]['vwap'] = vwap
            
            # Candlestick Patterns
            patterns = {
                'hammer': talib.HAMMER(opens, highs, lows, closes),
                'engulfing': talib.CDLENGULFING(opens, highs, lows, closes),
                'morning_star': talib.CDLMORNINGSTAR(opens, highs, lows, closes),
                'evening_star': talib.CDLEVENINGSTAR(opens, highs, lows, closes),
                'doji': talib.CDLDOJI(opens, highs, lows, closes),
                'three_white_soldiers': talib.CDL3WHITESOLDIERS(opens, highs, lows, closes),
                'three_black_crows': talib.CDL3BLACKCROWS(opens, highs, lows, closes)
            }
            self.indicators[timeframe]['candlestick_patterns'] = patterns
            
            # Moving Averages
            for period in self.ta_config['MOVING_AVERAGES']['EMA_PERIODS']:
                self.indicators[timeframe][f'ema_{period}'] = talib.EMA(closes, timeperiod=period)
            for period in self.ta_config['MOVING_AVERAGES']['SMA_PERIODS']:
                self.indicators[timeframe][f'sma_{period}'] = talib.SMA(closes, timeperiod=period)
                
            # Divergence Detection
            self._calculate_divergences(timeframe, closes, rsi, macd)
            
            # Market Regime Detection
            self.market_regime[timeframe] = self._detect_market_regime(timeframe)
            
    def _calculate_divergences(self, timeframe: str, closes: np.ndarray, rsi: np.ndarray, macd: np.ndarray) -> None:
        """Berechnet Divergenzen zwischen Preis und Indikatoren"""
        # Finde lokale Extrema
        price_highs = self._find_local_extrema(closes, is_max=True)
        price_lows = self._find_local_extrema(closes, is_max=False)
        rsi_highs = self._find_local_extrema(rsi, is_max=True)
        rsi_lows = self._find_local_extrema(rsi, is_max=False)
        macd_highs = self._find_local_extrema(macd, is_max=True)
        macd_lows = self._find_local_extrema(macd, is_max=False)
        
        # Speichere Divergenzen
        self.indicators[timeframe]['divergences'] = {
            'rsi_bearish': self._check_bearish_divergence(closes, rsi, price_highs, rsi_highs),
            'rsi_bullish': self._check_bullish_divergence(closes, rsi, price_lows, rsi_lows),
            'macd_bearish': self._check_bearish_divergence(closes, macd, price_highs, macd_highs),
            'macd_bullish': self._check_bullish_divergence(closes, macd, price_lows, macd_lows)
        }
        
    def _find_local_extrema(self, data: np.ndarray, is_max: bool, window: int = 5) -> List[int]:
        """Findet lokale Maxima oder Minima in einer Datenreihe"""
        indices = []
        for i in range(window, len(data) - window):
            if is_max:
                if data[i] == max(data[i-window:i+window+1]):
                    indices.append(i)
            else:
                if data[i] == min(data[i-window:i+window+1]):
                    indices.append(i)
        return indices
        
    def _check_bearish_divergence(self, prices: np.ndarray, indicator: np.ndarray, 
                                price_highs: List[int], ind_highs: List[int]) -> bool:
        """Überprüft auf bearische Divergenz"""
        if len(price_highs) < 2 or len(ind_highs) < 2:
            return False
            
        # Prüfe die letzten beiden Hochs
        last_two_price_highs = sorted(price_highs[-2:])
        last_two_ind_highs = sorted(ind_highs[-2:])
        
        if prices[last_two_price_highs[1]] > prices[last_two_price_highs[0]] and \
           indicator[last_two_ind_highs[1]] < indicator[last_two_ind_highs[0]]:
            return True
        return False
        
    def _check_bullish_divergence(self, prices: np.ndarray, indicator: np.ndarray,
                                price_lows: List[int], ind_lows: List[int]) -> bool:
        """Überprüft auf bullische Divergenz"""
        if len(price_lows) < 2 or len(ind_lows) < 2:
            return False
            
        # Prüfe die letzten beiden Tiefs
        last_two_price_lows = sorted(price_lows[-2:])
        last_two_ind_lows = sorted(ind_lows[-2:])
        
        if prices[last_two_price_lows[1]] < prices[last_two_price_lows[0]] and \
           indicator[last_two_ind_lows[1]] > indicator[last_two_ind_lows[0]]:
            return True
        return False
        
    def _detect_market_regime(self, timeframe: str) -> str:
        """Erkennt das aktuelle Marktregime (Trend/Range)"""
        ind = self.indicators[timeframe]
        
        # ADX für Trendstärke
        adx = ind['adx'][-1]
        
        # Bollinger Band Breite für Volatilität
        bb_width = (ind['bb_upper'][-1] - ind['bb_lower'][-1]) / ind['bb_middle'][-1]
        
        # Prüfe auf Trendmarkt
        if adx > 25:
            if ind['ema_9'][-1] > ind['sma_50'][-1]:
                return 'uptrend'
            else:
                return 'downtrend'
        
        # Prüfe auf Range-Markt
        elif bb_width < 0.05:  # Enge Bollinger Bänder
            return 'range'
            
        return 'undefined'
        
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Generiert Trading-Signale basierend auf Multi-Timeframe-Analyse
        
        Args:
            data: Dictionary mit aktuellen Marktdaten
            
        Returns:
            Optional[Dict]: Trading-Signal oder None
        """
        if not self._is_market_condition_suitable():
            return None
            
        # Analysiere jeden Timeframe
        signals = {}
        weights = {
            '1d': 0.4,  # Langfristiger Trend
            '4h': 0.3,  # Mittelfristiger Trend
            '1h': 0.2,  # Kurzfristiger Trend
            '15m': 0.1  # Einstiegszeitpunkt
        }
        
        for timeframe in self.data.keys():
            if timeframe not in weights:
                continue
                
            signal = self._analyze_timeframe(timeframe)
            if signal:
                signals[timeframe] = signal
                
        if not signals:
            return None
            
        # Kombiniere Signale
        combined_signal = self._combine_timeframe_signals(signals, weights)
        if not combined_signal:
            return None
            
        # Berechne Entry, Stop Loss und Take Profit
        entry_price = self.data['15m']['close'].iloc[-1]
        atr = self.indicators['1h']['atr'][-1]
        
        stop_loss = self._calculate_stop_loss(
            entry_price,
            combined_signal['type'],
            atr
        )
        
        take_profit = self._calculate_take_profit(
            entry_price,
            combined_signal['type'],
            stop_loss
        )
        
        return {
            'type': combined_signal['type'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': combined_signal['confidence'],
            'timeframes': signals,
            'reason': combined_signal['reason']
        }
        
    def _analyze_timeframe(self, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Analysiert einen einzelnen Timeframe
        
        Args:
            timeframe: Zu analysierender Timeframe
            
        Returns:
            Optional[Dict]: Signal-Dictionary oder None
        """
        ind = self.indicators[timeframe]
        df = self.data[timeframe]
        
        # Bestimme Marktregime
        regime = self.market_regime[timeframe]
        
        # Sammle Signale basierend auf Marktregime
        bull_signals = []
        bear_signals = []
        
        # === Trend-basierte Signale ===
        if regime in ['uptrend', 'downtrend']:
            # ADX - Trendstärke
            adx_strength = ind['adx'][-1]
            if adx_strength > self.ta_config['ADX']['STRONG_TREND_THRESHOLD']:
                if regime == 'uptrend':
                    bull_signals.append(('ADX', 1.2))  # Höhere Gewichtung in Trendphasen
                else:
                    bear_signals.append(('ADX', 1.2))
            
            # VWAP als Unterstützung/Widerstand
            if regime == 'uptrend' and df['close'].iloc[-1] > ind['vwap'][-1]:
                bull_signals.append(('VWAP', 1.0))
            elif regime == 'downtrend' and df['close'].iloc[-1] < ind['vwap'][-1]:
                bear_signals.append(('VWAP', 1.0))
                
            # OBV-Trendbestätigung
            if ind['obv'][-1] > ind['obv_ema'][-1]:
                bull_signals.append(('OBV', 0.8))
            else:
                bear_signals.append(('OBV', 0.8))
                
        # === Range-basierte Signale ===
        elif regime == 'range':
            # RSI mit dynamischen Grenzen
            if ind['rsi'][-1] < ind['rsi_bounds']['oversold']:
                bull_signals.append(('RSI', 1.2))  # Höhere Gewichtung in Range-Phasen
            elif ind['rsi'][-1] > ind['rsi_bounds']['overbought']:
                bear_signals.append(('RSI', 1.2))
            
            # Bollinger Bands
            if df['close'].iloc[-1] < ind['bb_lower'][-1]:
                bull_signals.append(('BB', 1.2))
            elif df['close'].iloc[-1] > ind['bb_upper'][-1]:
                bear_signals.append(('BB', 1.2))
                
            # Stochastic
            if ind['stoch_k'][-1] < 20 and ind['stoch_d'][-1] < 20:
                bull_signals.append(('Stochastic', 1.0))
            elif ind['stoch_k'][-1] > 80 and ind['stoch_d'][-1] > 80:
                bear_signals.append(('Stochastic', 1.0))
        
        # === Gemeinsame Signale für alle Regime ===
        # MACD
        if ind['macd'][-1] > ind['macd_signal'][-1] and ind['macd_hist'][-1] > 0:
            bull_signals.append(('MACD', 1.0))
        elif ind['macd'][-1] < ind['macd_signal'][-1] and ind['macd_hist'][-1] < 0:
            bear_signals.append(('MACD', 1.0))
        
        # Divergenzen
        if ind['divergences']['rsi_bullish'] or ind['divergences']['macd_bullish']:
            bull_signals.append(('Divergence', 1.5))  # Hohe Gewichtung für Divergenzen
        if ind['divergences']['rsi_bearish'] or ind['divergences']['macd_bearish']:
            bear_signals.append(('Divergence', 1.5))
            
        # Candlestick Patterns
        patterns = ind['candlestick_patterns']
        bullish_patterns = ['hammer', 'morning_star', 'three_white_soldiers']
        bearish_patterns = ['evening_star', 'three_black_crows']
        
        for pattern in bullish_patterns:
            if patterns[pattern][-1] > 0:
                bull_signals.append((f'Candle_{pattern}', 0.8))
                
        for pattern in bearish_patterns:
            if patterns[pattern][-1] > 0:
                bear_signals.append((f'Candle_{pattern}', 0.8))
                
        # Doji in Trendumkehrzonen
        if patterns['doji'][-1] > 0:
            if df['close'].iloc[-1] > ind['bb_upper'][-1]:
                bear_signals.append(('Doji_Resistance', 0.5))
            elif df['close'].iloc[-1] < ind['bb_lower'][-1]:
                bull_signals.append(('Doji_Support', 0.5))
                
        # Engulfing Patterns mit Volumenbestätigung
        if patterns['engulfing'][-1] > 0 and self._check_volume_confirmation(df):
            if df['close'].iloc[-1] > df['open'].iloc[-1]:  # Bullish Engulfing
                bull_signals.append(('Engulfing_Volume', 1.0))
            else:  # Bearish Engulfing
                bear_signals.append(('Engulfing_Volume', 1.0))
        
        # Signalpersistenz prüfen
        if timeframe not in self.signal_history:
            self.signal_history[timeframe] = []
            
        # Berechne Gesamtscores
        bull_score = sum(weight for _, weight in bull_signals)
        bear_score = sum(weight for _, weight in bear_signals)
        
        # Speichere aktuelles Signal
        current_signal = None
        if bull_score > bear_score:
            current_signal = 'bullish'
        elif bear_score > bull_score:
            current_signal = 'bearish'
            
        self.signal_history[timeframe].append(current_signal)
        if len(self.signal_history[timeframe]) > 3:  # Behalte nur die letzten 3 Signale
            self.signal_history[timeframe].pop(0)
            
        # Prüfe Signalkonsistenz
        if len(self.signal_history[timeframe]) == 3:
            if not all(sig == current_signal for sig in self.signal_history[timeframe]):
                return None  # Signal nicht konsistent genug
        
        # Generiere Signal basierend auf der Mehrheit der Indikatoren
        min_signals = self.config.get('TRADING', 'MIN_INDICATORS_AGREEMENT', 3)
        if len(bull_signals) >= min_signals and bull_score > bear_score:
            return {
                'type': 'long',
                'confidence': bull_score / (bull_score + bear_score),
                'indicators': [sig[0] for sig in bull_signals],
                'regime': regime,
                'reason': f"Bullish signals in {regime} regime: {', '.join(sig[0] for sig in bull_signals)}"
            }
        elif len(bear_signals) >= min_signals and bear_score > bull_score:
            return {
                'type': 'short',
                'confidence': bear_score / (bull_score + bear_score),
                'indicators': [sig[0] for sig in bear_signals],
                'regime': regime,
                'reason': f"Bearish signals in {regime} regime: {', '.join(sig[0] for sig in bear_signals)}"
            }
            
        return None
        
    def _combine_timeframe_signals(self,
                                 signals: Dict[str, Dict[str, Any]],
                                 weights: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Kombiniert Signale aus verschiedenen Timeframes
        
        Args:
            signals: Dictionary mit Signalen pro Timeframe
            weights: Gewichtung der Timeframes
            
        Returns:
            Optional[Dict]: Kombiniertes Signal oder None
        """
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        reasons = []
        regime_consistency = True
        
        # Prüfe Regime-Konsistenz über Timeframes
        regimes = [signal['regime'] for signal in signals.values()]
        main_regime = max(set(regimes), key=regimes.count)  # Häufigstes Regime
        regime_agreement = sum(1 for r in regimes if r == main_regime) / len(regimes)
        
        # Mindestens 60% der Timeframes sollten im gleichen Regime sein
        if regime_agreement < 0.6:
            regime_consistency = False
        
        for timeframe, signal in signals.items():
            # Basis-Gewichtung für den Timeframe
            base_weight = weights.get(timeframe, 0.0)
            
            # Regime-basierte Gewichtungsanpassung
            regime_weight = 1.0
            if signal['regime'] == main_regime:
                regime_weight = 1.2  # Erhöhe Gewicht wenn Regime übereinstimmt
            elif signal['regime'] == 'range' and main_regime != 'range':
                regime_weight = 0.8  # Reduziere Gewicht bei Regime-Konflikt
                
            # Volatilitäts-basierte Gewichtungsanpassung
            volatility = self._calculate_volatility(self.data[timeframe])
            vol_weight = 1.0
            if volatility > 0.02:  # Hohe Volatilität
                if timeframe in ['15m', '30m']:
                    vol_weight = 0.8  # Reduziere Gewicht kürzerer Timeframes
                else:
                    vol_weight = 1.2  # Erhöhe Gewicht längerer Timeframes
                    
            # Indikator-Konsistenz-Gewichtung
            indicator_weight = 1.0
            if len(signal['indicators']) >= 4:  # Mehr übereinstimmende Indikatoren
                indicator_weight = 1.2
                
            # Kombiniere alle Gewichtungsfaktoren
            final_weight = base_weight * regime_weight * vol_weight * indicator_weight
            
            if signal['type'] == 'long':
                long_score += final_weight * signal['confidence']
            else:
                short_score += final_weight * signal['confidence']
                
            total_weight += final_weight
            reasons.append(f"{timeframe}: {signal['reason']}")
            
        if total_weight == 0:
            return None
            
        # Normalisiere Scores
        long_score /= total_weight
        short_score /= total_weight
        
        # Mindest-Konfidenz basierend auf Marktregime
        base_confidence = self.config.get('TRADING', 'MIN_CONFIDENCE_SCORE', 0.6)
        if main_regime == 'range':
            min_confidence = base_confidence * 1.2  # Höhere Anforderung in Seitwärtsmärkten
        else:
            min_confidence = base_confidence
            
        # Prüfe Signalkonsistenz über Timeframes
        if not regime_consistency:
            min_confidence *= 1.2  # Erhöhe Anforderungen bei inkonsistenten Regimes
            
        if long_score > short_score and long_score >= min_confidence:
            return {
                'type': 'long',
                'confidence': long_score,
                'regime': main_regime,
                'regime_agreement': regime_agreement,
                'reason': ' | '.join(reasons)
            }
        elif short_score > long_score and short_score >= min_confidence:
            return {
                'type': 'short',
                'confidence': short_score,
                'regime': main_regime,
                'regime_agreement': regime_agreement,
                'reason': ' | '.join(reasons)
            }
            
        return None
        
    def should_exit(self, position: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prüft ob eine Position geschlossen werden sollte
        
        Args:
            position: Aktuelle Position
            data: Aktuelle Marktdaten
            
        Returns:
            bool: True wenn Position geschlossen werden soll
        """
        timeframe = '1h'  # Nutze 1h für Exit-Entscheidungen
        ind = self.indicators[timeframe]
        regime = self.market_regime[timeframe]
        
        # Hole aktuelle Preisdaten
        current_price = data[timeframe]['close'].iloc[-1]
        entry_price = position['entry_price']
        
        # === Regime-basierte Exit-Bedingungen ===
        if regime == 'range':
            # In Range-Märkten schneller aussteigen
            if position['type'] == 'long':
                if current_price > ind['bb_upper'][-1] or ind['rsi'][-1] > ind['rsi_bounds']['overbought']:
                    return True
            else:
                if current_price < ind['bb_lower'][-1] or ind['rsi'][-1] < ind['rsi_bounds']['oversold']:
                    return True
                    
        elif regime in ['uptrend', 'downtrend']:
            # In Trendmärkten auf Trendumkehr achten
            if position['type'] == 'long' and regime == 'downtrend':
                # Zusätzliche Bestätigung durch Volumen und VWAP
                if (current_price < ind['vwap'][-1] and 
                    ind['obv'][-1] < ind['obv_ema'][-1] and
                    ind['adx'][-1] > self.ta_config['ADX']['STRONG_TREND_THRESHOLD']):
                    return True
            elif position['type'] == 'short' and regime == 'uptrend':
                if (current_price > ind['vwap'][-1] and 
                    ind['obv'][-1] > ind['obv_ema'][-1] and
                    ind['adx'][-1] > self.ta_config['ADX']['STRONG_TREND_THRESHOLD']):
                    return True
        
        # === Divergenz-basierte Exits ===
        if position['type'] == 'long' and ind['divergences']['rsi_bearish']:
            return True
        elif position['type'] == 'short' and ind['divergences']['rsi_bullish']:
            return True
            
        # === Candlestick-Pattern Exits ===
        patterns = ind['candlestick_patterns']
        if position['type'] == 'long':
            bearish_reversal = any(patterns[p][-1] > 0 for p in ['evening_star', 'three_black_crows'])
            if bearish_reversal and current_price > entry_price:  # Nur wenn im Profit
                return True
        else:
            bullish_reversal = any(patterns[p][-1] > 0 for p in ['morning_star', 'three_white_soldiers'])
            if bullish_reversal and current_price < entry_price:  # Nur wenn im Profit
                return True
                
        # === Momentum-basierte Exits ===
        if position['type'] == 'long':
            # Bearische Signale
            if (ind['macd'][-1] < ind['macd_signal'][-1] and  # MACD Crossover
                ind['stoch_k'][-1] > 80 and ind['stoch_d'][-1] > 80 and  # Überkauft
                current_price > entry_price):  # Nur wenn im Profit
                return True
        else:
            # Bullische Signale
            if (ind['macd'][-1] > ind['macd_signal'][-1] and  # MACD Crossover
                ind['stoch_k'][-1] < 20 and ind['stoch_d'][-1] < 20 and  # Überverkauft
                current_price < entry_price):  # Nur wenn im Profit
                return True
                
        # === Volumen-Profile Exits ===
        if self._check_volume_confirmation(data[timeframe]):
            if position['type'] == 'long' and current_price < ind['vwap'][-1]:
                return True
            elif position['type'] == 'short' and current_price > ind['vwap'][-1]:
                return True
                
        return False 