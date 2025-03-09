# Advanced Crypto Trading Bot

Ein fortschrittlicher Kryptowährungs-Trading-Bot mit KI-Integration, technischer Analyse und Risikomanagement.

## Features

### 1. Technische Analyse
- Umfangreiche technische Indikatoren (RSI, MACD, Bollinger Bands, ADX, etc.)
- Candlestick-Muster-Erkennung
- Divergenz-Analyse
- Multi-Timeframe-Analyse
- Dynamische Support/Resistance-Level

### 2. KI-Integration
- GPT-4 und Gemini Pro für Marktanalyse
- Ensemble-Ansatz mit gewichtetem Voting
- Sentiment-Analyse
- Muster-Erkennung in historischen Daten
- Automatische Parameteroptimierung

### 3. Risikomanagement
- Dynamische Position Sizing basierend auf ATR
- Automatische Stop-Loss und Take-Profit-Anpassung
- Trailing Stops mit mehreren Aktivierungsstufen
- Portfolio-Korrelationsanalyse
- Value at Risk (VaR) Berechnung
- Maximum Drawdown Kontrolle

### 4. Backtesting
- Umfassende Backtesting-Engine
- Performance-Metriken (Sharpe Ratio, Win Rate, etc.)
- Transaktionskosten und Slippage-Simulation
- Monte Carlo Simulation
- Walk-Forward-Optimierung

### 5. Monitoring & Benachrichtigungen
- Telegram-Integration für Benachrichtigungen
- Detaillierte Logging
- Performance-Tracking
- Fehlerbehandlung und Wiederherstellung

## Installation

1. Repository klonen:
```bash
git clone https://github.com/SnaQOG/Cursor.git
cd trading-bot
```

2. Python-Umgebung erstellen und aktivieren:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oder
.venv\Scripts\activate  # Windows
```

3. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

4. Umgebungsvariablen konfigurieren:
```bash
cp .env.example .env
# Bearbeite .env und füge deine API-Keys hinzu
```

## Konfiguration

### Umgebungsvariablen

Wichtige Umgebungsvariablen in der `.env`-Datei:

```env
# API Keys
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Konfiguration
TRADING_PAIRS=["BTCUSDT_UMCBL","ETHUSDT_UMCBL"]
TRADING_TIMEFRAMES=["1h","30m"]
TRADING_UPDATE_INTERVAL=60

# Risikomanagement
TRADING_MAX_RISK_PER_TRADE=0.02
TRADING_MAX_TOTAL_RISK=0.06
TRADING_MIN_RISK_REWARD=2.0

# AI Konfiguration
AI_ANALYSIS_WEIGHT=0.3
AI_MIN_CONFIDENCE=0.7
```

### Trading-Konfiguration

Die Hauptkonfiguration erfolgt in `src/config/config.py`. Hier können Sie verschiedene Parameter anpassen:

- Technische Indikatoren
- Risikomanagement-Parameter
- Backtesting-Einstellungen
- API-Konfiguration
- KI-Parameter

## Verwendung

### Trading Bot starten

```bash
python Trading.py
```

### Backtesting durchführen

```bash
python backtest.py --strategy strategy_name --start 2023-01-01 --end 2024-03-01
```

### Telegram-Bot starten

```bash
python telegram_menu.py
```

## Projektstruktur

```
trading-bot/
├── src/
│   ├── api/              # API-Kommunikation
│   ├── analysis/         # Technische & KI-Analyse
│   ├── backtesting/      # Backtesting-Engine
│   ├── config/           # Konfigurationsdateien
│   ├── data/            # Historische Daten
│   ├── models/          # Datenmodelle
│   ├── strategies/      # Trading-Strategien
│   ├── utils/           # Hilfsfunktionen
│   └── tests/           # Unit Tests
├── logs/                # Log-Dateien
├── notebooks/           # Jupyter Notebooks
├── .env                 # Umgebungsvariablen
├── requirements.txt     # Python-Abhängigkeiten
└── README.md           # Dokumentation
```

## Trading-Strategien

Der Bot unterstützt verschiedene Trading-Strategien:

1. **Multi-Timeframe-Momentum**
   - Kombiniert Signale aus verschiedenen Zeitrahmen
   - Verwendet RSI, MACD und Bollinger Bands
   - Bestätigung durch Volumen-Profile

2. **AI-Enhanced Pattern Recognition**
   - Erkennt Candlestick-Muster
   - KI-basierte Mustervalidierung
   - Sentiment-Analyse-Integration

3. **Trend-Following mit dynamischen Stops**
   - ADX-basierte Trend-Erkennung
   - Dynamische Stop-Loss-Anpassung
   - Trailing-Stop-Optimierung

## Risikomanagement

Der Bot implementiert mehrere Risikomanagement-Strategien:

1. **Position Sizing**
   - ATR-basierte Größenberechnung
   - Account-Risiko-Limitierung
   - Korrelations-Checks

2. **Stop-Loss-Strategien**
   - Technische Stops basierend auf ATR
   - Time-based Stops
   - Trailing Stops mit mehreren Aktivierungsstufen

3. **Portfolio-Management**
   - Maximales Gesamtrisiko
   - Korrelationsbasierte Positionslimits
   - VaR-Berechnung

## Performance-Monitoring

Der Bot bietet verschiedene Monitoring-Funktionen:

1. **Telegram-Benachrichtigungen**
   - Trade-Ausführungen
   - Stop-Loss/Take-Profit-Hits
   - Fehler und Warnungen
   - Performance-Updates

2. **Logging**
   - Detaillierte Trade-Logs
   - Fehler-Tracking
   - Performance-Metriken

3. **Performance-Metriken**
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Profit Factor

## Fehlerbehandlung

Der Bot implementiert robuste Fehlerbehandlung:

1. **API-Fehler**
   - Automatische Wiederholungsversuche
   - Exponentielles Backoff
   - Failover-Mechanismen

2. **Daten-Validierung**
   - Konsistenzprüfungen
   - Ausreißer-Erkennung
   - Daten-Qualitätschecks

3. **Wiederherstellung**
   - Automatischer Neustart
   - Zustandswiederherstellung
   - Fehlerbenachrichtigungen

## Beitragen

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit deine Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne einen Pull Request

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die [LICENSE](LICENSE) Datei für Details.

## Kontakt

Dominik Diener - [@SnaQOG](https://twitter.com/SnaQOG)

Projekt Link: [https://github.com/SnaQOG/Cursor](https://github.com/SnaQOG/Cursor)
