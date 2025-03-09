import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Trading configuration parameters."""
    
    # General settings
    DEVELOPMENT_MODE: bool = False
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "data"
    
    # Trading parameters
    TRADING_PAIRS: list = None  # Set from environment
    TIMEFRAMES: list = ["1h", "30m"]
    UPDATE_INTERVAL: int = 60  # seconds
    
    # Account settings
    INITIAL_BALANCE: float = 10000.0
    COMMISSION_RATE: float = 0.001  # 0.1%
    SLIPPAGE: float = 0.0005  # 0.05%
    
    # Risk management
    MAX_RISK_PER_TRADE: float = 0.02  # 2% per trade
    MAX_TOTAL_RISK: float = 0.06  # 6% total
    MIN_RISK_REWARD: float = 2.0  # Minimum 2:1 ratio
    POSITION_SIZING_ATR: float = 1.5  # ATR multiplier
    MIN_POSITION_SIZE: float = 0.001
    MAX_POSITION_SIZE: float = 1.0
    CHECK_CORRELATION: bool = True
    
    # Stop loss and take profit
    ATR_MULTIPLIER: float = 2.0
    TRAILING_STOP_ACTIVATION: float = 0.02  # 2% profit
    TRAILING_STOP_DISTANCE: float = 0.01  # 1% from price
    
    # Technical indicators
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30
    RSI_OVERBOUGHT: float = 70
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: float = 25
    
    STOCH_RSI_PERIOD: int = 14
    STOCH_RSI_K: int = 3
    STOCH_RSI_D: int = 3
    STOCH_RSI_OVERSOLD: float = 20
    STOCH_RSI_OVERBOUGHT: float = 80
    
    # AI Analysis
    AI_ANALYSIS_WEIGHT: float = 0.3  # 30% weight for AI signals
    AI_MIN_CONFIDENCE: float = 0.7  # Minimum confidence for AI signals
    
    # Backtesting
    BACKTEST_START_DATE: str = "2023-01-01"
    BACKTEST_END_DATE: str = "2024-03-01"
    
    # API settings
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_DELAY: int = 5  # seconds
    API_TIMEOUT: int = 30  # seconds
    
    def __post_init__(self):
        """Load values from environment variables."""
        # Load trading pairs
        pairs_str = os.getenv('TRADING_PAIRS', '["BTCUSDT_UMCBL"]')
        self.TRADING_PAIRS = eval(pairs_str)
        
        # Override values from environment if provided
        for field in self.__dataclass_fields__:
            env_value = os.getenv(f'TRADING_{field}')
            if env_value is not None:
                # Convert to appropriate type
                field_type = type(getattr(self, field))
                if field_type == bool:
                    setattr(self, field, env_value.lower() == 'true')
                else:
                    try:
                        setattr(self, field, field_type(env_value))
                    except ValueError:
                        pass  # Keep default value if conversion fails

class AIConfig:
    """AI-specific configuration."""
    
    def __init__(self):
        self.AI_ANALYSIS_WEIGHT = float(os.getenv('AI_ANALYSIS_WEIGHT', '0.3'))
        self.AI_MIN_CONFIDENCE = float(os.getenv('AI_MIN_CONFIDENCE', '0.7'))
        
        # Model-specific settings
        self.GPT_MODEL = "gpt-4"
        self.GPT_MAX_TOKENS = 1000
        self.GPT_TEMPERATURE = 0.7
        
        self.GEMINI_MODEL = "gemini-pro"
        self.GEMINI_MAX_TOKENS = 1000
        self.GEMINI_TEMPERATURE = 0.7
        
        # Analysis parameters
        self.ANALYSIS_TIMEFRAMES = ["1h", "4h", "1d"]
        self.MIN_DATA_POINTS = 100
        self.HISTORICAL_CONTEXT_DAYS = 30
        
        # Feature engineering
        self.TECHNICAL_FEATURES = [
            "price_momentum",
            "volume_profile",
            "volatility_metrics",
            "trend_strength",
            "support_resistance"
        ]
        
        self.MARKET_FEATURES = [
            "correlation_matrix",
            "sector_performance",
            "market_sentiment",
            "volatility_regime"
        ]
        
        # Validation thresholds
        self.CONFIDENCE_THRESHOLD = 0.7
        self.AGREEMENT_THRESHOLD = 0.8  # Required agreement between models
        self.MAX_POSITION_HOLD_TIME = 48  # hours
        
        # Risk parameters
        self.MAX_DRAWDOWN_THRESHOLD = 0.1  # 10%
        self.POSITION_SIZE_FACTOR = 0.8  # Reduce size for AI-driven trades
        
def load_config() -> Dict[str, Any]:
    """Load and return all configuration parameters."""
    trading_config = TradingConfig()
    ai_config = AIConfig()
    
    return {
        'trading': trading_config,
        'ai': ai_config,
        'version': '2.0.0'
    }

# Global configuration instance
CONFIG = load_config() 