import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Zentrale Konfigurationsverwaltung für den Trading Bot.
    Lädt Konfigurationen aus .env Dateien und JSON-Konfigurationsdateien.
    Implementiert Singleton-Pattern für konsistente Konfigurationsverwaltung.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.config: Dict[str, Any] = {}
        self._load_environment()
        self._load_config_files()
        self._validate_config()

    def _load_environment(self) -> None:
        """Lädt Umgebungsvariablen aus .env Datei"""
        try:
            env_path = Path('.env')
            if env_path.exists():
                load_dotenv(env_path)
                logger.info("Umgebungsvariablen aus .env geladen")
            else:
                logger.warning(".env Datei nicht gefunden")

            # Lade kritische Konfigurationswerte
            self.config['API'] = {
                'BITGET_API_KEY': os.getenv('BITGET_API_KEY'),
                'BITGET_SECRET_KEY': os.getenv('BITGET_SECRET_KEY'),
                'BITGET_PASSPHRASE': os.getenv('BITGET_PASSPHRASE'),
                'BITGET_API_BASE': os.getenv('BITGET_API_BASE', 'https://api.bitget.com')
            }

            self.config['TELEGRAM'] = {
                'BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
                'CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
                'GROUP_ID': os.getenv('TELEGRAM_GROUP_ID')
            }

            self.config['AI'] = {
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
                'GPT4_MAX_TOKENS': int(os.getenv('GPT4_MAX_TOKENS', 8000)),
                'GPT4_TEMPERATURE': float(os.getenv('GPT4_TEMPERATURE', 0.5)),
                'GEMINI_MAX_TOKENS': int(os.getenv('GEMINI_MAX_TOKENS', 4096)),
                'GEMINI_TEMPERATURE': float(os.getenv('GEMINI_TEMPERATURE', 0.5))
            }

        except Exception as e:
            logger.error(f"Fehler beim Laden der Umgebungsvariablen: {str(e)}")
            raise

    def _load_config_files(self) -> None:
        """Lädt Konfigurationen aus JSON-Dateien"""
        config_dir = Path('config')
        if not config_dir.exists():
            logger.warning("Konfigurationsverzeichnis nicht gefunden")
            return

        try:
            # Lade Trading-Konfiguration
            trading_config_path = config_dir / 'trading_config.json'
            if trading_config_path.exists():
                with open(trading_config_path) as f:
                    self.config['TRADING'] = json.load(f)
            
            # Lade technische Analyse Konfiguration
            ta_config_path = config_dir / 'technical_analysis_config.json'
            if ta_config_path.exists():
                with open(ta_config_path) as f:
                    self.config['TA'] = json.load(f)

            # Lade Risikomanagement-Konfiguration
            risk_config_path = config_dir / 'risk_management_config.json'
            if risk_config_path.exists():
                with open(risk_config_path) as f:
                    self.config['RISK'] = json.load(f)

        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfigurationsdateien: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """Überprüft ob alle notwendigen Konfigurationswerte vorhanden sind"""
        required_configs = {
            'API': ['BITGET_API_KEY', 'BITGET_SECRET_KEY', 'BITGET_PASSPHRASE'],
            'TELEGRAM': ['BOT_TOKEN', 'CHAT_ID', 'GROUP_ID'],
            'TRADING': ['TRADING_PAIRS', 'ANALYSIS_TIMEFRAMES']
        }

        for section, keys in required_configs.items():
            if section not in self.config:
                raise ValueError(f"Fehlender Konfigurationsabschnitt: {section}")
            
            for key in keys:
                if not self.config[section].get(key):
                    raise ValueError(f"Fehlender Konfigurationswert: {section}.{key}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Holt einen Konfigurationswert aus dem spezifizierten Abschnitt
        
        Args:
            section: Konfigurationsabschnitt (z.B. 'API', 'TRADING')
            key: Schlüssel des Konfigurationswerts
            default: Standardwert falls nicht gefunden

        Returns:
            Konfigurationswert oder default wenn nicht gefunden
        """
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """
        Holt einen kompletten Konfigurationsabschnitt
        
        Args:
            section: Name des Konfigurationsabschnitts

        Returns:
            Dict mit allen Konfigurationswerten des Abschnitts oder None
        """
        return self.config.get(section)

    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Aktualisiert einen Konfigurationswert
        
        Args:
            section: Konfigurationsabschnitt
            key: Schlüssel des zu aktualisierenden Werts
            value: Neuer Wert
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        logger.info(f"Konfiguration aktualisiert: {section}.{key}")

    def save_config(self, section: str) -> None:
        """
        Speichert einen Konfigurationsabschnitt in die entsprechende JSON-Datei
        
        Args:
            section: Zu speichernder Konfigurationsabschnitt
        """
        if section not in self.config:
            raise ValueError(f"Unbekannter Konfigurationsabschnitt: {section}")

        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        filename = f"{section.lower()}_config.json"
        config_path = config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config[section], f, indent=4)
            logger.info(f"Konfiguration gespeichert: {filename}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration {filename}: {str(e)}")
            raise 