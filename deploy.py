import json
import os

def main():
    # Lade Konfiguration
    with open('config.json') as f:
        config = json.load(f)
    
    # Erstelle .env Datei
    with open('.env', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}={value}\n')
    
    # Installiere Abhängigkeiten
    os.system('pip install -r requirements.txt')
    
    # Starte den Bot
    os.system('python Trading.py')

if __name__ == '__main__':
    main() 