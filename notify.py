import json
import os
import requests

def main():
    # Lade Konfiguration
    with open('config.json') as f:
        config = json.load(f)
    
    bot_token = config.get('TELEGRAM_BOT_TOKEN')
    chat_id = config.get('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        status = os.getenv('JOB_STATUS', 'unknown')
        message = '✅ Deployment erfolgreich abgeschlossen' if status == 'success' else '❌ Deployment fehlgeschlagen'
        
        requests.post(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
        )

if __name__ == '__main__':
    main() 