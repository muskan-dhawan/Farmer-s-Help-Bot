import requests
import logging
from functools import lru_cache

def safe_request(url, timeout=5, headers=None):
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"API Request Failed: {e}")
        if attempt < 2:
            import time
            time.sleep(1 * (attempt + 1))
    return None

@lru_cache(maxsize=100)
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
    data = safe_request(url)
    if data:
        current = data.get('current', {})
        temp = current.get('temperature_2m', 28.0)
        humidity = current.get('relative_humidity_2m', 60.0)
        rainfall = current.get('precipitation', 100.0)
        if rainfall == 0: rainfall = 50.0  
        return temp, humidity, rainfall
    return 28.0, 60.0, 100.0
