import urllib.parse
import re
import logging
import requests
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
def handle_location(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    data = safe_request(url, headers={'User-Agent': 'crop-bot/1.0'})
    if data:
        state = data.get('address', {}).get('state', None)
        return lat, lon, state
    return lat, lon, None

@lru_cache(maxsize=100)
def get_coordinates(place):
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(place)}&format=json&addressdetails=1"
    data = safe_request(url, headers={'User-Agent': 'crop-bot/1.0'})
    if isinstance(data, list) and data:
        try:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            state = data[0].get('address', {}).get('state', None)
            return lat, lon, state
        except Exception:
            pass
    return None, None, None

def extract_location_query(text):
    text = text.strip()
    patterns = [
        r"\bi am in\s+(.+)",
        r"\bi am at\s+(.+)",
        r"\bi live in\s+(.+)",
        r"\bfrom\s+([a-zA-Z\s]+)",
        r"\bin\s+([a-zA-Z\s]+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            location = match.group(1)
            location = re.split(r"\bindia\b", location, flags=re.IGNORECASE)[0].strip()
            return location.strip(" .,!?:;")

    return text
