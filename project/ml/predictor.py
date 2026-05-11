import pandas as pd
import numpy as np
import logging
import os
from ml.model_loader import load_ml_models
from services.weather import get_weather

model, label_encoder, features = load_ml_models()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CROP_YIELD_PATH = os.path.join(DATA_DIR, 'crop_yield.csv')
SOIL_DATA_PATH = os.path.join(DATA_DIR, 'state_soil_data.csv')

def load_state_crop_map():
    try:
        df = pd.read_csv(CROP_YIELD_PATH)
        top_crops = df.groupby("state")["crop"].value_counts().groupby(level=0).head(5)
        state_crops = {}
        for (state, crop), _count in top_crops.items():
            state_crops.setdefault(str(state).lower(), []).append(str(crop).capitalize())
        return state_crops
    except Exception:
        return {}

STATE_CROP_MAP = load_state_crop_map()

def get_soil(state_name):
    N, P, K, pH = 50, 40, 40, 6.5
    try:
        soil_df = pd.read_csv(SOIL_DATA_PATH)
        match = soil_df[soil_df['state'].str.lower() == state_name.lower()]
        if not match.empty:
            row = match.iloc[0]
            return row['N'], row['P'], row['K'], row['pH']
    except Exception:
        pass
    return N, P, K, pH

def predict_crop(lat, lon, state_name):
    if not model: return []
    
    temp, humidity, rainfall = get_weather(lat, lon)
    N, P, K, ph = get_soil(state_name)
    input_data = [N, P, K, temp, humidity, ph, rainfall]

    probs = model.predict_proba([input_data])[0]
    ranked_idx = list(np.argsort(probs)[::-1])
    
    state_filter = STATE_CROP_MAP.get(state_name.lower())
    state_filter_set = {c.lower() for c in state_filter} if state_filter else None
    
    results = []
    for idx in ranked_idx:
        crop = str(label_encoder.inverse_transform([idx])[0])
        if state_filter_set and crop.lower() not in state_filter_set:
            continue
        confidence = probs[idx]
        results.append((crop.capitalize(), "Best Choice" if confidence > 0.6 else "Good Option"))
        if len(results) == 3: break
        
    return results
