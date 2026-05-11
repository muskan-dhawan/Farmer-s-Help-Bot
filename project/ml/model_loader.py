import os
import pickle
import sys
import logging

def load_ml_models():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        model = pickle.load(open(os.path.join(model_path, 'crop_pipeline.pkl'), 'rb'))
        label_encoder = pickle.load(open(os.path.join(model_path, 'label_encoder.pkl'), 'rb'))
        features = pickle.load(open(os.path.join(model_path, 'features.pkl'), 'rb'))
        return model, label_encoder, features
    except FileNotFoundError:
        logging.error("❌ Model files not found in 'models/' directory.")
        return None, None, None
    except Exception as e:
        logging.error(f"❌ Error loading models: {e}")
        return None, None, None
