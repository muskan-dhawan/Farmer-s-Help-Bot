import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import requests
import warnings
import logging
import io
import json
import urllib.parse
import re
from functools import lru_cache
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from database_util import DatabaseManager

load_dotenv()

# ===============================
# 🔧 GLOBALS & SETUP
# ===============================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CROP_YIELD_PATH = os.path.join(DATA_DIR, 'crop_yield.csv')
DB_PATH = os.getenv("BOT_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'farmer_bot.sqlite3'))
translation_cache = {}
rag_model = None

def load_state_crop_map():
    try:
        df = pd.read_csv(CROP_YIELD_PATH)
        if not {'state', 'crop'}.issubset(df.columns):
            return {}

        top_crops = (
            df.groupby("state")["crop"]
              .value_counts()
              .groupby(level=0)
              .head(5)
        )

        state_crops = {}
        for (state, crop), _count in top_crops.items():
            state_crops.setdefault(str(state).lower(), []).append(str(crop).capitalize())

        return state_crops
    except Exception as e:
        logging.warning(f"Dynamic crop map unavailable: {e}")
        return {}

STATE_CROP_MAP = load_state_crop_map()
db = DatabaseManager(DB_PATH)
db.cleanup_expired(expiry_hours=48)

SOIL_DATA_PATH = os.path.join(DATA_DIR, 'state_soil_data.csv')
try:
    soil_df = pd.read_csv(SOIL_DATA_PATH)
    soil_df['state_lower'] = soil_df['state'].str.lower()
except Exception as e:
    logging.error(f"Failed to load soil data: {e}")
    soil_df = None

try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model = pickle.load(open(os.path.join(model_path, 'crop_pipeline.pkl'), 'rb'))
    label_encoder = pickle.load(open(os.path.join(model_path, 'label_encoder.pkl'), 'rb'))
    features = pickle.load(open(os.path.join(model_path, 'features.pkl'), 'rb'))
except FileNotFoundError:
    sys.exit("❌ Model not found. Run training first.")

# ===============================
# 🔒 MEMORY & SECURITY (Phase 6.5 & 7.5)
# ===============================
user_memory = {}
user_requests = {}

def check_spam(user_id):
    now = time.time()
    if user_id not in user_requests:
        user_requests[user_id] = []
    
    # Keep only requests within the last 60 seconds
    user_requests[user_id] = [t for t in user_requests[user_id] if now - t < 60]
    
    # Limit max 5 per minute
    if len(user_requests[user_id]) >= 5:
        return True
        
    user_requests[user_id].append(now)
    return False

def save_user_context(user_id, state, lat, lon):
    context = user_memory.get(user_id, {})
    context.update({
        'state': state, 
        'lat': lat, 
        'lon': lon,
        'timestamp': time.time()
    })
    user_memory[user_id] = context
    db.upsert_user(user_id, state=state, lat=lat, lon=lon, timestamp=context['timestamp'])

def set_user_mode(user_id, mode):
    context = user_memory.get(user_id, {})
    if mode is None:
        context.pop('mode', None)
    else:
        context['mode'] = mode
    user_memory[user_id] = context
    db.upsert_user(user_id, mode=mode)

def save_user_language(user_id, lang):
    context = user_memory.get(user_id, {})
    context['lang'] = lang
    user_memory[user_id] = context
    db.upsert_user(user_id, lang=lang)

def save_user_last_crop(user_id, crop):
    context = user_memory.get(user_id, {})
    context['last_crop'] = crop
    user_memory[user_id] = context
    db.upsert_user(user_id, last_crop=crop)

def get_user_language(user_id):
    context = get_user_context(user_id) or {}
    return context.get('lang', 'en')

def get_user_context(user_id):
    context = user_memory.get(user_id)
    if context:
        return context

    context = db.get_user(user_id)
    if context:
        user_memory[user_id] = context
    return context

def is_location_expired(user_id, expiry_hours=24):
    context = get_user_context(user_id)
    if not context:
        return True
    
    last_time = context.get('timestamp', 0)
    current_time = time.time()
    hours_passed = (current_time - last_time) / 3600
    return hours_passed > expiry_hours

def safe_request(url, timeout=5, headers=None):
    # Phase 7 Reliability Wrapper
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            logging.warning(f"API timeout on attempt {attempt + 1}/3: {e}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code and 400 <= status_code < 500:
                logging.error(f"API client error {status_code}: {e}")
                return None
            logging.warning(f"API server error on attempt {attempt + 1}/3: {e}")
        except Exception as e:
            logging.error(f"API Request Failed: {e}")
            return None

        if attempt < 2:
            time.sleep(1 * (attempt + 1))

    return None

def cache_translation(key, value, max_size=1000):
    if len(translation_cache) >= max_size:
        translation_cache.pop(next(iter(translation_cache)))
    translation_cache[key] = value

# ===============================
# 🌐 TRANSLATION
# ===============================
def translate_to_english(text):
    if len(text.strip()) < 5:
        return text, 'en'

    cache_key = ("to_en", text)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        lang = detect(text)
        # Phase 6: Language fallback strict check
        allowed_langs = ['hi','en','ta','te','bn','mr','gu','kn','ml','pa','or']
        if lang not in allowed_langs:
            lang = 'en'
            
        translated = GoogleTranslator(source=lang, target='en').translate(text)
        result = (translated, lang)
        cache_translation(cache_key, result)
        return result
    except Exception as e:
        logging.error(f"Translation detect error: {e}")
        return text, 'en'

def translate_back(text, lang):
    if lang == 'en': return text

    cache_key = ("from_en", text, lang)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        translated = GoogleTranslator(source='en', target=lang).translate(text)
        cache_translation(cache_key, translated)
        return translated
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return text

def translate_with_prefix(prefix, text, lang):
    return prefix + translate_back(text, lang)

# ===============================
# 💬 BOT PROMPTS (Phase 6 Dynamic Translation)
# ===============================
def get_start_message():
    return """🙏 Namaste!

Please talk in your preferred language (Hindi, Gujarati, Tamil, etc.)

👉 You can type in your own language  
👉 I will understand and reply in the same language

💡 Example:
- "मैं उत्तर प्रदेश में हूँ"
- "હું અમદાવાદમાં છું"
- "I am in Punjab"
"""

def get_location_prompt_en():
    return """📍 Please share your location

👉 Steps:
1. Tap 📎 (Attach)
2. Select "Location"
3. Tap "Send Current Location"

💡 This helps me give accurate farming advice
"""

# ===============================
# 📍 LOCATION APIS
# ===============================
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
        except (KeyError, TypeError, ValueError) as e:
            logging.error(f"Location parse error: {e}")
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

# ===============================
# 🌦️ WEATHER API
# ===============================
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

# ===============================
# 🌱 SOIL
# ===============================
def get_soil(state_name):
    N, P, K, pH = 50, 40, 40, 6.5
    if not state_name or soil_df is None: return N, P, K, pH
    try:
        match = soil_df[soil_df['state_lower'] == state_name.lower()]
        if not match.empty:
            row = match.iloc[0]
            return row['N'], row['P'], row['K'], row['pH']
    except Exception as e:
        logging.error(f"Soil processing error: {e}")
    return N, P, K, pH

def soil_to_text(N, P, K, pH):
    return f"""
Soil Nutrients:
- Nitrogen: {N}
- Phosphorus: {P}
- Potassium: {K}
- pH: {pH}

Interpretation:
- Suitable for moderate fertility crops
- Neutral pH supports most crops
"""
# ===============================
# 🧠 PREDICT CORE
# ===============================
def get_confidence_label(conf):
    if conf >= 0.60: return "Best Choice"
    if conf >= 0.30: return "Good Option"
    return "Tryable Option"

def get_state_crop_filter(state_name):
    if not state_name:
        return None
    return STATE_CROP_MAP.get(state_name.lower())

def infer_crop_from_text(text):
    if not text:
        return None

    text_lower = text.lower()
    known_crops = set()
    for crops in STATE_CROP_MAP.values():
        known_crops.update(crops)

    try:
        known_crops.update(str(crop).capitalize() for crop in label_encoder.classes_)
    except Exception:
        pass

    for crop in known_crops:
        if crop.lower() in text_lower:
            return crop

    return None

def predict_crop(lat, lon, state_name):
    temp, humidity, rainfall = get_weather(lat, lon)
    N, P, K, ph = get_soil(state_name)

    input_data = [N, P, K, temp, humidity, ph, rainfall]

    probs = model.predict_proba([input_data])[0]
    ranked_idx = list(np.argsort(probs)[::-1])
    state_filter = get_state_crop_filter(state_name)
    state_filter_set = {c.lower() for c in state_filter} if state_filter else None
    filtered_idx = []

    for idx in ranked_idx:
        crop = str(label_encoder.inverse_transform([idx])[0])
        if state_filter_set and crop.lower() not in state_filter_set:
            continue
        filtered_idx.append(idx)

        if len(filtered_idx) == 3:
            break

    if state_filter and not filtered_idx:
        logging.warning(f"No model crops matched state filter for {state_name}; using raw ML ranking.")
        filtered_idx = ranked_idx[:3]
    elif not state_filter:
        filtered_idx = ranked_idx[:3]

    results = []
    for idx in filtered_idx:
        crop = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]
        conf_label = get_confidence_label(confidence)
        results.append((str(crop).capitalize(), conf_label))

    return results

# ===============================
# 📚 RAG & LLM (OLLAMA) PRO-LEVEL
# ===============================
def get_problem_keywords(problem_text):
    if not problem_text:
        return []

    text = problem_text.lower()
    keyword_groups = {
        "pest": ["pest", "keeda", "kiit", "insect", "worm", "armyworm", "bollworm"],
        "disease": ["disease", "rog", "bimari", "fungus", "fungal", "rot"],
        "water": ["water", "paani", "irrigation", "moisture"],
        "fertilizer": ["fertilizer", "khaad", "nitrogen", "phosphorus", "potassium", "nutrient"],
        "seed": ["seed", "beej", "sowing"],
    }

    keywords = []
    for main_word, aliases in keyword_groups.items():
        if any(alias in text for alias in aliases):
            keywords.extend([main_word] + aliases)

    return list(dict.fromkeys(keywords))

def get_rag_model():
    global rag_model
    if rag_model is not None:
        return rag_model

    try:
        from sentence_transformers import SentenceTransformer
        rag_model = SentenceTransformer("all-MiniLM-L6-v2")
        return rag_model
    except Exception as e:
        logging.warning(f"FAISS RAG embedding model unavailable: {e}")
        return None

def semantic_rag_search(query, documents, top_k=3):
    if not query or not documents:
        return []

    try:
        import faiss

        embedding_model = get_rag_model()
        if embedding_model is None:
            return []

        embeddings = embedding_model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        _scores, indices = index.search(query_embedding.astype("float32"), min(top_k, len(documents)))

        return [documents[i] for i in indices[0] if i >= 0]
    except Exception as e:
        logging.warning(f"FAISS RAG search unavailable, using keyword fallback: {e}")
        return []

def fetch_kcc_data(crop, state, problem_text=None):
    kcc_key = os.getenv("KCC_API_KEY")
    crop_label = crop or "local crops"

    if not kcc_key or not state:
        return (
            "- No exact KCC match found because crop/location data is incomplete.\n"
            "- General safe advice: share your crop, state, and problem details before applying any pesticide or fertilizer."
        )

    try:
        # 🔥 API expects UPPERCASE state
        state_api = state.upper()
        problem_keywords = get_problem_keywords(problem_text)

        url = (
            "https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f"
            f"?api-key={kcc_key}"
            "&format=json"
            "&limit=10"
            f"&filters[StateName]={urllib.parse.quote(state_api)}"
        )

        data = safe_request(url, timeout=45)

        if not data:
            return f"- No recent KCC data available for {state}."

        records = data.get("records", [])

        rag_documents = []
        exact_context = []
        crop_context = []

        for r in records:
            query = r.get("QueryText", "")
            answer = r.get("KccAns", "")
            combined_text = (query + " " + answer).lower()

            # 🔥 Filter by crop manually
            crop_matches = not crop or crop.lower() in combined_text
            if crop_matches and query.strip() and answer.strip():
                formatted = f"- Q: {query}\n  A: {answer}"
                rag_documents.append(formatted)
                if problem_keywords:
                    if any(keyword in combined_text for keyword in problem_keywords):
                        exact_context.append(formatted)
                    else:
                        crop_context.append(formatted)
                else:
                    crop_context.append(formatted)

        rag_query = " ".join(part for part in [str(crop or ""), str(state or ""), str(problem_text or "")] if part.strip())
        rag_context = semantic_rag_search(rag_query, rag_documents)
        if rag_context:
            logging.info(f"FAISS RAG returned {len(rag_context)} KCC results for {state}.")
            return "\n".join(rag_context)

        if exact_context:
            return "\n".join(exact_context[:3])

        if crop_context:
            return "\n".join(crop_context[:3])

    except Exception as e:
        logging.error(f"KCC API error: {e}")

    problem_note = f" for {problem_text}" if problem_text else ""
    return (
        f"- No exact KCC match found{problem_note} in {crop_label}, {state}.\n"
        "- General safe advice: inspect the field closely, avoid random chemical use, and contact a local agriculture officer if symptoms spread."
    )

def get_knowledge(crop):
    try:
        kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'knowledge', f'{crop.lower()}.txt')
        with open(kb_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"{crop} generally yields best in well-drained soils and requires moderate fertilizer application."

def filter_kcc_context(kcc_data, crop):
    lines = kcc_data.split("\n")
    filtered = [line for line in lines if crop.lower() in line.lower()]
    if not filtered: 
        return kcc_data 
    return "\n".join(filtered[:5])

def build_prompt(crop, weather, soil, knowledge, kcc_data):
    return f"""
You are an expert agricultural assistant for Indian farmers.

STRICT RULES:
- Use ONLY the given data
- Do NOT assume anything
- If information missing → say "Data not available"
- Answer in simple farmer language
- Keep answer short and practical

Crop: {crop}

Weather Data:
{weather}

Soil Data:
{soil}

Scientific Knowledge:
{knowledge}

Farmer Experience (KCC):
{kcc_data}

Tasks:
1. Why crop is suitable
2. Sowing time
3. 2–3 practical tips
4. One warning (if any)

Do NOT add anything outside given data.
"""

def build_general_prompt(state, crop_focus, weather, soil, kcc_data):
    return f"""
You are an expert agricultural assistant for Indian farmers.

Use only the data below. Give short, practical farming tips.

State: {state}
Crop focus: {crop_focus}

Weather Data:
{weather}

Soil Data:
{soil}

Farmer Experience (KCC):
{kcc_data}

Tasks:
1. 3 practical tips for current farming care
2. One water/fertilizer caution
3. One safety warning
"""

def ask_llm(prompt):
    payload = {
        "model": "llama3", 
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()['response']

def safe_llm_call(prompt, fallback="🌾 Crop suitable. Follow standard farming practices. Detailed advice unavailable."):
    try:
        return ask_llm(prompt)
    except Exception as e:
        logging.warning(f"Ollama local LLM unavailable/timeout: {e}")
        return fallback

def simplify_kcc_advice(kcc_data):
    prompt = f"""
Explain this in simple farmer language.
Keep it short, practical, and safe.

{kcc_data}
"""
    simple = safe_llm_call(prompt, fallback=None)
    if simple:
        return simple

    return (
        kcc_data
        .replace("- Q:", "-")
        .replace("Q:", "")
        .replace("  A:", "  ")
        .replace("A:", "")
    )

def clean_response(text):
    lines = text.split("\n")
    clean = [l for l in lines if len(l.strip()) > 5]
    return "\n".join(clean[:10])

def generate_advice(lat, lon, state):
    results = predict_crop(lat, lon, state)
    best_crop = results[0][0] 

    weather = get_weather(lat, lon)
    N, P, K, pH = get_soil(state)
    soil_text = soil_to_text(N, P, K, pH)

    knowledge = get_knowledge(best_crop)
    kcc_data = fetch_kcc_data(best_crop, state)
    kcc_data = filter_kcc_context(kcc_data, best_crop)

    prompt = build_prompt(best_crop, weather, soil_text, knowledge, kcc_data)

    answer = safe_llm_call(prompt)
    answer = clean_response(answer)

    return results, answer

def generate_general_advice(lat, lon, state, crop_focus="local crops", user_query=None):
    weather = get_weather(lat, lon)
    N, P, K, pH = get_soil(state)
    soil_text = soil_to_text(N, P, K, pH)
    rag_query = f"{state} farming tips {user_query or ''}".strip()
    kcc_data = fetch_kcc_data(None, state, rag_query)

    prompt = build_general_prompt(state, crop_focus, weather, soil_text, kcc_data)
    if user_query:
        prompt += f"\nFarmer question:\n{user_query}\n"

    answer = safe_llm_call(
        prompt,
        fallback=(
            "General farming tips unavailable from the local LLM.\n"
            "Keep field moisture consistent, inspect crops weekly, and avoid applying chemicals without identifying the exact problem."
        )
    )
    return clean_response(answer)

def final_response(lat, lon, state, user_id):
    results, advice = generate_advice(lat, lon, state)
    
    # Save the best crop to memory for follow-up help
    best_crop = results[0][0]
    save_user_last_crop(user_id, best_crop)

    reply = "🌾 Top crop recommendations for your area:\n"
    for crop, conf_label in results:
        reply += f"✅ {crop} ({conf_label})\n"
        
    reply += f"\n📘 Advice:\n{advice}"
    return reply

# ===============================
# 🤖 INTENT DETECTION (Phase 8 Optimization)
# ===============================
def detect_intent(text):
    text = text.lower()

    if text.strip() in ["hi", "hello", "namaste", "start", "hey"]:
        return "greeting"

    # Location indicators
    if any(phrase in text for phrase in ["i am in", "i am at", "i live in", "from "]):
        return "location"

    if re.fullmatch(r"in\s+[a-zA-Z\s]+", text.strip()):
        return "location"

    # 🔥 Fix 2: Detect problem (pest, disease, water, fertilizer, seed)
    if any(word in text for word in ["keeda", "kiit", "pest", "disease", "rog", "bimari", "paani", "irrigation", "khaad", "fertilizer", "beej", "seed", "yellow", "leaf", "leaves", "spots", "rot"]):
        return "problem"

    # Help indicators
    if any(word in text for word in ["help", "assistant", "madad"]):
        return "help"

    # Prediction Indicators
    if any(word in text for word in ["crop", "suggest", "kya ugaye", "fasal", "sowing", "plantation"]):
        return "crop"

    # General advice indicators
    if any(word in text for word in ["tips", "tricks", "advice", "kaise kare"]):
        return "general"

    if re.fullmatch(r"[a-zA-Z\s]+", text.strip()) and len(text.split()) <= 3:
        return "location"
    
    return "unknown"

# ===============================
# 💬 UNIFIED CHAT HANDLER
# ===============================
def chatbot_response(user_id="default_user", user_input=None, lat=None, lon=None):
    
    # 1. Security & Spam
    if check_spam(user_id):
        return translate_back("⚠️ You're sending messages too fast. Please wait a minute. 🙏", get_user_language(user_id))
        
    if user_input and len(user_input) > 200:
        return translate_back("⚠️ Please keep your message under 200 characters.", get_user_language(user_id))

    # 2. Greeting / Start Logic
    # 🔥 Fix 1: Removed 'help' from greetings
    greetings = ['hi', 'hello', 'namaste', 'start', 'hey', 'नमस्ते']
    if user_input and user_input.lower().strip() in greetings:
        translated, lang = translate_to_english(user_input)
        save_user_language(user_id, lang)
        return translate_back(get_start_message(), lang)

    # 3. Location coordinates received (Strong Intent Pillar)
    if lat is not None and lon is not None:
        lang = get_user_language(user_id)
        lat, lon, state = handle_location(lat, lon)
        if state:
            save_user_context(user_id, state, lat, lon)
            return translate_with_prefix(
                "📍 ",
                f"""Location set: {state}

Now you can ask:
Crop suggestion
Disease problem
Farming tips""",
                lang
            )
        return translate_with_prefix(
            "📍 ",
            "Location received, but I could not detect your state. Please type your state name.",
            lang
        )

    # 4. Text Input Processing (Dynamic Intent Routing)
    if user_input:
        translated, lang = translate_to_english(user_input)
        save_user_language(user_id, lang)
        logging.info(f"[USER {user_id}] Lang: '{lang}', Query: '{translated}'")

        # 🔥 Intent Detection
        intent = detect_intent(translated)
        context = get_user_context(user_id)
        if context and context.get('mode') == "help" and intent == "unknown":
            intent = "problem"
        logging.info(f"[USER {user_id}] Intent: {intent}")

        # 🔥 Fix 3: BETTER STRUCTURE (RAG-FIRST)
        # Branch 0: Translated greeting/start.
        if intent == "greeting":
            return translate_back(get_start_message(), lang)

        # Branch A: Location memory update.
        if intent == "location":
            location_query = extract_location_query(translated.lower())
            lat, lon, state = get_coordinates(location_query)

            if lat is None or lon is None or not state:
                return translate_back(
                    "I could not detect your location. Please share your state name or send current location.",
                    lang
                )

            save_user_context(user_id, state, lat, lon)
            return translate_with_prefix(
                "📍 ",
                f"""Location set: {state}

Now you can ask:
Crop suggestion
Disease problem
Farming tips""",
                lang
            )

        # Branch B: Generic help. Ask for the problem before calling KCC.
        if intent == "help":
            set_user_mode(user_id, "help")
            help_reply = """🙏 What help do you need?

👉 Disease
👉 Water
👉 Fertilizer
👉 Seeds

Please write your problem."""
            return translate_back(help_reply, lang)

        # Branch C: Stated farm problem. Use KCC/RAG.
        elif intent == "problem":
            set_user_mode(user_id, "help")
            context = get_user_context(user_id)
            crop = infer_crop_from_text(translated) or (context.get('last_crop') if context else None)
            state = context.get('state') if context else None

            if not state:
                loc_lat, loc_lon, loc_state = get_coordinates(extract_location_query(translated.lower()))
                if loc_lat is not None and loc_lon is not None and loc_state:
                    save_user_context(user_id, loc_state, loc_lat, loc_lon)
                    context = get_user_context(user_id)
                    state = loc_state

            if crop:
                save_user_last_crop(user_id, crop)

            if not crop or not state:
                missing_reply = """Please share crop and location first.

Example:
Cotton pest problem in Gujarat
Wheat water problem in Punjab"""
                return translate_back(missing_reply, lang)

            kcc_advice = fetch_kcc_data(crop, state, translated)
            kcc_advice = simplify_kcc_advice(kcc_advice)
            set_user_mode(user_id, None)
            return translate_back(kcc_advice, lang)

        # Branch D: General farming tips using KCC + LLM.
        elif intent == "general":
            context = get_user_context(user_id)
            if not context or is_location_expired(user_id):
                return translate_back(get_location_prompt_en(), lang)

            set_user_mode(user_id, "help")
            last_crop = context.get('last_crop', 'local crops')
            advice = generate_general_advice(context['lat'], context['lon'], context['state'], last_crop, translated)
            return translate_back(advice, lang)

        # Branch E: Crop prediction
        elif intent == "crop":
            set_user_mode(user_id, "crop")
            lat, lon, state = get_coordinates(translated)
            if lat is None or lon is None:
                location_query = extract_location_query(translated.lower())
                if location_query != translated.lower():
                    lat, lon, state = get_coordinates(location_query)

            if lat is None or lon is None:
                context = get_user_context(user_id)
                if context:
                    if is_location_expired(user_id):
                        prompt_en = "📍 Location update required\n" + get_location_prompt_en()
                        return translate_back(prompt_en, lang)
                    lat, lon, state = context['lat'], context['lon'], context['state']
                else:
                    return translate_back(get_location_prompt_en(), lang)
            else:
                save_user_context(user_id, state, lat, lon)

            reply = final_response(lat, lon, state, user_id)
            return translate_back(reply, lang)

        # Branch F: Clarification
        else:
            clarification = """Please tell me what you need:

👉 Crop suggestion
👉 Disease/pest help
👉 Water advice
👉 Fertilizer advice

Please include your location if you want crop suggestion."""
            return translate_back(clarification, lang)

    # 🔥 Fix 4: No duplicate return here
    return translate_back(get_location_prompt_en(), get_user_language(user_id))

if __name__ == "__main__":
    logging.info("WhatsApp Bot running locally.")
    print("-------------------------------------------------")
    # Simulating first chat establishing Memory
    print("User: હું અમદાવાદમાં છું (I am in Ahmedabad)")
    print("Bot:\n" + chatbot_response(user_id="user_123", user_input="હું અમદાવાદમાં છું"))
    print("-------------------------------------------------")
    # Simulating second chat utilizing Memory (User sends no location string, just generic query)
    print("User (Generic Memory Test): suggest best farming tricks")
    print("Bot:\n" + chatbot_response(user_id="user_123", user_input="suggest best farming tricks"))
    print("-------------------------------------------------")
