


import os
import re
import time
import logging
import hashlib
import requests
from difflib import get_close_matches

from services.rag_service import RAGService
from services.kcc_api import fetch_kcc_data
from services.location import get_coordinates, extract_location_query, handle_location
from ml.predictor import predict_crop
from db.database_util import DatabaseManager

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ── env / globals ─────────────────────────────────────────────────────────────
OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODELS = [
    # 🔥 FAST + STABLE
    "deepseek/deepseek-chat:free",

    # 🔥 VERY RELIABLE SMALL MODEL
    "meta-llama/llama-3.2-3b-instruct:free",

    # ⚡ GOOD BACKUP
    "mistralai/mistral-7b-instruct:free",

    # ⚡ LIGHTWEIGHT FALLBACK
    "google/gemma-2b-it:free",

    # ⚡ LARGE BUT RATE LIMITED
    "meta-llama/llama-3.3-70b-instruct:free",

    # ⚡ EXPERIMENTAL (sometimes slow)
    "nousresearch/hermes-3-llama-3.1-8b:free"
]

db  = DatabaseManager(os.getenv("BOT_DB_PATH", "farmer_bot.sqlite3"))
rag = RAGService()


# ============================================================
# 2. CROP DETECTION  (utils.py + bot_logic.py — deduplicated)
# ============================================================

KNOWN_CROPS = [
    "wheat", "rice", "maize", "barley", "millet", "jowar", "bajra", "ragi",
    "gram", "chana", "lentil", "arhar", "tur", "moong", "urad",
    "cotton", "sugarcane", "mustard", "groundnut", "soybean", "sunflower",
    "potato", "onion", "tomato", "chilli", "peas", "brinjal",
    "cabbage", "cauliflower", "spinach", "carrot",
    "mango", "banana", "apple", "grapes", "orange", "papaya",
    "guava", "pomegranate", "tea", "coffee", "coconut",
]

CROP_SYNONYMS = {
    "paddy":   "rice",
    "basmati": "rice",
    "gehun":   "wheat",
    "makka":   "maize",
    "sarson":  "mustard",
    "aloo":    "potato",
    "pyaz":    "onion",
    "tamatar": "tomato",
    "mirchi":  "chilli",
}


def normalize_text(text: str) -> str:
    """Strip non-alpha chars and lowercase."""
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())


def detect_crop_from_text(text: str) -> str | None:
    """
    Returns the canonical crop name found in text, or None.
    Priority: synonym map → direct match → fuzzy match (cutoff 0.8).
    """
    t     = normalize_text(text)
    words = t.split()

    for w in words:
        if w in CROP_SYNONYMS:
            return CROP_SYNONYMS[w]

    for w in words:
        if w in KNOWN_CROPS:
            return w

    for w in words:
        match = get_close_matches(w, KNOWN_CROPS, n=1, cutoff=0.8)
        if match:
            return match[0]

    return None


# ============================================================
# 3. INTENT DETECTION  (intents.py merged into bot_logic.py pipeline)
#
#    Pipeline (in order):
#      Step 1 — exact / keyword rules  →  0 ms   (~90 % of messages)
#      Step 2 — LLM classification     →  2–5 s  (genuinely ambiguous only)
#      Step 3 — safe fallback "general"→  0 ms
#
#    NOTE: greeting is intentionally excluded here.
#    telegram_bot.py intercepts "hi / hello / namaste" before calling
#    chatbot_response(), so the intent pipeline never sees them.
# ============================================================

# keyword sets
_LOCATION_PHRASES = [
    "i am in", "i am at", "i'm in", "i'm at",
    "i live in", "from ", "mera gaon", "mere gaon",
]
_PROBLEM_WORDS = [
    "pest", "disease", "rog", "bimari", "kida", "keeda", "kiit",
    "yellow", "brown", "leaf", "leaves", "spot", "spots", "rot",
    "wilt", "wither", "dying", "insect", "worm", "fungus", "fungal",
    "attack", "damage", "problem", "issue",
    "कीड़ा", "बीमारी", "ਕੀੜਾ", "ਬਿਮਾਰੀ",
]
_CROP_REC_PHRASES = [
    "best crop", "which crop", "what crop", "what to grow",
    "suggest crop", "recommend crop", "kya ugaye", "kaun si fasal",
]
_CROP_REC_WORDS = [
    "suggest", "recommend", "sowing", "plantation",
    "kya ugaun", "fasal batao",
]
_GENERAL_WORDS = [
    "tips", "advice", "how to", "kaise kare", "kaise",
    "fertilizer", "khaad", "irrigation", "soil", "nutrient",
    "कैसे", "खाद",
]
_HELP_WORDS = [
    "help", "madad", "मदद", "ਸਹਾਇਤਾ",
]

_VALID_LLM_INTENTS = {
    "crop_recommendation", "problem", "general", "location", "greeting",
}


def _has(text, words):
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _detect_intent_rules(text: str) -> str | None:
    """
    Fast keyword-based intent detection.
    Returns intent string on a confident match, or None if ambiguous.
    """
    t     = text.lower().strip()
    words = t.split()

    # 1. Explicit location phrases
    if _has(t, _LOCATION_PHRASES):
        return "location"

    # 2. Problem keywords — highest farming priority
    if _has(t, _PROBLEM_WORDS):
        return "problem"

    # 3. Help request — ask user to describe their problem
    if _has(t, _HELP_WORDS):
        return "help"

    # 4. Specific crop-rec multi-word phrases
    if _has(t, _CROP_REC_PHRASES):
        return "crop_recommendation"

    # 5. Crop-rec single keywords
    if _has(t, _CROP_REC_WORDS):
        return "crop_recommendation"

    # 6. Crop word present in text → also covers "crop", "fasal", "फसल"
    if any(w in t for w in ["crop", "fasal", "फसल", "ਫਸਲ"]):
        return "crop_recommendation"

    # 7. General farming advice keywords
    if _has(t, _GENERAL_WORDS):
        return "general"

    # 8. A known crop name typed alone → treat as problem inquiry
    if detect_crop_from_text(t):
        return "problem"

    # 9. Pure alphabetic short text (≤ 3 words) with no other signal → location
    #    e.g. "Gujarat", "Punjab district"
    if re.fullmatch(r"[a-zA-Z\s]+", t) and len(words) <= 3:
        return "location"

    # 10. Ambiguous — defer to LLM
    return None


def _detect_intent_llm(text: str) -> str | None:
    """
    LLM-based intent classifier — called ONLY when rules return None.
    Uses max_tokens=10 and a tight 8 s timeout to keep latency low.
    """
    if not OPENROUTER_API_KEY:
        return None

    prompt = (
        "You are classifying a message from an Indian farmer.\n"
        "Reply with EXACTLY one word — nothing else:\n"
        "greeting | location | problem | crop_recommendation | general\n\n"
        f"Message: {text[:200].strip()}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in MODELS:
        for attempt in range(2):
            try:
                res = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                    },
                    timeout=8,
                )

                if res.status_code == 429:
                    time.sleep(1)
                    continue

                if res.status_code != 200:
                    logger.warning(f"[intent LLM] {model} → HTTP {res.status_code}")
                    break  # try next model

                raw   = res.json()["choices"][0]["message"]["content"]
                clean = re.sub(r"[^a-z_]", "", raw.strip().lower())

                if clean in _VALID_LLM_INTENTS:
                    logger.info(f"[intent LLM] '{text[:40]}' → {clean}")
                    return clean

                logger.warning(f"[intent LLM] unexpected: '{raw}'")

            except requests.exceptions.Timeout:
                logger.warning(f"[intent LLM] timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"[intent LLM] error: {e}")

    return None


def detect_intent(text: str) -> str:
    """
    Master intent router.

    Step 1 → rules  (0 ms)   — covers ~90% of messages
    Step 2 → LLM    (2–5 s)  — only for genuinely ambiguous text
    Step 3 → fallback "general"
    """
    intent = _detect_intent_rules(text)

    if intent is not None:
        logger.info(f"[intent rules] '{text[:40]}' → {intent}")
        return intent

    logger.info(f"[intent] ambiguous → calling LLM: '{text[:40]}'")
    llm_intent = _detect_intent_llm(text)

    if llm_intent:
        return llm_intent

    logger.info("[intent] fallback → general")
    return "general"


# ============================================================
# 4. LLM CLIENT  (bot_logic.py — hash cache, eviction, retry)
# ============================================================

_LLM_CACHE: dict[str, str] = {}


def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to OpenRouter and return the response text.
    Results are cached by MD5 hash of the prompt.
    Cache evicts the oldest 50% of entries when it exceeds 500 items.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("[LLM] OPENROUTER_API_KEY not set — skipping LLM call.")
        return ""

    key = _cache_key(prompt)
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in MODELS:
        for attempt in range(2):
            try:
                res = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert agriculture advisor for Indian farmers. "
                                    "Be concise, practical, and use simple language."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=15,
                )

                if res.status_code == 429:
                    time.sleep(2)
                    continue

                if res.status_code != 200:
                    logger.warning(f"[LLM] {model} → HTTP {res.status_code}")
                    break  # try next model

                result = res.json()["choices"][0]["message"]["content"]

                # evict oldest 50% when cache is full
                if len(_LLM_CACHE) >= 500:
                    for old_key in list(_LLM_CACHE.keys())[:250]:
                        del _LLM_CACHE[old_key]

                _LLM_CACHE[key] = result
                return result

            except requests.exceptions.Timeout:
                logger.warning(f"[LLM] timeout — model={model} attempt={attempt + 1}")
            except Exception as e:
                logger.error(f"[LLM] error: {e}")

    return ""


# ============================================================
# 5. HANDLERS  (handlers.py — greeting, location, crop rec)
# ============================================================

def handle_greeting() -> str:
    """
    Returns the standard greeting message.
    NOTE: telegram_bot.py intercepts common greetings (hi/hello/namaste)
    before they reach chatbot_response(), so this is called only when
    the intent pipeline routes here (e.g. translated greeting text).
    """
    return (
        "🙏 Namaste! I am your farming assistant.\n\n"
        "You can ask me:\n"
        "🌾 Crop suggestions\n"
        "🐛 Pest or disease problems\n"
        "💧 Water and fertilizer advice\n\n"
        "Please share your location or state name to begin."
    )


def handle_location_intent(text: str, user_id: str) -> str:
    """
    Extracts a location from text, geocodes it, and saves to DB.
    Returns a confirmation or error message.
    """
    location_query = extract_location_query(text)
    lat, lon, state = get_coordinates(location_query)

    if lat is not None and state:
        db.upsert_user(user_id, state=state, lat=lat, lon=lon, mode="problem")
        logger.info(f"[location] user={user_id} state={state} lat={lat} lon={lon}")
        return f"📍 Location set: {state}.\n\nNow tell me your farming problem."

    return (
        "⚠️ Could not detect your location.\n"
        "Please type your state name (e.g. Gujarat, Punjab, Maharashtra)."
    )


def handle_crop_recommendation(lat, lon, state: str) -> str:
    """
    Runs the ML crop predictor and formats the top-3 results.
    """
    if lat is None or lon is None or not state:
        return "📍 Please share your location first so I can suggest the right crops."

    results = predict_crop(lat, lon, state)

    if not results:
        return (
            "⚠️ Could not generate crop recommendations right now.\n"
            "Please ensure your location is set correctly."
        )

    reply = "🌾 Recommended crops for your area:\n\n"
    for crop, confidence in results:
        reply += f"✅ {crop} — {confidence}\n"

    reply += "\nReply with a crop name to get detailed advice."
    return reply



def handle_problem(user_id: str, text: str, context: dict) -> str:
    """
    Handles a farming problem report.
    Data priority: KCC API (high) → RAG/FAISS (medium) → LLM only (low).
    """
    # ── crop resolution ──────────────────────────────────────────────────────
    detected_crop = detect_crop_from_text(text)

    if detected_crop:
        crop = detected_crop
        db.upsert_user(user_id, last_crop=crop)
    else:
        crop = context.get("last_crop")

    if not crop:
        return (
            "🌾 Please tell me which crop has the problem.\n"
            "Example: 'wheat pest problem' or 'cotton yellow leaves'"
        )

    state = context.get("state")
    query = f"{crop} {text}"

    # ── data fetch ───────────────────────────────────────────────────────────
    kcc_results = fetch_kcc_data(crop, state, text)
    rag_results = rag.search(query, state, crop)

    has_kcc = kcc_results and "No specific" not in kcc_results
    has_rag = bool(rag_results)

    # ── build prompt based on data availability ──────────────────────────────
    if has_kcc:
        # HIGH confidence — real KCC farmer advisory data
        confidence_label = "High"
        data_section     = f"Farmer Advisory Data (KCC):\n{kcc_results}"

    elif has_rag:
        # MEDIUM confidence — FAISS knowledge base
        confidence_label = "Medium"
        data_section     = "Knowledge Base:\n" + "\n".join(rag_results[:3])

    else:
        # LOW confidence — LLM general knowledge only
        confidence_label = "Low"
        data_section     = (
            f"Crop: {crop}\n"
            f"State: {state or 'unknown'}\n"
            "No specific database match found. Use general agricultural knowledge."
        )

    prompt = f"""You are an agriculture advisor helping an Indian farmer.

Problem reported:
{text}

{data_section}

STRICT RULES:
- Use ONLY the data provided above
- Write in simple, practical farmer language
- Do NOT copy text — summarise clearly
- If data is insufficient, say so honestly

Reply in this exact format:

Cause:
[what is causing the problem]

Solution:
- Step 1
- Step 2
- Step 3

Warning:
[any safety or timing warning]

Confidence: {confidence_label}
"""

    answer = ask_llm(prompt)

    if answer and answer.strip():
        return answer.strip()

    # ── safe static fallback ─────────────────────────────────────────────────
    return (
        "Cause:\nIssue detected but detailed data is unavailable.\n\n"
        "Solution:\n"
        "- Inspect your crop closely for visible symptoms\n"
        "- Maintain proper irrigation\n"
        "- Avoid applying chemicals without identifying the problem\n\n"
        "Warning:\nConsult your local agriculture officer if symptoms spread.\n\n"
        "Confidence: Low"
    )


# ============================================================
# 7. MAIN CHAT ROUTER  (bot_logic.py — chatbot_response)
# ============================================================

def chatbot_response(user_id: str, text: str = None, lat=None, lon=None) -> str:
    """
    Central entry point called by telegram_bot.py for every message.

    Flow:
      new user       → ask name
      ask_name mode  → save name, ask location
      GPS coords     → reverse-geocode, save state
      ask_location   → parse text location
      else           → detect intent → route to handler
    """
    context = db.get_user(user_id) or {}
    mode    = context.get("mode", "new")

    # ── onboarding: new user ─────────────────────────────────────────────────
    if mode == "new":
        db.upsert_user(user_id, mode="ask_name")
        return "🙏 Namaste! What is your name?"

    # ── onboarding: waiting for name ─────────────────────────────────────────
    if mode == "ask_name":
        if not text or not text.strip():
            return "Please tell me your name so I can help you better."
        name = text.strip()
        db.upsert_user(user_id, name=name, mode="ask_location")
        return f"Nice to meet you, {name}! 📍 Please share your state or location."

    # ── GPS location received ─────────────────────────────────────────────────
    if lat is not None and lon is not None:
        resolved_lat, resolved_lon, state = handle_location(lat, lon)
        if state:
            db.upsert_user(
                user_id,
                state=state,
                lat=resolved_lat,
                lon=resolved_lon,
                mode="problem",
            )
            return (
                f"📍 Location set: {state}\n\n"
                "Now tell me your farming problem or ask for crop suggestions."
            )
        return "⚠️ Could not detect your state from GPS. Please type your state name."

    # ── text location expected ────────────────────────────────────────────────
    if mode == "ask_location":
        return handle_location_intent(text or "", user_id)

    # ── intent routing ────────────────────────────────────────────────────────
    if text:
        intent = detect_intent(text)
        logger.info(f"[router] user={user_id} intent={intent} text='{text[:40]}'")

        if intent == "greeting":
            return handle_greeting()

        elif intent == "location":
            return handle_location_intent(text, user_id)

        elif intent == "help":
            db.upsert_user(user_id, mode="problem")
            return (
                "🙏 I can help you with:\n\n"
                "👉 Pest or disease problems\n"
                "👉 Crop suggestions\n"
                "👉 Water / fertilizer advice\n\n"
                "Please describe your problem with your crop name.\n"
                "Example: 'wheat leaves turning yellow in Punjab'"
            )

        elif intent == "crop_recommendation":
            return handle_crop_recommendation(
                context.get("lat"),
                context.get("lon"),
                context.get("state"),
            )

        elif intent == "problem":
            # Clear stale last_crop if user mentions no crop in this message
            if not detect_crop_from_text(text):
                db.upsert_user(user_id, last_crop=None)
            return handle_problem(user_id, text, context)

        else:
            # "general" or unknown
            return (
                "🌱 I can help with crops, pests, diseases, and farming tips.\n\n"
                "Try asking:\n"
                "• 'wheat pest problem in Gujarat'\n"
                "• 'suggest best crop for Punjab'\n"
                "• 'cotton yellow leaves'"
            )

    return "How can I help you? Type your farming question."