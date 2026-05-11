import os
import urllib.parse
import logging
import requests
from services.weather import safe_request

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

def fetch_kcc_data(crop, state, problem_text=None):
    kcc_key = os.getenv("KCC_API_KEY")
    crop_label = crop or "local crops"

    if not kcc_key or not state:
        return (
            "- No exact KCC match found because crop/location data is incomplete.\n"
            "- General safe advice: share your crop, state, and problem details before applying any pesticide or fertilizer."
        )

    try:
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
        results = []

        for r in records:
            query = r.get("QueryText", "")
            answer = r.get("KccAns", "")
            combined_text = (query + " " + answer).lower()

            crop_match = crop and crop.lower() in combined_text

            problem_match = False
            if problem_keywords:
                problem_match = any(k in combined_text for k in problem_keywords)

            if (crop_match or not crop) and (problem_match or not problem_keywords):
                if query.strip() and answer.strip():
                    formatted = f"- Q: {query}\n  A: {answer}"
                    results.append(formatted)

        return "\n".join(results[:3]) if results else f"- No specific KCC advice found for {crop_label} in {state}."

    except Exception as e:
        logging.error(f"KCC API error: {e}")
        return f"- KCC API is currently unavailable."
