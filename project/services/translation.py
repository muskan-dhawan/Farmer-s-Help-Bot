"""
Multilingual mode (NO translation)

We do NOT detect or translate language.
We directly process user input using multilingual embeddings + LLM.
"""

def to_en(text, forced_lang=None):
    return text, "auto"

def from_en(text, lang):
    return text