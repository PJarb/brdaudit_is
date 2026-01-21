import re

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
