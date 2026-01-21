from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"


def load_model():
    # โหลด logistic model จาก GitHub
    model_path = MODEL_DIR / "logistic_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"logistic_model.pkl not found at {model_path}")

    logistic_model = joblib.load(model_path)

    # โหลด sentence2vec จาก Hugging Face
    s2v_model = SentenceTransformer(
        "Pachinee/sentence2vec-brd"   # 👈 แก้เป็น username ของคุณ
    )

    return logistic_model, s2v_model


def predict_label(texts, logistic_model, s2v_model):
    """
    texts: list[str]
    return: list[str] -> ['Clear', 'Unclear']
    """
    embeddings = s2v_model.encode(
        texts,
        convert_to_numpy=True
    )

    preds = logistic_model.predict(embeddings)

    return ["Clear" if p == 1 else "Unclear" for p in preds]
