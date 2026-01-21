from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

def load_model():
    # โหลด logistic model จาก GitHub
    logistic_model = joblib.load(MODEL_DIR / "logistic_model.pkl")

    # โหลด sentence2vec จาก Hugging Face
    s2v_model = SentenceTransformer(
        "USERNAME/sentence2vec-brd"
    )

    return logistic_model, s2v_model
