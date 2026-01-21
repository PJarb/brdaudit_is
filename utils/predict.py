from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"

def load_model():
    logistic_model = joblib.load(MODEL_DIR / "logistic_model.pkl")

    s2v_model = SentenceTransformer(
        "Pachinee/sentence2vec-brd"
    )

    return logistic_model, s2v_model
