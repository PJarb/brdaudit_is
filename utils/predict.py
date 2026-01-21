from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

def load_model():
    logistic_model = joblib.load(MODEL_DIR / "logistic_model.pkl")
    s2v_model = SentenceTransformer(str(MODEL_DIR / "sentence2vec_model"))
    return logistic_model, s2v_model


def predict_label(texts, logistic_model, s2v_model):
    embeddings = s2v_model.encode(
        list(texts),
        convert_to_numpy=True
    )
    preds = logistic_model.predict(embeddings)
    return ["Clear" if p == 1 else "Unclear" for p in preds]
