from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

# ---------------------------
# Path setup
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"

# ---------------------------
# Load models
# ---------------------------
def load_model():
    logistic_model = joblib.load(MODEL_DIR / "logistic_model.pkl")

    s2v_model = SentenceTransformer(
        "Pachinee/sentence2vec-brd"
    )

    return logistic_model, s2v_model


# ---------------------------
# Predict
# ---------------------------
def predict_label(texts, logistic_model, s2v_model):
    embeddings = s2v_model.encode(
        texts,
        convert_to_numpy=True
    )
    preds = logistic_model.predict(embeddings)
    return ["Clear" if p == 1 else "Unclear" for p in preds]
