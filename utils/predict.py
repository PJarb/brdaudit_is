from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
import os

def load_model():
    # --- DEBUG (ช่วยดู path จริงบน Streamlit) ---
    print("CWD:", os.getcwd())
    print("FILES:", os.listdir())

    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "model" / "logistic_model.pkl"

    print("MODEL PATH:", model_path)
    print("MODEL EXISTS:", model_path.exists())

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logistic_model = joblib.load(model_path)

    s2v_model = SentenceTransformer(
        "Pachinee/sentence2vec-brd"   # 👈 ของคุณ
    )

    return logistic_model, s2v_model
