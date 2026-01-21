import joblib
from sentence_transformers import SentenceTransformer

def load_model():
    clf = joblib.load("model/clf.pkl")
    s2v_model = SentenceTransformer("model/s2v_model")
    return clf, s2v_model

def predict_label(texts, clf, s2v_model):
    embeddings = s2v_model.encode(
        list(texts),
        convert_to_numpy=True
    )
    preds = clf.predict(embeddings)
    return ["Clear" if p == 1 else "Unclear" for p in preds]
