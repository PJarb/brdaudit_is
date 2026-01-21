import streamlit as st
import pandas as pd
import json

from utils.preprocess import preprocess_text
from utils.predict import load_model, predict_label

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="BRD Clarity Detection",
    layout="wide"
)

st.title("📄 BRD Clarity Detection Tool")
st.caption("Detect whether business requirements are Clear or Unclear")

# ---------------------------
# Load model (cache)
# ---------------------------
@st.cache_resource
def init_model():
    clf, s2v = load_model()
    return clf, s2v

clf, s2v_model = init_model()

# ---------------------------
# Input section
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload requirement file (JSON or TXT)",
    type=["json", "txt"]
)

text_input = st.text_area(
    "Or paste requirement text here",
    height=200
)

# ---------------------------
# Process
# ---------------------------
if st.button("Analyze"):
    if uploaded_file:
        if uploaded_file.name.endswith(".json"):
            data = json.load(uploaded_file)
            texts = [d["text"] for d in data]
        else:
            texts = uploaded_file.read().decode("utf-8").splitlines()

    elif text_input.strip():
        texts = text_input.splitlines()
    else:
        st.warning("Please upload a file or paste text")
        st.stop()

    df = pd.DataFrame({"text": texts})
    df["clean_text"] = df["text"].apply(preprocess_text)

    preds = predict_label(df["clean_text"], clf, s2v_model)
    df["Prediction"] = df["clean_text"].apply(lambda x: preds.pop(0))

    # ---------------------------
    # Output
    # ---------------------------
    st.subheader("Result")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Result (CSV)",
        df.to_csv(index=False),
        file_name="brd_clarity_result.csv"
    )
