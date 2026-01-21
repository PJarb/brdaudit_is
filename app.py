import sys
from pathlib import Path

# --- FIX IMPORT PATH FOR STREAMLIT ---
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
# ------------------------------------

import streamlit as st
import pandas as pd
import json

from utils.preprocess import preprocess_text
from utils.predict import load_model, predict_label
