# src/services/tabnet_service.py
import os
import joblib
import numpy as np
import streamlit as st
from pytorch_tabnet.tab_model import TabNetClassifier
from utils.load_css import load_css
load_css()  # otomatis load utils/style.css


# =============================================================
# Paths model & preprocessor
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "tabnet_credit_risk_model")

TABNET_PATH = os.path.join(MODEL_DIR, "model.zip")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_tab.pkl")

# =============================================================
# Load model & preprocessor
# =============================================================
@st.cache_resource
def load_tabnet_model():
    if not os.path.exists(TABNET_PATH):
        raise FileNotFoundError(f"TabNet model not found: {TABNET_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    model = TabNetClassifier()
    model.load_model(TABNET_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor

# =============================================================
# Fungsi prediksi TabNet
# =============================================================
def predict_tabnet_risk(input_df):
    """
    input_df : pandas.DataFrame (1 row)
    return : tuple(pred_label, probas)
    """
    model, preprocessor = load_tabnet_model()

    # Preprocessing
    X = preprocessor.transform(input_df)
    X = X.astype(np.float32)

    # Prediksi probabilitas dan label
    probs = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])
    
    return [pred], [float(probs[1])]  # probabilitas gagal bayar


def predict_tabnet_batch(df):
    model, preprocessor = load_tabnet_model()

    # preprocessing
    X = preprocessor.transform(df)

    # Keras predict → probability
    probs = model.predict(X).reshape(-1)

    # binary decision
    preds = (probs >= 0.5).astype(int)

    df_out = df.copy()
    df_out["prediction"] = preds
    df_out["prob_gagal_bayar"] = probs
    df_out["prediction_label"] = np.where(
        preds == 1, "Gagal Bayar", "Lancar Bayar"
    )

    return df_out