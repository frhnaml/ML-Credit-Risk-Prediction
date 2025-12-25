# src/services/mlp_service.py
import os
import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model



# =============================================================
# Path model & preprocessor
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "mlp_credit_risk_model")

MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.h5")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_mlp.pkl")

# =============================================================
# Load model & preprocessor dengan cache
# =============================================================
@st.cache_resource
def load_mlp_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MLP model not found: {MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor

# =============================================================
# Fungsi prediksi risiko kredit
# =============================================================
def predict_mlp_risk(input_df):
    """
    input_df : pandas.DataFrame (1 row) berisi fitur nasabah
    return : tuple(pred_label, probas)
        pred_label : list[int] → 0 (Lancar Bayar) / 1 (Gagal Bayar)
        probas : list[float] → probabilitas gagal bayar
    """
    model, preprocessor = load_mlp_model()

    # Preprocessing
    X = preprocessor.transform(input_df)
    X = X.astype(np.float32)

    # Prediksi
    probs = model.predict(X)[0]         # [prob_rendah, prob_tinggi] jika output softmax
    if probs.shape[0] == 2:             # softmax 2 output
        pred = int(np.argmax(probs))
        proba_gagal = float(probs[1])   # probabilitas kelas 1 → gagal bayar
    else:                               # single sigmoid output
        pred = int(probs[0] >= 0.5)
        proba_gagal = float(probs[0])

    # Return label dan probabilitas gagal bayar
    return [pred], [proba_gagal]

def predict_mlp_batch(df):
    model, preprocessor = load_mlp_model()

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
