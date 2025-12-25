import streamlit as st
import pandas as pd
import numpy as np

from utils.load_css import load_css
from services.mlp_service import predict_mlp_batch
from services.tabnet_service import predict_tabnet_batch
# from services.ft_transformer_service import predict_ft_batch
from services.ft_transformer_service import FTTransformerService

# ============================================================
# INIT
# ============================================================
load_css()

st.set_page_config(
    page_title="Batch Prediction – Kredit Risk",
    page_icon="📂",
    layout="wide"
)

if "results" not in st.session_state:
    st.session_state.results = {}

# ============================================================
# HEADER
# ============================================================
st.title("📂 Batch Prediction – Kredit Risk Dataset")
st.markdown(
    """
    Halaman ini memungkinkan **prediksi risiko kredit secara batch**
    menggunakan dataset `.csv`.

    Cocok untuk:
    - Analisis portofolio kredit
    - Simulasi risiko
    - Evaluasi performa model
    """
)

st.markdown("---")

# ============================================================
# UPLOAD DATASET
# ============================================================
uploaded_file = st.file_uploader(
    "Upload Dataset Kredit (.csv)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Total data: **{len(df)} baris** | **{df.shape[1]} kolom**")

    st.markdown("---")

    # ========================================================
    # MODEL SELECTION
    # ========================================================
    st.subheader("🧠 Pilih Model untuk Batch Prediction")

    model_choices = st.multiselect(
        "Pilih model:",
        [
            "MLP (Base Neural Network)",
            "TabNet (Pretrained)",
            "FT-Transformer (Pretrained)"
        ],
        default=["MLP (Base Neural Network)"]
    )

    run_btn = st.button("🚀 Jalankan Batch Prediction")

    # ========================================================
    # RUN PREDICTION
    # ========================================================
    ft_service = FTTransformerService()
    if run_btn and model_choices:
        with st.spinner("⏳ Memproses batch prediction..."):
            results = {}

            if "MLP (Base Neural Network)" in model_choices:
                results["MLP"] = predict_mlp_batch(df)

            if "TabNet (Pretrained)" in model_choices:
                results["TabNet"] = predict_tabnet_batch(df)

            if "FT-Transformer (Pretrained)" in model_choices:
                results["FT-Transformer"] = ft_service.predict_batch(df)


            st.session_state.results = results

        st.success("✅ Batch prediction selesai")

# ============================================================
# READ RESULTS FROM SESSION STATE (INI KUNCI!)
# ============================================================
results = st.session_state.results

if results:
    st.markdown("---")

    # ========================================================
    # RESULT DISPLAY
    # ========================================================
    st.subheader("📈 Hasil Prediksi (Preview)")

    for model_name, result_df in results.items():
        st.markdown(f"### {model_name}")
        st.dataframe(result_df.head(10), use_container_width=True)

    # ========================================================
    # DISTRIBUSI RISIKO
    # ========================================================
    st.markdown("---")
    st.subheader("📊 Distribusi Risiko per Model")

    cols = st.columns(len(results))

    for col, (model_name, result_df) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {model_name}")

            risk_counts = result_df["prediction_label"].value_counts()
            high = int(risk_counts.get("Gagal Bayar", 0))
            low = int(risk_counts.get("Lancar Bayar", 0))

            st.metric("⚠️ Risiko Tinggi", high)
            st.metric("✅ Risiko Rendah", low)
            st.bar_chart(risk_counts)

    # ========================================================
    # INTERPRETASI
    # ========================================================
    st.markdown("---")
    st.subheader("📝 Interpretasi Risiko Kredit")

    for model_name, result_df in results.items():
        risk_counts = result_df["prediction_label"].value_counts()
        high = int(risk_counts.get("Gagal Bayar", 0))
        low = int(risk_counts.get("Lancar Bayar", 0))
        total = high + low

        high_pct = (high / total) * 100 if total > 0 else 0

        st.markdown(
            f"""
            **{model_name}** memprediksi bahwa dari **{total} nasabah**:
            - ⚠️ **{high} ({high_pct:.2f}%)** berisiko **gagal bayar**
            - ✅ **{low} ({100-high_pct:.2f}%)** diprediksi **lancar bayar**

            👉 Profil risiko dataset ini cenderung
            **{"TINGGI" if high_pct > 50 else "RENDAH"}** menurut model.
            """
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================
    st.markdown("---")
    st.subheader("⬇️ Download Hasil")

    for model_name, result_df in results.items():
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download hasil {model_name}",
            csv,
            file_name=f"batch_prediction_{model_name.lower()}.csv",
            mime="text/csv"
        )

else:
    st.info("📌 Upload dataset dan jalankan prediksi untuk melihat hasil.")
