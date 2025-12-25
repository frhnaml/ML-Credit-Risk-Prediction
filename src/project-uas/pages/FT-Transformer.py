# pages/FT-Transformer.py

import streamlit as st
import matplotlib.pyplot as plt
from services.ft_transformer_service import FTTransformerService



# =============================================================
# Konfigurasi Halaman
# =============================================================
st.set_page_config(
    page_title="FT-Transformer Prediksi Risiko Kredit",
    page_icon="🤖",
    layout="wide"
)

# =============================================================
# Label Mapping (UI Friendly)
# =============================================================
FEATURE_LABELS = {
    "person_age": "Usia Nasabah",
    "person_income": "Pendapatan Nasabah (USD)",
    "person_home_ownership": "Kepemilikan Rumah",
    "person_emp_length": "Lama Bekerja (tahun)",
    "loan_intent": "Tujuan Pinjaman",
    "loan_grade": "Grade Pinjaman",
    "loan_amnt": "Jumlah Pinjaman",
    "loan_int_rate": "Tingkat Bunga (%)",
    "loan_percent_income": "Pinjaman / Pendapatan (%)",
    "cb_person_default_on_file": "Pernah Default Sebelumnya?",
    "cb_person_cred_hist_length": "Lama Riwayat Kredit (tahun)"
}

# =============================================================
# Load Service (cached)
# =============================================================
@st.cache_resource
def load_service():
    return FTTransformerService(model_name="ft_transformer_model2")

service = load_service()

features = service.get_features()
cat_options = service.get_categorical_options()

# =============================================================
# UI Header
# =============================================================
st.title("🤖 Prediksi Risiko Kredit – FT-Transformer")
st.markdown(
    """
    Halaman ini menggunakan model **FT-Transformer** 
    untuk memprediksi risiko gagal bayar kredit berdasarkan data nasabah.
    """
)


# =============================================================
# Input Form (STYLE SAMA DENGAN MLP)
# =============================================================
with st.form("ft_form", clear_on_submit=False):
    st.subheader("📝 Masukkan Data Nasabah")

    col1, col2, col3 = st.columns(3)
    input_data = {}

    with col1:
        input_data["person_age"] = st.number_input(
            FEATURE_LABELS["person_age"], min_value=18, max_value=100, value=30
        )
        input_data["person_income"] = st.number_input(
            FEATURE_LABELS["person_income"], min_value=0, value=50000
        )
        input_data["person_home_ownership"] = st.selectbox(
            FEATURE_LABELS["person_home_ownership"],
            options=cat_options["person_home_ownership"]
        )
        input_data["person_emp_length"] = st.number_input(
            FEATURE_LABELS["person_emp_length"], min_value=0, max_value=50, value=5
        )

    with col2:
        input_data["loan_intent"] = st.selectbox(
            FEATURE_LABELS["loan_intent"],
            options=cat_options["loan_intent"]
        )
        input_data["loan_grade"] = st.selectbox(
            FEATURE_LABELS["loan_grade"],
            options=cat_options["loan_grade"]
        )
        input_data["loan_amnt"] = st.number_input(
            FEATURE_LABELS["loan_amnt"], min_value=0, value=10000
        )
        input_data["loan_int_rate"] = st.number_input(
            FEATURE_LABELS["loan_int_rate"], min_value=0.0, max_value=100.0, value=10.0
        )

    with col3:
        input_data["loan_percent_income"] = st.number_input(
            FEATURE_LABELS["loan_percent_income"], min_value=0.0, max_value=10.0, value=0.5
        )
        input_data["cb_person_default_on_file"] = st.selectbox(
            FEATURE_LABELS["cb_person_default_on_file"],
            options=cat_options["cb_person_default_on_file"]
        )
        input_data["cb_person_cred_hist_length"] = st.number_input(
            FEATURE_LABELS["cb_person_cred_hist_length"], min_value=0, value=3
        )

    submitted = st.form_submit_button("🔮 Prediksi Risiko Kredit")

# =============================================================
# Prediction Result
# =============================================================
if submitted:
    try:
        pred_class, probs = service.predict(input_data)
        proba_lancar = float(probs[0])
        proba_gagal = float(probs[1])

        st.subheader("🔎 Hasil Prediksi")

        if pred_class == 1:
            st.error(
                f"⚠️ **Risiko Tinggi (Gagal Bayar)**\n\n"
                f"Probabilitas Gagal Bayar: **{proba_gagal:.2%}**"
            )
        else:
            st.success(
                f"✅ **Risiko Rendah (Lancar Bayar)**\n\n"
                f"Probabilitas Lancar Bayar: **{proba_lancar:.2%}**"
            )


        # =================================================
        # Visualisasi Probabilitas (compact & clean)
        # =================================================
        st.subheader("📊 Probabilitas Kredit")

        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.bar(
            ["Lancar Bayar", "Gagal Bayar"],
            [proba_lancar, proba_gagal],
            width=0.5
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilitas")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2%}",
                ha="center",
                fontweight="bold"
            )

        st.pyplot(fig)

        # =================================================
        # Confidence Bar
        # =================================================
        st.subheader("📈 Confidence Level")
        confidence = proba_gagal if pred_class == 1 else proba_lancar
        st.progress(int(confidence * 100))
        st.caption(f"Confidence: **{confidence:.2%}**")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat prediksi")
        st.exception(e)
