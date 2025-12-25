# src/pages/mlp_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from services.mlp_service import predict_mlp_risk  # backend MLP service



st.set_page_config(page_title="MLP Prediksi Risiko Kredit", page_icon="🧩", layout="wide")

st.title("🧩 Prediksi Risiko Kredit – MLP (Base NN)")
st.markdown(
    """
    Halaman ini menggunakan model **MLP (Feedforward Neural Network)** 
    untuk memprediksi risiko gagal bayar kredit berdasarkan data nasabah.
    """
)

# =============================================================
# Input Form di halaman utama
# =============================================================
with st.form(key="mlp_form", clear_on_submit=False):
    st.subheader("Masukkan Data Nasabah")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("Usia Nasabah", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Pendapatan Nasabah (USD)", min_value=0, value=50000)
        person_home_ownership = st.selectbox("Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE"])
        person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0, max_value=50, value=5)
    
    with col2:
        loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Grade Pinjaman", ["A","B","C","D","E","F","G"])
        loan_amnt = st.number_input("Jumlah Pinjaman", min_value=0, value=10000)
        loan_int_rate = st.number_input("Tingkat Bunga (%)", min_value=0.0, max_value=100.0, value=10.0)
    
    with col3:
        loan_percent_income = st.number_input("Pinjaman / Pendapatan (%)", min_value=0.0, max_value=10.0, value=0.5)
        cb_person_default_on_file = st.selectbox("Pernah Default Sebelumnya?", ["Y","N"])
        cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", min_value=0, value=3)

    submit_btn = st.form_submit_button("🔮 Prediksi Risiko Kredit")

# =============================================================
# Buat dataframe input
# =============================================================
input_data = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_default_on_file": [cb_person_default_on_file],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length]
})

# =============================================================
# Prediksi & Visualisasi
# =============================================================
if submit_btn:
    result, proba = predict_mlp_risk(input_data)
    proba_gagal = proba[0]
    proba_lancar = 1 - proba_gagal

    # =========================
    # Hasil prediksi dalam card
    # =========================
    st.subheader("🔎 Hasil Prediksi")
    
    if result[0] == 1:
        st.error(
            f"⚠️ Prediksi: **Risiko Tinggi (Gagal Bayar)**\n\n"
            f"Probabilitas Gagal Bayar: {proba_gagal:.2%}"
        )
    else:
        st.success(
            f"✅ Prediksi: **Risiko Rendah (Lancar Bayar)**\n\n"
            f"Probabilitas Gagal Bayar: {proba_gagal:.2%}"
        )

    # =========================
    # Visualisasi probabilitas compact
    # =========================
    st.subheader("📊 Probabilitas Kredit")
    fig, ax = plt.subplots(figsize=(5,2.5))  # ringkas tapi jelas
    bars = ax.bar(
        ["Lancar Bayar", "Gagal Bayar"],
        [proba_lancar, proba_gagal],
        color=["#2ecc71","#e74c3c"],
        width=0.5
    )
    ax.set_ylim(0,1)
    ax.set_ylabel("Probabilitas")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Lancar Bayar", "Gagal Bayar"], fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tambahkan label nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2%}", ha='center', fontweight='bold')

    st.pyplot(fig)
    
    # =========================
    # Progress bar tambahan
    # =========================
    st.subheader("📈 Confidence Level")
    st.progress(int(proba_lancar*100) if result[0]==0 else int(proba_gagal*100))

