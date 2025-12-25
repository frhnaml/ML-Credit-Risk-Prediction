# src/Prediction.py
import streamlit as st
from utils.load_css import load_css

load_css()

st.set_page_config(
    page_title="Sistem Prediksi Risiko Kredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# Sidebar (INFORMASI SAJA, TANPA NAVIGASI MODEL)
# =============================================================
with st.sidebar:
    st.title("💳 Kredit Risk Dashboard")
    st.caption("Sistem Prediksi Risiko Kredit Nasabah")
    st.markdown("---")
    st.caption("© 2025 – ML Portfolio Dashboard")

# =============================================================
# MAIN CONTENT
# =============================================================
st.markdown("# 🚀 Sistem Prediksi Risiko Kredit Nasabah")

st.markdown(
    """
    Dashboard ini merupakan **sistem prediksi risiko kredit** berbasis *machine learning*
    yang dikembangkan untuk mengevaluasi kemungkinan **gagal bayar** nasabah.

    Sistem ini mengimplementasikan **tiga pendekatan model tabular** dengan tingkat
    kompleksitas dan arsitektur yang berbeda, mulai dari model dasar hingga
    *pretrained transformer*.
    """
)

st.markdown("---")

# =============================================================
# MODEL INFORMATION (CARD STYLE - HORIZONTAL)
# =============================================================
st.subheader("📚 Model yang Digunakan")

col1, col2, col3 = st.columns(3)

# =============================================================
# MLP CARD
# =============================================================
with col1:
    st.markdown("### 🧩 MLP (Base NN)")
    st.markdown(
        """
        Model neural network dasar sebagai **baseline**.
        
        - From scratch (tanpa pretrained)
        - Fokus fundamental NN
        - Cepat & interpretable
        """
    )

    with st.expander("🔍 Detail Model"):
        st.markdown(
            """
            **Arsitektur & Teknik:**
            - Input: fitur numerik + kategorikal (encoded)
            - Dense layers (Fully Connected)
            - Aktivasi: ReLU
            - Output: Sigmoid
            - Loss: Binary Cross Entropy
            - Optimizer: Adam

            **Peran dalam sistem:**
            - Baseline pembanding
            - Validasi manfaat transfer learning
            """
        )

# =============================================================
# TABNET CARD
# =============================================================
with col2:
    st.markdown("### 📊 TabNet (Pretrained)")
    st.markdown(
        """
        Model tabular berbasis **attention & feature selection**.
        
        - Transfer learning
        - Explainable ML
        - Minim feature engineering
        """
    )

    with st.expander("🔍 Detail Model"):
        st.markdown(
            """
            **Karakteristik Utama:**
            - Attention-based feature selection
            - Pretrained weights
            - Interpretabilitas tinggi

            **Teknologi:**
            - Library: `pytorch-tabnet`
            - Optimasi berbasis gradient descent

            **Kelebihan:**
            - Performa kuat pada data tabular kompleks
            """
        )

# =============================================================
# FT-TRANSFORMER CARD
# =============================================================
with col3:
    st.markdown("### ⚡ FT-Transformer")
    st.markdown(
        """
        Model tabular **state-of-the-art** berbasis Transformer.
        
        - Embedding + attention
        - Cocok numerik & kategorikal
        - Production-ready
        """
    )

    with st.expander("🔍 Detail Model"):
        st.markdown(
            """
            **Arsitektur:**
            - Numerical → projection layer
            - Categorical → embedding
            - Transformer blocks (multi-head attention)
            - Output: binary classification

            **Framework:**
            - `rtdl-revisiting-models`
            - PyTorch

            **Keunggulan:**
            - Akurasi tinggi
            - Stabil untuk batch & single inference
            """
        )


st.markdown("---")
st.subheader("🧠 Tech Stack yang Digunakan")

col_ts1, col_ts2, col_ts3 = st.columns(3)

with col_ts1:
    st.markdown("""
    <div class="model-card">
        <h4>🐍 Core & Data</h4>
        <ul>
            <li><b>Python</b> — bahasa utama</li>
            <li><b>pandas</b> — data manipulation</li>
            <li><b>NumPy</b> — numerical computing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_ts2:
    st.markdown("""
    <div class="model-card">
        <h4>🤖 Machine Learning</h4>
        <ul>
            <li><b>scikit-learn</b> — preprocessing & encoding</li>
            <li><b>PyTorch</b> — MLP & FT-Transformer</li>
            <li><b>pytorch-tabnet</b> — TabNet pretrained</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_ts3:
    st.markdown("""
    <div class="model-card">
        <h4>🧩 Deployment & Tools</h4>
        <ul>
            <li><b>Streamlit</b> — dashboard UI</li>
            <li><b>joblib</b> — model persistence</li>
            <li><b>rtdl-revisiting-models</b> — FT-Transformer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.caption(
    "💡 Gunakan menu halaman di sidebar untuk melakukan prediksi "
    "menggunakan masing-masing model atau menjalankan batch prediction."
)
