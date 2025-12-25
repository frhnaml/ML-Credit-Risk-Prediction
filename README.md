# Sistem Prediksi Risiko Kredit Nasabah Menggunakan Neural Network dan Transfer Learning pada Data Tabular

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem prediksi risiko kredit (Credit Risk Prediction) berbasis Machine Learning untuk data tabular. Sistem ini bertujuan untuk memprediksi kemungkinan default pinjaman berdasarkan karakteristik demografis dan finansial peminjam.

Dalam proyek ini, diimplementasikan tiga model neural network sesuai dengan ketentuan modul, yaitu:

- Neural Network Base (Non-Pretrained)
  Model Multilayer Perceptron (MLP) yang dibangun dan dilatih dari awal tanpa menggunakan bobot pretrained. Model ini digunakan sebagai baseline untuk data tabular.

- Model Pretrained 1 – TabNet
  Model deep learning khusus data tabular yang menggunakan mekanisme attention dan feature selection, diadaptasi sebagai pendekatan transfer learning untuk meningkatkan performa prediksi.

- Model Pretrained 2 – FT-Transformer
  Model berbasis Transformer untuk data tabular yang memanfaatkan self-attention guna menangkap hubungan kompleks antar fitur, digunakan sebagai model pretrained kedua dalam skema transfer learning.

Proyek ini mencakup seluruh tahapan pipeline Machine Learning, mulai dari preprocessing data, pelatihan model, hingga evaluasi performa. Evaluasi dilakukan secara komprehensif menggunakan classification report (accuracy, precision, recall, dan F1-score), confusion matrix, serta grafik training dan validation loss serta accuracy untuk menganalisis performa dan potensi overfitting pada setiap model.

Sebagai tahap akhir, seluruh model yang telah dilatih diintegrasikan ke dalam aplikasi web berbasis Streamlit, sehingga pengguna dapat melakukan prediksi risiko kredit secara interaktif melalui antarmuka website yang dijalankan secara lokal.

## Tujuan proyek:

- Membangun pipeline *training → evaluasi* secara end-to-end
- Membandingkan performa beberapa model *Machine Learning*
- Menyajikan hasil prediksi dalam bentuk sistem website yang dapat dijalankan secara lokal

## Struktur Repository

UAP/
│
├── .venv/
├── .pdm-python
├── pdm.lock
├── pyproject.toml
├── .gitignore
├── credit_risk_dataset.csv
├── get-pip.py
├── src/
│   ├── frhn_UAP_TrainModel.ipynb   (NOTEBOOK TRAINING)
│   └── project-uas/
│       │
│       ├── Prediction.py         
│       ├── assets/
│       │   └── style.css/
│       │
│       ├── models/
│       │   ├── mlp_model/
│       │   ├── tabnet_model/
│       │   └── ft_transformer_model/
│       │
│       ├── pages/
│       │   ├── Batch_Prediction.py
│       │   ├── FT_Transformer.py
│       │   ├── MLP.py
│       │   └── TabNet.py
│       │
│       ├── services/
│       │   ├── mlp_service.py
│       │   ├── tabnet_service.py
│       │   └── ft_transformer_service.py
│       │
│       └── utils/
│           ├── load_css.py

## Dataset

Dataset berhasil dimuat dengan detail sebagai berikut:

- Jumlah data : 32.581 baris
- Jumlah fitur : 12 kolom
- Ukuran memori: 1763 KB

Daftar Fitur : 
Fitur Numerik (7)
- person_age
- person_income
- person_emp_length
- loan_amnt
- loan_int_rate
- loan_percent_income
- cb_person_cred_hist_length

Fitur Kategorikal (4)
- person_home_ownership
- loan_intent
- loan_grade
- cb_person_default_on_file

Target : 
- loan_status

Definisi Target: 
0 → 
1 →
