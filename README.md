


<h1 align="center">📊 Sistem Prediksi Risiko Kredit Nasabah Menggunakan Neural Network dan Transfer Learning pada Data Tabular</h1>

---

<h1 align="center">📑 Table of Contents</h1>

1. [Project Overview](#-project-overview)
   - [Deskripsi Proyek](#-deskripsi-proyek)
   - [Tujuan Proyek](#-tujuan-proyek)
2. [Struktur Repository](#-struktur-repository)
3. [Dataset](#-dataset)
   - [Daftar Fitur](#-daftar-fitur)
   - [Definisi Target](#-definisi-target)
4. [Preprocessing Data](#-preprocessing-data)
5. [Model yang Digunakan](#-model-yang-digunakan)
6. [Hasil Evaluasi Model](#-hasil-evaluasi-model-test-set)
7. [Analisis Perbandingan Model](#-analisis-perbandingan-model)
8. [Panduan Menjalankan Sistem Website](#-panduan-menjalankan-sistem-website-secara-lokal)
9. [Dashboard Streamlit Prediksi](#-dashboard-streamlit-prediksi)
10. [Link Live Demo](#-link-live-demo)
11. [Resources](#-resources)

---

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

---

## 🎯 Tujuan proyek:

- Membangun pipeline *training → evaluasi* secara end-to-end
- Membandingkan performa beberapa model *Machine Learning*
- Menyajikan hasil prediksi dalam bentuk sistem website yang dapat dijalankan secara lokal

---

## 🗂️ Struktur Repository
```
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
```
---

## 📊 Dataset

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
0 → Resiko Rendah (Lancar Bayar)
1 → Resiko Tinggi (Gagal Bayar)

---

## 🧪 Preprocessing Data

### Pemisahan Data

Data dibagi menggunakan stratified split untuk menjaga proporsi kelas target:
Split	      Jumlah Data
Train	        22.806
Validation	   4.887
Test	         4.888

### Preprocessing MLP
- Numerical: Median Imputation + StandardScaler
- Categorical: Most Frequent Imputation + Ordinal Encoding
Digunakan pipeline ColumnTransformer untuk konsistensi training & inference.

### Preprocessing Tabnet
- Numerical: Median Imputation
- Categorical: Ordinal Encoding
Tanpa scaling numerik karena TabNet mampu menangani skala fitur secara internal.

### Preprocessing FT-Transformer
- Numerical: Median Imputation + StandardScaler
- Categorical: Label Encoding
Encoding disimpan untuk keperluan inference

---

## 🧠 Model yang Digunakan
Proyek ini menggunakan tiga model Machine Learning yang diimplementasikan dan dilatih secara terpisah, dengan tujuan membandingkan performa model baseline dan model deep learning lanjutan pada data tabular.

### 🔹 Multi-Layer Perceptron (MLP)
Model Multi-Layer Perceptron (MLP) digunakan sebagai baseline deep learning untuk data tabular.
Model ini diimplementasikan menggunakan TensorFlow/Keras dengan arsitektur fully-connected neural network.

#### Arsitektur Model
Model terdiri dari beberapa hidden layer dengan aktivasi ReLU dan regularisasi sebagai berikut:
- Input layer sesuai jumlah fitur hasil preprocessing
- Dense (128) + ReLU
- Batch Normalization
- Dropout (0.3)
- Dense (64) + ReLU
- Batch Normalization
- Dropout (0.3)
- Dense (32) + ReLU
- Dropout (0.2)
- Output layer Dense (1) dengan aktivasi Sigmoid

Arsitektur ini dirancang untuk:
- Menangani non-linearitas data
- Mengurangi risiko overfitting melalui Dropout dan Batch Normalization
- Konfigurasi Training
- Loss function: Binary Crossentropy
- Optimizer: Adam (learning rate = 1e-3)
- Metrics: Accuracy dan AUC
- Batch size: 256
- Epoch maksimum: 50

Untuk mencegah overfitting, digunakan:
- EarlyStopping dengan monitoring val_loss (patience = 5)
- ModelCheckpoint untuk menyimpan model terbaik berdasarkan val_auc

### 🟢 TabNet
Model TabNet digunakan sebagai model deep learning yang secara khusus dirancang untuk data tabular.
TabNet memanfaatkan mekanisme sequential attention untuk melakukan feature selection secara eksplisit pada setiap decision step.

#### Konfigurasi Model
Model diimplementasikan menggunakan library pytorch-tabnet dengan konfigurasi utama:
- Decision dimension (n_d) = 16
- Attention dimension (n_a) = 16
- Number of steps (n_steps) = 5
- Sparse regularization (lambda_sparse) = 1e-4
- Mask type: entmax (sparse attention)

Optimizer dan scheduler:
- Optimizer: Adam (learning rate = 2e-2)
- Learning rate scheduler: StepLR (step size = 10, gamma = 0.9)

#### Training Strategy
- Data dikonversi ke tipe float32
- Training dilakukan hingga maksimum 100 epoch
- Early stopping diterapkan dengan patience = 15
- Evaluasi dilakukan pada data train dan validation set menggunakan accuracy

TabNet dipilih karena kemampuannya:
- Menangani fitur numerik dan kategorikal secara efisien
- Memberikan performa yang baik tanpa scaling numerik eksplisit
- Memiliki interpretabilitas melalui attention mask

### 🟣 FT-Transformer
Model FT-Transformer merupakan model berbasis Transformer architecture yang dirancang khusus untuk data tabular. Model ini digunakan sebagai model paling kompleks dalam proyek ini.

#### Representasi Data
- Fitur numerik diproses sebagai continuous features
- Fitur kategorikal direpresentasikan menggunakan embedding berdasarkan cardinality masing-masing fitur
- Seluruh data dikonversi ke PyTorch tensor
- Training dilakukan pada CPU atau GPU (jika tersedia)

#### Arsitektur Model
Model diimplementasikan menggunakan library rtdl-revisiting-models dengan konfigurasi:
- Jumlah fitur numerik sesuai dataset
- Embedding untuk fitur kategorikal berdasarkan cardinality
- Dimensi blok (d_block) = 32
- Jumlah blok Transformer (n_blocks) = 4
- Jumlah attention heads = 8
- Dropout pada attention, feed-forward, dan residual connection
- Output layer disesuaikan dengan jumlah kelas target (loan_status).

#### Konfigurasi Training
- Loss function: CrossEntropyLoss
- Optimizer: AdamW (learning rate = 1e-3, weight decay = 1e-5)
- Early stopping manual dengan patience = 7 berdasarkan validation loss
- Model terbaik disimpan berdasarkan validation loss terendah

FT-Transformer dipilih karena kemampuannya:
- Menangkap interaksi kompleks antar fitur
- Memberikan performa tinggi pada data tabular kompleks
- Menjadi pembanding utama terhadap MLP dan TabNet

---

## 📈 Hasil Evaluasi Model (test set)
### MLP (Multi-Layer Perceptron)
Accuracy: 0.91
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
|   0   |   0.91    |  0.98  |   0.95   |
|   1   |   0.91    |  0.67  |   0.77   |


Training Curves:
| Accuracy Curves| Loss Curve | 
|----------------|------------|
|![Contoh Prediksi6](src/project-uas/assets/train&val-acc-mlp.png) | ![MLP Loss Curve](src/project-uas/assets/train&val-loss-mlp.png)|


#### Analisis Evaluasi Model MLP
Model MLP menunjukkan performa yang cukup solid dengan accuracy sebesar 0.91. Model ini sangat baik dalam mengklasifikasikan kelas No Default (kelas 0), tercermin dari nilai recall yang tinggi (0.98), artinya hampir seluruh peminjam yang tidak gagal bayar berhasil terdeteksi dengan benar. Namun, performa pada kelas Default(kelas 1) masih terbatas, khususnya pada recall (0.67), yang mengindikasikan bahwa sebagian kasus gagal bayar masih salah diklasifikasikan sebagai No Default (lancar bayar). 

### TabNet
Accuracy: 0.92
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
|   0   |   0.92    |  0.99  |   0.95   |
|   1   |   0.95    |  0.68  |   0.79   |

Training Curves:
| Accuracy Curves| Loss Curve | 
|----------------|------------|
|![Contoh Prediksi6](src/project-uas/assets/train&val-acc-tabnet.png) | ![Contoh Prediksi6](src/project-uas/assets/train&val-loss-tabnet.png)|

<!-- ![TabNet Accuracy Curve](src/project-uas/assets/train&val-acc-tabnet.png)
![TabNet Loss Curve](src/project-uas/assets/train&val-loss-tabnet.png) -->

#### Analisis Evaluasi Model TabNet
TabNet memberikan peningkatan performa dibandingkan MLP dengan accuracy 0.92. Model ini mempertahankan performa yang sangat baik pada kelas No Default (recall 0.99) sekaligus meningkatkan precision pada kelas Default hingga 0.95. Artinya, ketika TabNet memprediksi Default, prediksi tersebut lebih dapat dipercaya. Namun, recall kelas Default masih berada di angka 0.68, yang menunjukkan bahwa meskipun prediksi lebih presisi, sebagian kasus gagal bayar masih belum sepenuhnya terdeteksi.

### FT-Transformer

Accuracy: 0.93
ROC-AUC: 0.92
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
|   0   |   0.92    |  0.99  |   0.95   |
|   1   |   0.96    |  0.71  |   0.82   |

Training Curves:
| Accuracy Curves| Loss Curve | 
|----------------|------------|
|![FT-Transformer Accuracy Curve](src/project-uas/assets/train&val-acc-fttransformer.png) |![FT-Transformer Loss Curve](src/project-uas/assets/train&val-loss-fttransformer.png)|

#### Analisis Evaluasi Model FT-Transformer
FT-Transformer merupakan model dengan performa terbaik di antara ketiga model yang diuji. Model ini mencapai accuracy tertinggi (0.93) serta ROC-AUC sebesar 0.92, yang menandakan kemampuan diskriminatif yang sangat baik antara kelas Default dan No Default. Precision untuk kelas Default mencapai 0.96, sementara recall meningkat menjadi 0.71, menghasilkan F1-score tertinggi (0.82) untuk kelas default.

---

## 📊 Analisis Perbandingan Model

| Model | Accuracy | Analisis |
|------|----------|----------|
| **MLP (Multi-Layer Perceptron)** | **0.91** | MLP menunjukkan performa yang stabil dan efisien sebagai baseline deep learning pada data tabular. Model ini memiliki keseimbangan precision dan recall yang baik untuk kelas mayoritas (Class 0), dengan F1-score tinggi (0.95). Namun, recall pada kelas minoritas (Class 1) masih relatif lebih rendah (0.67), yang mengindikasikan keterbatasan dalam mendeteksi kasus berisiko tinggi. Secara keseluruhan, MLP unggul dari sisi kesederhanaan, kecepatan training, dan efisiensi komputasi. |
| **TabNet** | **0.92** | TabNet memanfaatkan mekanisme attention untuk melakukan feature selection secara dinamis selama proses training. Model ini menunjukkan peningkatan performa dibandingkan MLP, terutama pada precision kelas minoritas (0.95) dan F1-score (0.79). Recall Class 1 masih terbatas (0.68), tetapi TabNet menawarkan keunggulan tambahan berupa interpretabilitas fitur, yang penting untuk analisis risiko kredit dan pengambilan keputusan berbasis model. |
| **FT-Transformer** | **0.93** | FT-Transformer memberikan performa terbaik di antara ketiga model dengan accuracy tertinggi (0.93) dan ROC-AUC sebesar 0.92. Model ini mampu menangkap interaksi kompleks antar fitur tabular, tercermin dari precision tinggi pada kelas minoritas (0.96) dan peningkatan recall Class 1 (0.71) dibandingkan model lain. Meskipun memerlukan biaya komputasi dan kompleksitas training yang lebih tinggi, FT-Transformer menawarkan generalisasi yang lebih baik dan performa paling seimbang untuk kasus klasifikasi risiko kredit. |

---

## 🚀 Panduan Menjalankan Sistem Website Secara Lokal
Aplikasi web ini menggunakan Streamlit sebagai antarmuka interaktif untuk melakukan prediksi risiko kredit berdasarkan data input pengguna. Sistem akan memproses data dan menampilkan hasil prediksi menggunakan model Machine Learning terbaik yang telah dilatih.

### Prasyarat
Pastikan lingkungan Anda telah memenuhi persyaratan berikut:
- Python 3.8+ sudah terinstall
- VS Code dengan PowerShell terminal
- Akses ke model yang telah disimpan (models/), pastikan file model tersedia dan path 

### Langkah Instalasi dan Menjalankan Aplikasi
1. Install PDM (Package Manager)
- Buka terminal PowerShell di VS Code
- Jalankan perintah berikut:
```
Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing | python -
```
- Setelah instalasi, output akan menampilkan path seperti 
```
C:\Users\<username>\AppData\Roaming\Python\Scripts.
```
- Copy path tersebut.
- Buka Advanced System Settings > Environment Variables > User Variables > New, paste path, lalu restart terminal/VS Code.

2. Inisialisasi Proyek
- Di root folder UAP/, jalankan: pdm init.
- Lengkapi prompt terminal (nama proyek: "project-uas", dll.).

3. Install Dependencies untuk Streamlit App:
- Pindah ke folder: cd UAP/project-uas.
- Aktifkan virtual environment: 
```
.venv\Scripts\Activate.ps1 (untuk PowerShell; gunakan source .venv/bin activate untuk bash/Linux/Mac).
```
- Jalankan: 
```
pip install -r requirements.txt (ini akan install Streamlit, TensorFlow, dll.).
```

4. Jalankan Aplikasi
- Di folder UAP/streamlit-app, jalankan: 
```
streamlit run app.py
```
- Tunggu hingga browser terbuka otomatis di http://localhost:... (atau buka manual).

5. Cara Penggunaan
- Masukkan data peminjam melalui form yang tersedia
- Pilih model prediksi yang ingin digunakan (MLP, TabNet, atau FT-Transformer)
- Klik tombol Predict

Sistem akan menampilkan:
- Hasil prediksi (Default / No Default)
- Probabilitas risiko
- Interpretasi hasil prediksi

---

## 🖥️ Dasboard Prediksi
![Contoh Prediksi1](src/project-uas/assets/dasboard-prediction.png)
![Contoh Prediksi2](src/project-uas/assets/dasboard-prediction2.png)
![Contoh Prediksi3](src/project-uas/assets/dasboard-prediction3.png)
![Contoh Prediksi4](src/project-uas/assets/dasboard-prediction4.png)
![Contoh Prediksi5](src/project-uas/assets/dasboard-prediction5.png)

---

## 🔗 Link Live Demo
not yet

---

## 📚 Resources
- **Dataset**: [Google Drive – Credit Risk Dataset](https://drive.google.com/file/d/1fyfnK8xgmEpNbaVqJ8lu3EurKSrYfDZ1/view?usp=sharing)
- **Model**: [Google Drive – Trained Models](https://drive.google.com/drive/folders/1gsIc70cST92o2unsZykUbAfX7zN-ZT-F?usp=sharing)
- **Google Colab Hasil Training**: [Colab Notebook](https://colab.research.google.com/drive/1fQtxsq8T7yujCDGG2upZ1F2WedA7BErj?usp=sharing)

---

# 👤 Author 
Achmal Farhan Ashidiqy  
🎓 Informatics Engineering  
📍 Muhammadiyah University of Malang  

