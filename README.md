# BBCA Stock Direction Prediction

Proyek ini memprediksi arah harga saham **BBCA (Bank Central Asia)** di Bursa Efek Indonesia (IDX) menggunakan algoritma Machine Learning: **Decision Trees, Random Forests, Gradient Boosting, KNN, Naive Bayes, SVM, PCA, dan LASSO (Logistic Regression L1)**.

Data diambil dari **Yahoo Finance** (ticker: `BBCA.JK`) selama 10 tahun terakhir.

---

## 📋 Fitur Utama

- **Pengambilan Data Otomatis**: Menggunakan `yfinance` untuk mengambil data historis 10 tahun.
- **Feature Engineering**: Menghitung indikator teknikal (SMA, RSI, High-Low Diff, Open-Close Diff).
- **Model Machine Learning**:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (Gaussian)
  - Support Vector Machine (SVM - RBF Kernel)
  - PCA + Random Forest (Dimensionality Reduction)
  - LASSO (Logistic Regression dengan regularisasi L1)
- **Evaluasi Model**: Accuracy, Precision, Recall, F1-Score per model.
- **Output**: Ringkasan performa semua model di terminal.

---

## 🚀 Cara Menjalankan

### 1. Prasyarat
- Python 3.8+
- Virtual environment (disarankan)

### 2. Setup Environment & Instalasi
```bash
# Clone repo
git clone https://github.com/vanderstark/bbca-stock-prediction.git
cd bbca-stock-prediction

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependensi
pip install -r requirements.txt
```

Isi `requirements.txt`:
```text
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
```

### 3. Menjalankan Script
```bash
python stock_prediction.py
```

---

## 📊 Hasil Eksperimen (Data: BBCA.JK, 10 Tahun, Test 20%)

| Model | Akurasi | Precision (Naik) | Recall (Naik) | F1 (Naik) |
|-------|---------|------------------|---------------|-----------|
| **LASSO (L1 LogReg)** | **0.5820** | 0.51 | 0.30 | 0.38 |
| **SVM (RBF)** | 0.5738 | 0.49 | 0.37 | 0.43 |
| **Naive Bayes** | 0.5656 | 0.48 | 0.41 | 0.45 |
| **Gradient Boosting** | 0.5594 | 0.47 | 0.38 | 0.42 |
| **KNN (k=5)** | 0.5533 | 0.47 | 0.50 | 0.48 |
| **Random Forest** | 0.5512 | 0.46 | 0.41 | 0.44 |
| **PCA + Random Forest** | 0.5471 | 0.47 | 0.50 | 0.48 |
| **Decision Tree** | 0.5266 | 0.44 | 0.48 | 0.46 |

> **Catatan**: Akurasi ~52-58% adalah baseline normal untuk prediksi arah harga saham (random walk hypothesis). Model terbaik: **LASSO (L1 Logistic Regression)** dengan akurasi 58.2%.

### Classification Report Detail

#### Decision Tree
```
              precision    recall  f1-score   support
           0       0.60      0.56      0.58       282
           1       0.44      0.48      0.46       206
    accuracy                           0.53       488
```

#### Random Forest
```
              precision    recall  f1-score   support
           0       0.60      0.65      0.63       282
           1       0.46      0.41      0.44       206
    accuracy                           0.55       488
```

#### Gradient Boosting
```
              precision    recall  f1-score   support
           0       0.60      0.69      0.64       282
           1       0.47      0.38      0.42       206
    accuracy                           0.56       488
```

#### KNN (k=5)
```
              precision    recall  f1-score   support
           0       0.62      0.60      0.61       282
           1       0.47      0.50      0.48       206
    accuracy                           0.55       488
```

#### Naive Bayes
```
              precision    recall  f1-score   support
           0       0.61      0.68      0.64       282
           1       0.48      0.41      0.45       206
    accuracy                           0.57       488
```

#### SVM (RBF)
```
              precision    recall  f1-score   support
           0       0.61      0.72      0.66       282
           1       0.49      0.37      0.43       206
    accuracy                           0.57       488
```

#### LASSO (L1 Logistic Regression)
```
              precision    recall  f1-score   support
           0       0.61      0.79      0.69       282
           1       0.51      0.30      0.38       206
    accuracy                           0.58       488
```

#### PCA + Random Forest
```
              precision    recall  f1-score   support
           0       0.62      0.58      0.60       282
           1       0.47      0.50      0.48       206
    accuracy                           0.55       488
```

---

## ⚙️ Konfigurasi yang Dapat Diubah

Di bagian atas script `stock_prediction.py`:

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `TICKER` | `"BBCA.JK"` | Kode saham di Yahoo Finance |
| `START_DATE` | 10 tahun lalu | Rentang awal data |
| `END_DATE` | Hari ini | Rentang akhir data |
| `TEST_SIZE` | `0.2` | Proporsi data testing |
| `RANDOM_STATE` | `42` | Seed untuk reproducibility |

---

## 🧠 Penjelasan Model

| Model | Deskripsi |
|-------|-----------|
| **Decision Tree** | Pohon keputusan sederhana, interpretable. |
| **Random Forest** | Ensemble banyak Decision Tree, anti-overfitting. |
| **Gradient Boosting** | Sequential boosting, sering performa terbaik di data tabular. |
| **KNN** | Berbasis jarak, sederhana tapi sensitif terhadap skala. |
| **Naive Bayes** | Probabilistik, cepat, asumsi independensi fitur. |
| **SVM (RBF)** | Margin maksimal dengan kernel non-linear. |
| **LASSO (L1)** | Seleksi fitur otomatis via regularisasi L1. |
| **PCA + RF** | Reduksi dimensi lalu Random Forest. |

---

## 📝 Catatan Penting

1. **Ini adalah contoh edukasi**, bukan saran investasi.
2. Prediksi arah harga saham sangat sulit. Akurasi ~50-58% normal untuk baseline.
3. Untuk produksi, perlu:
   - Feature engineering lanjutan (MACD, Bollinger Bands, Sentiment, Macro)
   - Hyperparameter tuning (GridSearchCV, Optuna)
   - Walk-forward validation / Time Series Cross-Validation
   - Risk management & position sizing
4. Data `yfinance` kadang memiliki missing values.

---

## 📁 Struktur Proyek

```text
bbca-stock-prediction/
├── stock_prediction.py    # Script utama
├── requirements.txt       # Dependensi Python
└── README.md              # Dokumentasi ini
```

---

## 📄 Lisensi

Proyek ini untuk tujuan pembelajaran dan penelitian. Bebas digunakan dan dimodifikasi.

---

**Dibuat dengan ❤️ untuk riset kuantitatif sederhana.**