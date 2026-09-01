# BBCA Stock Direction Prediction

Proyek ini memprediksi arah harga saham **BBCA (Bank Central Asia)** di Bursa Efek Indonesia (IDX) menggunakan algoritma Machine Learning: **Decision Trees, Random Forests, XGBoost, PCA, dan LASSO (Logistic Regression L1)**.

Data diambil dari **Yahoo Finance** (ticker: `BBCA.JK`) selama 10 tahun terakhir.

---

## 📋 Fitur Utama

- **Pengambilan Data Otomatis**: Menggunakan `yfinance` untuk mengambil data historis 10 tahun.
- **Feature Engineering**: Menghitung indikator teknikal (SMA, RSI, High-Low Diff, Open-Close Diff).
- **Model Machine Learning**:
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier
  - PCA + Decision Tree (Dimensionality Reduction)
  - LASSO (Logistic Regression dengan regularisasi L1)
- **Evaluasi Model**: Accuracy, Precision, Recall, F1-Score per model.
- **Output**: Ringkasan performa semua model di terminal.

---

## 🚀 Cara Menjalankan

### 1. Prasyarat
- Python 3.8+
- `pip` atau `uv` untuk manajemen paket

### 2. Instalasi Dependensi
```bash
# Opsi A: Menggunakan pip
pip install -r requirements.txt

# Opsi B: Menggunakan uv (lebih cepat)
uv pip install -r requirements.txt
```

Isi `requirements.txt`:
```text
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
```

### 3. Menjalankan Script
```bash
python stock_prediction.py
```

Atau jika menggunakan `uv`:
```bash
uv run python stock_prediction.py
```

---

## 📊 Output yang Dihasilkan

Script akan menampilkan:
1. Progres pengambilan data dari Yahoo Finance
2. Jumlah data setelah feature engineering
3. Progres pelatihan setiap model
4. **Ringkasan hasil**:
   - Akurasi setiap model
   - Classification Report (Precision, Recall, F1-Score) untuk kelas Naik (1) dan Turun (0)

Contoh output:
```text
===================================
=== RINGKASAN HASIL MODEL ===
===================================

Model: Decision Tree
Akurasi: 0.5234
Classification Report:
              precision    recall  f1-score   support

           0       0.51      0.54      0.53       450
           1       0.53      0.51      0.52       430

    accuracy                           0.52       880
   macro avg       0.52      0.52      0.52       880
weighted avg       0.52      0.52      0.52       880

...
```

---

## ⚙️ Konfigurasi yang Dapat Diubah

Di bagian atas script `stock_prediction.py`, Anda dapat mengubah:

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `TICKER` | `"BBCA.JK"` | Kode saham di Yahoo Finance (contoh: `"AAPL"` untuk Apple, `"BTC-USD"` untuk Bitcoin) |
| `START_DATE` | 10 tahun lalu | Rentang awal data (format `YYYY-MM-DD`) |
| `END_DATE` | Hari ini | Rentang akhir data |
| `TEST_SIZE` | `0.2` | Proporsi data untuk testing (20%) |
| `RANDOM_STATE` | `42` | Seed untuk reproducibility |

---

## 🧠 Penjelasan Model

| Model | Deskripsi |
|-------|-----------|
| **Decision Tree** | Model pohon keputusan sederhana, mudah diinterpretasikan. |
| **Random Forest** | Ensemble dari banyak Decision Tree, mengurangi overfitting. |
| **XGBoost** | Gradient Boosting yang dioptimasi untuk kecepatan dan performa. |
| **PCA + Decision Tree** | Mengurangi dimensi fitur dengan PCA (95% variance) lalu klasifikasi. |
| **LASSO (L1 Logistic Regression)** | Logistic Regression dengan regularisasi L1 untuk seleksi fitur otomatis. |

---

## 📝 Catatan Penting

1. **Ini adalah contoh edukasi**, bukan saran investasi.
2. Prediksi arah harga saham sangat sulit (efisiensi pasar). Akurasi ~50-55% adalah normal untuk baseline.
3. Untuk produksi, perlu:
   - Feature engineering lanjutan (MACD, Bollinger Bands, Sentiment, dll)
   - Hyperparameter tuning (GridSearchCV, Optuna)
   - Walk-forward validation / Time Series Cross-Validation
   - Risk management & position sizing
4. Data `yfinance` kadang memiliki missing values atau adjusted close yang perlu diverifikasi.

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