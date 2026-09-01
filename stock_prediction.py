import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import datetime

# --- Konfigurasi ---
TICKER = "BBCA.JK"  # Kode saham BBCA di Yahoo Finance (untuk IDX)
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=10*365)).strftime('%Y-%m-%d')
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
TEST_SIZE = 0.2    # 20% data untuk testing
RANDOM_STATE = 42

print(f"Mengambil data historis untuk {TICKER} dari {START_DATE} sampai {END_DATE}...")

# --- 1. Pengambilan Data ---
try:
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    if data.empty:
        raise ValueError("Tidak ada data yang ditemukan untuk ticker ini. Cek kode ticker.")
except Exception as e:
    print(f"Error saat mengambil data: {e}")
    print("Pastikan ticker sudah benar (contoh: BBCA.JK untuk BBCA di IDX).")
    exit()

print(f"Data berhasil diambil. Jumlah baris: {len(data)}")

# --- 2. Feature Engineering ---
# Target: prediksi arah harga (naik/turun)
# Shift the 'Close' price to get the next day's closing price
data['Next_Day_Close'] = data['Close'].shift(-1)
# 1 if next day's close is higher than current day's close, else 0
data['Target'] = (data['Next_Day_Close'] > data['Close']).astype(int)

# Indikator teknikal sederhana
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()

# RSI (Relative Strength Index)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Fitur tambahan
data['High_Low_Diff'] = data['High'] - data['Low']
data['Open_Close_Diff'] = data['Open'] - data['Close']
data['Volume'] = data['Volume']

# Hapus baris dengan nilai NaN yang muncul akibat rolling window/shift
data = data.dropna()

if data.empty:
    print("Setelah preprocessing dan dropna, tidak ada data tersisa. Sesuaikan rentang waktu atau fitur.")
    exit()

print(f"Data setelah feature engineering: {len(data)} baris.")

# --- 3. Pemisahan Fitur dan Target ---
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'High_Low_Diff', 'Open_Close_Diff']
X = data[features]
y = data['Target']

# Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=data.index)

# Pemisahan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

print(f"Data latih: {len(X_train)} baris, Data uji: {len(X_test)} baris.")

# --- 4. Implementasi dan Evaluasi Model ---

results = {}

# Model: Decision Tree
print("\n--- Melatih Decision Tree Classifier ---")
dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'report': classification_report(y_test, y_pred_dt)
}

# Model: Random Forest
print("\n--- Melatih Random Forest Classifier ---")
rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'report': classification_report(y_test, y_pred_rf)
}

# Model: XGBoost
print("\n--- Melatih XGBoost Classifier ---")
xgb_model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'report': classification_report(y_test, y_pred_xgb)
}

# Model: PCA + Decision Tree (contoh kombinasi)
print("\n--- Melatih PCA + Decision Tree ---")
pca = PCA(n_components=0.95) # Ambil 95% variansi
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

dt_pca_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_pca_model.fit(X_train_pca, y_train)
y_pred_dt_pca = dt_pca_model.predict(X_test_pca)
results['PCA + Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt_pca),
    'report': classification_report(y_test, y_pred_dt_pca)
}

# Model: LASSO (Logistic Regression with L1 regularization untuk klasifikasi)
print("\n--- Melatih LASSO (Logistic Regression with L1) ---")
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE, C=0.1) # C is inverse of regularization strength
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
results['LASSO (L1 Logistic Regression)'] = {
    'accuracy': accuracy_score(y_test, y_pred_lasso),
    'report': classification_report(y_test, y_pred_lasso)
}


# --- Hasil Akhir ---
print("\n===================================")
print("=== RINGKASAN HASIL MODEL ===")
print("===================================")
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    print(f"Akurasi: {metrics['accuracy']:.4f}")
    print("Classification Report:")
    print(metrics['report'])

print("\n--- Catatan ---")
print("C adalah parameter regulasi L1, semakin kecil C, semakin kuat regulasi L1.")
print("Untuk PCA, kami mempertahankan 95% variansi. Jumlah komponen yang terpilih: ", pca.n_components_)
print("Model ini hanya contoh dasar. Tuning parameter, feature engineering lanjutan, dan validasi yang lebih ketat sangat diperlukan untuk penggunaan di dunia nyata.")
