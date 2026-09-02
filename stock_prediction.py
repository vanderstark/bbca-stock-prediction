import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import datetime

# --- Konfigurasi ---
TICKER = "BBCA.JK"
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=10*365)).strftime('%Y-%m-%d')
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
TEST_SIZE = 0.2
RANDOM_STATE = 42

print(f"Mengambil data historis untuk {TICKER} dari {START_DATE} sampai {END_DATE}...")

# --- 1. Pengambilan Data ---
data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
if data.empty:
    raise ValueError("Tidak ada data yang ditemukan untuk ticker ini. Cek kode ticker.")

# Ensure columns are flat (not MultiIndex)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f"Data berhasil diambil. Jumlah baris: {len(data)}")
print(f"Kolom: {list(data.columns)}")

# --- 2. Feature Engineering ---
# Target: prediksi arah harga (naik/turun)
data['Next_Day_Close'] = data['Close'].shift(-1)
data['Target'] = (data['Next_Day_Close'] > data['Close']).astype(int)

# Indikator teknikal
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data['High_Low_Diff'] = data['High'] - data['Low']
data['Open_Close_Diff'] = data['Open'] - data['Close']
data['Volume'] = data['Volume']

data = data.dropna()
print(f"Data setelah feature engineering: {len(data)} baris.")

# --- 3. Pemisahan Fitur dan Target ---
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'High_Low_Diff', 'Open_Close_Diff']
X = data[features]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=data.index)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

print(f"Data latih: {len(X_train)} baris, Data uji: {len(X_test)} baris.")

# --- 4. Implementasi dan Evaluasi Model ---
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM (RBF)': SVC(kernel='rbf', random_state=RANDOM_STATE),
    'LASSO (L1 LogReg)': LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE, C=0.1),
}

results = {}
for name, model in models.items():
    print(f"\n--- Melatih {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, zero_division=0)
    }

# PCA + Random Forest
print("\n--- Melatih PCA + Random Forest ---")
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf_pca_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
rf_pca_model.fit(X_train_pca, y_train)
y_pred_rf_pca = rf_pca_model.predict(X_test_pca)
results['PCA + Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf_pca),
    'report': classification_report(y_test, y_pred_rf_pca, zero_division=0)
}

# --- Hasil Akhir ---
print("\n" + "=" * 60)
print("RINGKASAN HASIL MODEL")
print("=" * 60)
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    print(f"Akurasi: {metrics['accuracy']:.4f}")
    print("Classification Report:")
    print(metrics['report'])