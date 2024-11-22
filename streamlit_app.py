import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Judul Aplikasi
st.title("Rushland CSV Classifier - SVM")
st.write("Unggah file CSV Anda untuk melakukan klasifikasi menggunakan algoritma Support Vector Machine (SVM).")

# Fungsi untuk membaca file CSV
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"File berhasil dimuat! Kolom: {list(df.columns)}")
        return df
    except Exception as e:
        st.error(f"Gagal membaca file. Error: {e}")
        return None

# Fungsi untuk klasifikasi
def classify_data_svm(df):
    try:
        # Memilih kolom target
        target_column = st.selectbox("Pilih kolom target:", df.columns)
        feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Standarisasi data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train SVM
        model = SVC(kernel='linear', random_state=42)  # Gunakan kernel lain jika diperlukan
        model.fit(X_train, y_train)
        
        # Predict dan evaluasi
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Tampilkan hasil
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())
    except Exception as e:
        st.error(f"Klasifikasi gagal. Error: {e}")

# Input file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

# Proses jika file diunggah
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if df is not None:
        st.write("Pratinjau Data:")
        st.dataframe(df.head())
        classify_data_svm(df)
