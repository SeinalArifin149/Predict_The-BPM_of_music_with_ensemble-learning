import joblib
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi BPM Musik", layout="wide")

# --- 2. FUNGSI LOADING BERAT (CACHE) ---
# Fungsi ini melakukan pekerjaan berat: Load Model & Hitung MAE Global
# @st.cache_resource memastikan ini cuma jalan sekali di server
@st.cache_resource
def load_heavy_resources():
    try:
        # Load Model
        model = joblib.load("my_model.pkl")
        
        # Load Data Train
        df = pd.read_csv('train.csv')
        
        # Hitung MAE Global (Proses Lama)
        target_col = "BeatsPerMinute"
        cols_to_drop = ["id", target_col]
        features = [col for col in df.columns if col not in cols_to_drop]
        
        X_all = df[features]
        y_true = df[target_col]
        y_pred_all = model.predict(X_all)
        mae_global = mean_absolute_error(y_true, y_pred_all)
        
        return model, df, features, mae_global, None

    except Exception as e:
        return None, None, None, None, str(e)

# --- 3. LOGIKA SPLASH SCREEN (ANIMASI LOADING AWAL) ---
# Kita cek apakah data sudah siap di session_state?
if 'app_ready' not in st.session_state:
    st.session_state['app_ready'] = False

if not st.session_state['app_ready']:
    # --- TAMPILAN LOADING SCREEN ---
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("## 🎵 Sedang Mempersiapkan AI...")
        st.info("Sistem sedang membaca pola audio dan menghitung akurasi model. Mohon tunggu...")
        
        # Animasi Progress Bar
        bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.015) # Mainkan angka ini kalau mau loading lebih lama/cepat
            bar.progress(i)
        
        # Panggil Fungsi Berat
        model, df, features, mae_val, err = load_heavy_resources()
        
        if err:
            st.error(f"Gagal memuat sistem: {err}")
            st.warning("Pastikan file 'my_model.pkl' dan 'train.csv' ada di folder yang sama.")
            st.stop()
        
        # Simpan ke Session State
        st.session_state['model'] = model
        st.session_state['df'] = df
        st.session_state['features'] = features
        st.session_state['mae_global'] = mae_val
        st.session_state['app_ready'] = True
        
        st.success("✅ Sistem Siap!")
        time.sleep(0.5)
    
    # Hapus loading screen dan refresh ke menu utama
    placeholder.empty()
    st.rerun()

# =========================================================
# --- 4. HALAMAN UTAMA (Hanya muncul setelah loading) ---
# =========================================================

# Ambil data dari memory
model = st.session_state['model']
df = st.session_state['df']
features = st.session_state['features']
mae_global = st.session_state['mae_global']

st.title("🎧 Predict Beats-Per-Minute (BPM)")
st.write("Geser Slider atau Ketik Angka untuk memprediksi tempo lagu.")

# --- FUNGSI SINKRONISASI ---
def update_input(key):
    st.session_state[f"input_{key}"] = st.session_state[f"slider_{key}"]

def update_slider(key):
    st.session_state[f"slider_{key}"] = st.session_state[f"input_{key}"]

# --- INISIALISASI NILAI AWAL ---
for feat in features:
    if f"slider_{feat}" not in st.session_state:
        default_val = float(df[feat].mean())
        if "Duration" in feat or "Year" in feat:
             default_val = int(default_val)
        st.session_state[f"slider_{feat}"] = default_val
        st.session_state[f"input_{feat}"] = default_val

# --- BUILD UI WIDGET ---
def buat_input_sinkron(label, col_name):
    c1, c2 = st.columns([3, 1]) 
    min_v = float(df[col_name].min())
    max_v = float(df[col_name].max())
    
    # Deteksi Integer
    is_int = df[col_name].dtype == 'int64' and df[col_name].mean() > 100
    
    if is_int:
        step, fmt, min_v, max_v = 100, "%d", int(min_v), int(max_v)
    else:
        # Perbaikan di sini: tambahkan min_v dan max_v di sebelah kiri
        step, fmt, min_v, max_v = 0.001, "%.3f", float(min_v), float(max_v)
        
    with c1:
        st.slider(f"{label}", min_v, max_v, step=step, format=fmt,
                  key=f"slider_{col_name}", on_change=update_input, args=(col_name,))
    with c2:
        st.number_input("Input", min_v, max_v, step=step, format=fmt,
                        key=f"input_{col_name}", on_change=update_slider, args=(col_name,),
                        label_visibility="collapsed")

# Tampilkan Layout 2 Kolom
half = len(features) // 2
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("🎵 Audio Features A")
    for feat in features[:half]:
        buat_input_sinkron(feat, feat)

with col_right:
    st.subheader("🎹 Audio Features B")
    for feat in features[half:]:
        buat_input_sinkron(feat, feat)

st.divider()

# --- 5. EKSEKUSI (INSTANT PREDICTION) ---
if st.button("🚀 TEBAK BPM SEKARANG", type="primary", use_container_width=True):
    
    # Ambil Input
    input_data = [st.session_state[f"input_{feat}"] for feat in features]
    
    # PENTING: Bungkus jadi DataFrame biar nama kolom kebaca model
    X_input = pd.DataFrame([input_data], columns=features)

    # Prediksi
    hasil_prediksi = model.predict(X_input)[0]
    
    # Hitung Akurasi (MAE sudah dihitung di loading awal)
    acc_val = max(0, 100 - mae_global) 

    # Tampilkan Hasil
    st.balloons() # Efek Balon
    st.success("✅ Prediksi Selesai!")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Prediksi BPM", f"{hasil_prediksi:.2f}")
    with m2:
        st.metric("Model MAE", f"{mae_global:.3f}", 
                  delta=f"-{mae_global:.3f}", delta_color="inverse",
                  help="Error rata-rata model (dihitung saat loading awal)")
    with m3:
        st.metric("Estimasi Akurasi", f"{acc_val:.2f}", 
                  delta=f"{acc_val:.2f}")