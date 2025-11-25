import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi BPM Musik", layout="wide")

st.title("🎧 Predict Beats-Per-Minute (BPM)")
st.write("Masukkan fitur audio di bawah, AI akan menebak angka BPM-nya.")

# --- 1. LOAD MODEL & DATA ---
try:
    model = joblib.load("my_model.pkl")
except FileNotFoundError:
    st.error("❌ Model 'my_model.pkl' tidak ditemukan bang.")
    st.stop()

try:
    df = pd.read_csv('test.csv')
except FileNotFoundError:
    st.error("❌ File 'test.csv' gak ketemu bang.")
    st.stop()

# --- 2. SETUP INPUT (9 FITUR) ---
# Format 3 angka belakang koma
koma = 3 

# Bagi jadi 3 Baris (3 kolom per baris)
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)

# === BARIS 1 ===
with row1[0]:
    RhythmScore_in = st.slider("RhythmScore", 
        min_value=round(float(df['RhythmScore'].min()), koma), 
        max_value=round(float(df['RhythmScore'].max()), koma), 
        value=round(float(df['RhythmScore'].mean()), koma), format="%.3f")
with row1[1]:
    AudioLoudness_in = st.slider("AudioLoudness", 
        min_value=round(float(df['AudioLoudness'].min()), koma), 
        max_value=round(float(df['AudioLoudness'].max()), koma), 
        value=round(float(df['AudioLoudness'].mean()), koma), format="%.3f")
with row1[2]:
    VocalContent_in = st.slider("VocalContent", 
        min_value=round(float(df['VocalContent'].min()), koma), 
        max_value=round(float(df['VocalContent'].max()), koma), 
        value=round(float(df['VocalContent'].mean()), koma), format="%.3f")

# === BARIS 2 ===
with row2[0]:
    AcousticQuality_in = st.slider("AcousticQuality", 
        min_value=round(float(df['AcousticQuality'].min()), koma), 
        max_value=round(float(df['AcousticQuality'].max()), koma), 
        value=round(float(df['AcousticQuality'].mean()), koma), format="%.3f")
with row2[1]:
    InstrumentalScore_in = st.slider("InstrumentalScore", 
        min_value=round(float(df['InstrumentalScore'].min()), koma), 
        max_value=round(float(df['InstrumentalScore'].max()), koma), 
        value=round(float(df['InstrumentalScore'].mean()), koma), format="%.3f")
with row2[2]:
    LivePerformanceLikelihood_in = st.slider("Live Perf. Likelihood", 
        min_value=round(float(df['LivePerformanceLikelihood'].min()), koma), 
        max_value=round(float(df['LivePerformanceLikelihood'].max()), koma), 
        value=round(float(df['LivePerformanceLikelihood'].mean()), koma), format="%.3f")

# === BARIS 3 ===
with row3[0]:
    MoodScore_in = st.slider("MoodScore", 
        min_value=round(float(df['MoodScore'].min()), koma), 
        max_value=round(float(df['MoodScore'].max()), koma), 
        value=round(float(df['MoodScore'].mean()), koma), format="%.3f")

with row3[1]:
    # --- FITUR YANG TADI HILANG (TrackDurationMs) ---
    # Karena ini milidetik (integer besar), kita gak pake koma, dan stepnya 1000ms
    TrackDurationMs_in = st.slider("TrackDurationMs (Durasi)", 
        min_value=int(df['TrackDurationMs'].min()), 
        max_value=int(df['TrackDurationMs'].max()), 
        value=int(df['TrackDurationMs'].mean()), 
        step=1000)

with row3[2]:
    Energy_in = st.slider("Energy", 
        min_value=round(float(df['Energy'].min()), koma), 
        max_value=round(float(df['Energy'].max()), koma), 
        value=round(float(df['Energy'].mean()), koma), format="%.3f")

st.divider()

# --- 3. EKSEKUSI ---
col_btn, col_result = st.columns([1, 2])

with col_btn:
    predict_btn = st.button("🎵 TEBAK BPM", use_container_width=True)

if predict_btn:
    # Urutan ini HARUS SAMA dengan urutan kolom di CSV abang (selain ID)
    # RhythmScore, AudioLoudness, VocalContent, AcousticQuality, InstrumentalScore, LivePerformanceLikelihood, MoodScore, TrackDurationMs, Energy
    X_input = [[
        RhythmScore_in, 
        AudioLoudness_in, 
        VocalContent_in, 
        AcousticQuality_in,
        InstrumentalScore_in, 
        LivePerformanceLikelihood_in, 
        MoodScore_in, 
        TrackDurationMs_in, # <--- Sudah dimasukkan!
        Energy_in
    ]]

    try:
        # Prediksi
        predicted_bpm = model.predict(X_input)[0]
        
        # Cek apakah bisa hitung MAE (Butuh kolom target asli)
        mae_text = "N/A (Test Data)"
        target_col = 'BeatsPerMinute' 
        
        if target_col in df.columns:
            # Ambil fitur yg sama persis buat test
            cols_urutan = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']
            X_test_all = df[cols_urutan]
            y_true = df[target_col]
            y_pred_all = model.predict(X_test_all)
            mae_val = mean_absolute_error(y_true, y_pred_all)
            mae_text = f"{mae_val:.3f}"
        else:
             mae_note = "(File test.csv tidak punya kolom 'BeatsPerMinute' buat ngecek jawaban)"

        with col_result:
            st.success("✅ Prediksi Selesai!")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Prediksi BPM", f"{predicted_bpm:.2f}")
            with c2:
                st.metric("MAE (Error)", mae_text, help="Hanya muncul kalau ada kunci jawaban di CSV")
            
            if mae_text == "N/A (Test Data)":
                st.caption(mae_note)
            
    except Exception as e:
        st.error(f"Masih error bang: {e}")