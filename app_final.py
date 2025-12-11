import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from scipy.stats import linregress

# -----------------------------------------------------------------------------
# 1. AYARLAR VE YÜKLEME
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Nefropati - Tam Sistem", layout="wide", page_icon="🧠")

@st.cache_resource
def load_ai_assets():
    try:
        model = load_model('model_full_v2.keras')
        scaler = joblib.load('scaler_v2.pkl')
        features = joblib.load('features_list_v2.pkl')
        return model, scaler, features
    except Exception as e:
        return None, None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('mimic_t1d_final_v2.csv') # 94 MB'lık dosya
        df['charttime'] = pd.to_datetime(df['charttime'])
        df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
        
        # eGFR Hesabı (Filtreleme için gerekli)
        age_col = 'age' if 'age' in df.columns else 'anchor_age'
        
        def calc_egfr(row):
            if row['lab_name'] == 'Creatinine':
                scr = row['valuenum']
                if pd.isna(scr) or scr <= 0: return np.nan
                current_age = row[age_col]
                is_female = 1 if row['gender'] == 'F' else 0
                k = 0.7 if is_female else 0.9
                alpha = -0.329 if is_female else -0.411
                factor = 1.018 if is_female else 1.0
                try:
                    egfr = 142 * (min(scr/k, 1)**alpha) * (max(scr/k, 1)**-1.200) * (0.9938**current_age) * factor
                except: return np.nan
                return egfr
            return np.nan

        df['eGFR'] = df.apply(calc_egfr, axis=1)
        return df, age_col
    except FileNotFoundError:
        return None, None

# Yükle
model, scaler, feature_list = load_ai_assets()
df_raw, age_col_name = load_data()

if df_raw is None or model is None:
    st.error("🚨 Dosyalar eksik! (csv, keras, pkl dosyalarını kontrol edin)")
    st.stop()

# -----------------------------------------------------------------------------
# 2. HASTA GRUPLANDIRMA VE FİLTRELEME (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Hasta Gezgini")

@st.cache_data
def categorize_patients(dataframe):
    # Gruplama için özet tablo oluştur
    p_risk = dataframe.groupby('subject_id')['eGFR'].min().reset_index()
    p_gender = dataframe.groupby('subject_id')['gender'].first().reset_index()
    
    hba1c_data = dataframe[dataframe['lab_name'] == 'HbA1c'].sort_values('charttime')
    p_hba1c = hba1c_data.groupby('subject_id')['valuenum'].last().reset_index().rename(columns={'valuenum': 'last_hba1c'})
    
    summary = pd.merge(p_risk, p_gender, on='subject_id', how='left')
    summary = pd.merge(summary, p_hba1c, on='subject_id', how='left')
    
    def assign_risk(row):
        if pd.isna(row['eGFR']): return "⚪ Veri Yok"
        if row['eGFR'] < 60: return "🔴 Yüksek Risk (eGFR < 60)"
        if row['eGFR'] < 90: return "🟠 Orta Risk (eGFR 60-90)"
        return "🟢 Düşük Risk (eGFR > 90)"

    def assign_control(row):
        if pd.isna(row['last_hba1c']): return "⚪ Bilinmiyor"
        if row['last_hba1c'] < 7.0: return "✅ İyi Kontrol (< %7)"
        if row['last_hba1c'] < 9.0: return "⚠️ Orta Kontrol (%7-9)"
        return "⛔ Kötü Kontrol (> %9)"

    summary['Risk_Grubu'] = summary.apply(assign_risk, axis=1)
    summary['Kontrol_Grubu'] = summary.apply(assign_control, axis=1)
    return summary

patient_summary = categorize_patients(df_raw)

# --- FİLTRELER ---
st.sidebar.markdown("### 🎛️ Filtreleme Modu")
filter_mode = st.sidebar.radio(
    "Listeleme Kriteri:",
    ["Tüm Hastalar", "Böbrek Riskine Göre", "Diyabet Kontrolüne (HbA1c) Göre", "Cinsiyete Göre"]
)

filtered_patients = []

if filter_mode == "Tüm Hastalar":
    filtered_patients = patient_summary['subject_id'].unique()
    st.sidebar.caption(f"Toplam {len(filtered_patients)} hasta.")

elif filter_mode == "Böbrek Riskine Göre":
    groups = sorted(patient_summary['Risk_Grubu'].unique())
    sel = st.sidebar.selectbox("Risk Seviyesi:", groups)
    filtered_patients = patient_summary[patient_summary['Risk_Grubu'] == sel]['subject_id'].unique()

elif filter_mode == "Diyabet Kontrolüne (HbA1c) Göre":
    groups = sorted(patient_summary['Kontrol_Grubu'].unique())
    sel = st.sidebar.selectbox("HbA1c Durumu:", groups)
    filtered_patients = patient_summary[patient_summary['Kontrol_Grubu'] == sel]['subject_id'].unique()

elif filter_mode == "Cinsiyete Göre":
    sel = st.sidebar.selectbox("Cinsiyet:", ["F", "M"])
    filtered_patients = patient_summary[patient_summary['gender'] == sel]['subject_id'].unique()

# --- HASTA SEÇİMİ ---
if len(filtered_patients) > 0:
    st.sidebar.markdown("---")
    selected_id = st.sidebar.selectbox("📂 Hasta ID Seçin:", filtered_patients)
else:
    st.sidebar.error("Kriterlere uygun hasta yok.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. YAPAY ZEKA VERİ HAZIRLIĞI
# -----------------------------------------------------------------------------
def prepare_ai_input(patient_id, full_df):
    p_df = full_df[full_df['subject_id'] == patient_id].copy()
    
    # Pivot
    p_pivot = p_df.pivot_table(
        index=['charttime', 'gender', age_col_name], 
        columns='lab_name', 
        values='valuenum', 
        aggfunc='mean'
    ).reset_index()
    p_pivot.columns.name = None
    p_pivot = p_pivot.sort_values('charttime')
    
    # Feature Engineering (Modelin beklediği sütunlar)
    if 'Glucose' in p_pivot.columns:
        p_pivot['Glucose_Mean'] = p_pivot['Glucose'].rolling(5, min_periods=1).mean()
        p_pivot['Glucose_Std'] = p_pivot['Glucose'].rolling(5, min_periods=1).std()
        p_pivot['Glucose_CV'] = p_pivot['Glucose_Std'] / (p_pivot['Glucose_Mean'] + 1e-9)
        p_pivot['Hypo_Count'] = (p_pivot['Glucose'] < 70).rolling(5, min_periods=1).sum()
        p_pivot['Hyper_Count'] = (p_pivot['Glucose'] > 250).rolling(5, min_periods=1).sum()
    
    # EKSİK SÜTUNLARI DOLDUR
    for col in feature_list:
        if col not in p_pivot.columns:
            p_pivot[col] = 0 
    
    if 'eGFR_Slope' not in p_pivot.columns: p_pivot['eGFR_Slope'] = 0.0 

    p_pivot = p_pivot.ffill().fillna(0)
    return p_pivot

patient_ai_data = prepare_ai_input(selected_id, df_raw)

# -----------------------------------------------------------------------------
# 4. ANA EKRAN (AI + GRAFİK)
# -----------------------------------------------------------------------------
st.title(f"Hasta: {selected_id}")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Klinik Seyir (Çift Eksen)")
    
    # Grafik Verisi Hazırlığı
    p_raw = df_raw[df_raw['subject_id'] == selected_id].sort_values('charttime')
    egfr_data = p_raw.dropna(subset=['eGFR'])
    glu_data = p_raw[p_raw['lab_name'] == 'Glucose']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if not egfr_data.empty:
        fig.add_trace(go.Scatter(x=egfr_data['charttime'], y=egfr_data['eGFR'], name="eGFR (Böbrek)", line=dict(color='#2ca02c', width=3)), secondary_y=False)
        fig.add_hline(y=60, line_dash="dash", line_color="red", secondary_y=False)
    
    if not glu_data.empty:
        fig.add_trace(go.Scatter(x=glu_data['charttime'], y=glu_data['valuenum'], name="Glikoz", line=dict(color='#1f77b4', dash='dot')), secondary_y=True)
        
    fig.update_layout(title="Zaman İçinde Şeker ve Böbrek İlişkisi", height=400)
    fig.update_yaxes(title_text="eGFR", secondary_y=False)
    fig.update_yaxes(title_text="Glikoz", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🤖 Yapay Zeka Tahmini")
    
    if len(patient_ai_data) < 3:
        st.warning("Yetersiz veri.")
    else:
        # --- TAHMİN KISMI (DÜZELTİLDİ) ---
        
        # 1. Veriyi DataFrame Olarak Hazırla (Scaler isimleri görsün)
        # Sadece modelin istediği özellikleri al
        X_input_df = patient_ai_data[feature_list].copy()
        
        # 2. Son 10 satırı al (LSTM için)
        if len(X_input_df) > 10:
            X_input_df = X_input_df.iloc[-10:]
            
        # 3. DataFrame ile Transform et (Uyarıyı önler)
        X_scaled = scaler.transform(X_input_df)
        
        # 4. Şekil Ver (1, 10, Features)
        X_reshaped = pad_sequences([X_scaled], maxlen=10, padding='pre', dtype='float32', value=-99.0)
        
        # 5. Predict
        prob = model.predict(X_reshaped, verbose=0)[0][0]
        percent = prob * 100
        
        # Gösterge
        if percent < 40:
            st.success(f"DÜŞÜK RİSK\n# %{percent:.1f}")
            st.caption("Böbrek fonksiyonları stabil görünüyor.")
        elif percent < 75:
            st.warning(f"ORTA RİSK\n# %{percent:.1f}")
            st.caption("Dikkat! Glikoz dalgalanmaları böbreği yoruyor olabilir.")
        else:
            st.error(f"🚨 YÜKSEK RİSK\n# %{percent:.1f}")
            st.caption("Yakın gelecekte böbrek fonksiyon kaybı öngörülüyor.")
            
        st.progress(int(percent))
        
        # Risk Faktörleri
        st.markdown("**Neden?**")
        last_row = patient_ai_data.iloc[-1]
        st.metric("Glikoz CV (Dalgalanma)", f"{last_row.get('Glucose_CV', 0):.2f}", delta="Riskli" if last_row.get('Glucose_CV', 0) > 0.36 else "Normal", delta_color="inverse")
        st.metric("Yüksek Şeker Sıklığı", int(last_row.get('Hyper_Count', 0)))

# -----------------------------------------------------------------------------
# 5. DETAYLI PANEL
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 15 Parametreli Laboratuvar Paneli")

available_labs = p_raw['lab_name'].unique()
default_labs = [l for l in ['Potassium', 'Hemoglobin', 'Albumin', 'BUN'] if l in available_labs]
selected_labs = st.multiselect("Test Seçin:", available_labs, default=default_labs)

if selected_labs:
    subset = p_raw[p_raw['lab_name'].isin(selected_labs)]
    fig_multi = px.scatter(subset, x='charttime', y='valuenum', color='lab_name', facet_col='lab_name', facet_col_wrap=4, height=300)
    fig_multi.update_yaxes(matches=None)
    st.plotly_chart(fig_multi, use_container_width=True)