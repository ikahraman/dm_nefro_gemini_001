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
        df = pd.read_csv('mimic_t1d_final_v2.csv') 
        df['charttime'] = pd.to_datetime(df['charttime'])
        df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
        
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

model, scaler, feature_list = load_ai_assets()
df_raw, age_col_name = load_data()

if df_raw is None or model is None:
    st.error("🚨 Dosyalar eksik! (csv, keras, pkl dosyalarını kontrol edin)")
    st.stop()

# -----------------------------------------------------------------------------
# 2. SIDEBAR - GRUPLAMA
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Hasta Gezgini")

@st.cache_data
def categorize_patients(dataframe):
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

st.sidebar.markdown("### 🎛️ Filtreleme Modu")
filter_mode = st.sidebar.radio(
    "Listeleme Kriteri:",
    ["Tüm Hastalar", "Böbrek Riskine Göre", "Diyabet Kontrolüne (HbA1c) Göre", "Cinsiyete Göre"]
)

filtered_patients = []

if filter_mode == "Tüm Hastalar":
    filtered_patients = patient_summary['subject_id'].unique()
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

if len(filtered_patients) > 0:
    st.sidebar.markdown("---")
    selected_id = st.sidebar.selectbox("📂 Hasta ID Seçin:", filtered_patients)
else:
    st.sidebar.error("Kriterlere uygun hasta yok.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def prepare_ai_input_from_db(patient_id, full_df):
    p_df = full_df[full_df['subject_id'] == patient_id].copy()
    p_pivot = p_df.pivot_table(index=['charttime', 'gender', age_col_name], columns='lab_name', values='valuenum', aggfunc='mean').reset_index()
    p_pivot.columns.name = None
    p_pivot = p_pivot.sort_values('charttime')
    
    if 'Glucose' in p_pivot.columns:
        p_pivot['Glucose_Mean'] = p_pivot['Glucose'].rolling(5, min_periods=1).mean()
        p_pivot['Glucose_Std'] = p_pivot['Glucose'].rolling(5, min_periods=1).std()
        p_pivot['Glucose_CV'] = p_pivot['Glucose_Std'] / (p_pivot['Glucose_Mean'] + 1e-9)
        p_pivot['Hypo_Count'] = (p_pivot['Glucose'] < 70).rolling(5, min_periods=1).sum()
        p_pivot['Hyper_Count'] = (p_pivot['Glucose'] > 250).rolling(5, min_periods=1).sum()
    
    for col in feature_list:
        if col not in p_pivot.columns: p_pivot[col] = 0 
    
    if 'eGFR_Slope' not in p_pivot.columns: p_pivot['eGFR_Slope'] = 0.0 
    p_pivot = p_pivot.ffill().fillna(0)
    return p_pivot

patient_ai_data = prepare_ai_input_from_db(selected_id, df_raw)

# -----------------------------------------------------------------------------
# 4. DASHBOARD SEKME YAPISI
# -----------------------------------------------------------------------------
st.title(f"Hasta: {selected_id}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Klinik Seyir", "🧪 Geniş Panel", "📋 Detaylı Veri", "🔮 Simülasyon / Manuel Tahmin"])

# --- TAB 1: KLİNİK SEYİR ---
with tab1:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Glikoz vs. Böbrek İlişkisi")
        p_raw = df_raw[df_raw['subject_id'] == selected_id].sort_values('charttime')
        egfr_data = p_raw.dropna(subset=['eGFR'])
        glu_data = p_raw[p_raw['lab_name'] == 'Glucose']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if not egfr_data.empty:
            fig.add_trace(go.Scatter(x=egfr_data['charttime'], y=egfr_data['eGFR'], name="eGFR", line=dict(color='#2ca02c', width=3)), secondary_y=False)
            fig.add_hline(y=60, line_dash="dash", line_color="red", secondary_y=False)
        if not glu_data.empty:
            fig.add_trace(go.Scatter(x=glu_data['charttime'], y=glu_data['valuenum'], name="Glikoz", line=dict(color='#1f77b4', dash='dot')), secondary_y=True)
        fig.update_layout(height=400)
        fig.update_yaxes(title_text="eGFR", secondary_y=False)
        fig.update_yaxes(title_text="Glikoz", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🤖 Mevcut Durum Tahmini")
        if len(patient_ai_data) < 3:
            st.warning("Yetersiz veri.")
        else:
            X_input_df = pd.DataFrame(patient_ai_data[feature_list].values, columns=feature_list)
            if len(X_input_df) > 10: X_input_df = X_input_df.iloc[-10:]
            X_scaled = scaler.transform(X_input_df)
            X_reshaped = pad_sequences([X_scaled], maxlen=10, padding='pre', dtype='float32', value=-99.0)
            
            prob = model.predict(X_reshaped, verbose=0)[0][0]
            percent = prob * 100
            
            if percent < 40: st.success(f"DÜŞÜK RİSK\n# %{percent:.1f}")
            elif percent < 75: st.warning(f"ORTA RİSK\n# %{percent:.1f}")
            else: st.error(f"🚨 YÜKSEK RİSK\n# %{percent:.1f}")
            st.progress(int(percent))
            
            last_row = patient_ai_data.iloc[-1]
            st.metric("Glikoz CV", f"{last_row.get('Glucose_CV', 0):.2f}")
            st.metric("Hiperglisemi Sıklığı", int(last_row.get('Hyper_Count', 0)))

# --- TAB 2: GENİŞ PANEL ---
with tab2:
    available_labs = p_raw['lab_name'].unique()
    default_labs = [l for l in ['Potassium', 'Hemoglobin', 'Albumin', 'BUN'] if l in available_labs]
    selected_labs = st.multiselect("Test Seçin:", available_labs, default=default_labs)
    if selected_labs:
        subset = p_raw[p_raw['lab_name'].isin(selected_labs)]
        fig_multi = px.scatter(subset, x='charttime', y='valuenum', color='lab_name', facet_col='lab_name', facet_col_wrap=4, height=300)
        fig_multi.update_yaxes(matches=None)
        st.plotly_chart(fig_multi, use_container_width=True)

# --- TAB 3: HAM VERİ ---
with tab3:
    st.dataframe(patient_ai_data.sort_values('charttime', ascending=False), use_container_width=True)

# --- TAB 4: SİMÜLASYON / MANUEL TAHMİN (YENİ EKLENEN KISIM) ---
with tab4:
    st.header("🔮 Manuel Risk Simülasyonu")
    st.markdown("Veritabanında olmayan bir hasta için veya 'Glikozum şöyle olsaydı ne olurdu?' senaryoları için kullanın.")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.subheader("1. Hasta Verileri")
        sim_age = st.number_input("Yaş", 10, 90, 45)
        sim_hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 8.5)
        
        st.markdown("**Laboratuvar Değerleri (Mevcut)**")
        sim_creat = st.number_input("Kreatinin (mg/dL)", 0.1, 10.0, 1.2)
        sim_bun = st.number_input("BUN", 1, 100, 20)
        sim_potassium = st.number_input("Potasyum", 1.0, 10.0, 4.0)
        sim_hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)
        
    with col_sim2:
        st.subheader("2. Geçmiş Veri (Trend Analizi)")
        st.info("Virgülle ayırarak giriniz (En eskiden -> En yeniye)")
        
        sim_glucose_hist = st.text_area("Son 5-10 Glikoz Ölçümü (mg/dL)", "120, 130, 250, 80, 300, 110")
        sim_creat_hist = st.text_area("Son 3-5 Kreatinin Ölçümü (eGFR Eğim İçin)", "1.0, 1.1, 1.2, 1.3")
        
        calc_btn = st.button("Simülasyonu Çalıştır", type="primary")

    if calc_btn:
        st.divider()
        try:
            # Girdileri Parse Et
            gl_vals = [float(x.strip()) for x in sim_glucose_hist.split(',')]
            cr_vals = [float(x.strip()) for x in sim_creat_hist.split(',')]
            
            # --- FEATURE ENGINEERING (MANUEL) ---
            # 1. Glikoz İstatistikleri
            gl_series = pd.Series(gl_vals)
            sim_gl_mean = gl_series.mean()
            sim_gl_std = gl_series.std() if len(gl_vals) > 1 else 0
            sim_gl_cv = sim_gl_std / (sim_gl_mean + 1e-9)
            sim_hypo = (gl_series < 70).sum()
            sim_hyper = (gl_series > 250).sum()
            
            # 2. eGFR Slope Hesabı
            # Basitçe kreatininleri eGFR'ye çevirip eğim alalım
            egfr_vals = []
            for cr in cr_vals:
                # Erkek varsayalım basitlik için (veya cinsiyet inputu eklenebilir)
                # Formül: 142 * min(scr/0.9, 1)**-0.411 * max(scr/0.9, 1)**-1.209 * 0.9938**age
                # Basitleştirilmiş demo hesabı:
                e = 142 * (min(cr/0.9, 1)**-0.411) * (max(cr/0.9, 1)**-1.209) * (0.9938**sim_age)
                egfr_vals.append(e)
            
            if len(egfr_vals) > 1:
                slope, _, _, _, _ = linregress(range(len(egfr_vals)), egfr_vals)
            else:
                slope = 0
            
            # 3. Model Girdisi Oluştur (Sözlük olarak hazırla)
            # Modelin beklediği tüm sütunlar (feature_list) olmalı
            input_dict = {col: 0.0 for col in feature_list} # Önce hepsini 0 yap
            
            # Bildiklerimizi doldur
            input_dict['Glucose'] = gl_vals[-1]
            input_dict['HbA1c'] = sim_hba1c
            input_dict['Creatinine'] = sim_creat
            input_dict['BUN'] = sim_bun
            input_dict['Potassium'] = sim_potassium
            input_dict['Hemoglobin'] = sim_hemoglobin
            input_dict['Glucose_CV'] = sim_gl_cv
            input_dict['Hypo_Count'] = sim_hypo
            input_dict['Hyper_Count'] = sim_hyper
            input_dict['eGFR_Slope'] = slope
            
            # DataFrame'e çevir
            df_sim = pd.DataFrame([input_dict])
            
            # 4. TAHMİN
            # Tek bir satırı 10 kez çoğalt (Statik Film Tekniği - Padding sorununu aşmak için)
            X_sim_scaled = scaler.transform(df_sim)
            X_sim_reshaped = np.tile(X_sim_scaled, (1, 10, 1)) # (1, 10, Features)
            
            sim_prob = model.predict(X_sim_reshaped, verbose=0)[0][0]
            sim_percent = sim_prob * 100
            
            # 5. SONUÇ EKRANI
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.metric("Tahmini Risk Skoru", f"%{sim_percent:.1f}")
                if sim_percent > 70:
                    st.error("YÜKSEK RİSK")
                elif sim_percent > 40:
                    st.warning("ORTA RİSK")
                else:
                    st.success("DÜŞÜK RİSK")
                    
            with c_res2:
                st.write("**Modelin Gördüğü Kritik Göstergeler:**")
                st.write(f"- **Glikoz Kararsızlığı (CV):** {sim_gl_cv:.2f}")
                st.write(f"- **eGFR Eğimi (Trend):** {slope:.2f} ml/min/yıl")
                st.write(f"- **Hiperglisemi Atakları:** {sim_hyper} kez")
                
        except Exception as e:
            st.error(f"Hesaplama hatası: {e}")