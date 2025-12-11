import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# -----------------------------------------------------------------------------
# 1. SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="T1DM & Nefropati Analiz Paneli",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE İŞLEME (ETL)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('type_1_and_renal_damage.csv')
    except FileNotFoundError:
        st.error("❌ 'type_1_and_renal_damage.csv' dosyası bulunamadı! Lütfen dosyayı proje klasörüne ekleyin.")
        st.stop()
        
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    age_col = 'age'
    if 'real_age' in df.columns:
        age_col = 'real_age'
    elif 'age_at_test' in df.columns:
        age_col = 'age_at_test'
    
    # --- eGFR HESAPLAMA ---
    def calc_egfr(row):
        if row['lab_name'] == 'Creatinine':
            scr = row['valuenum']
            if pd.isna(scr) or scr <= 0: return np.nan
            
            current_age = row[age_col]
            is_female = 1 if row['gender'] == 'F' else 0
            
            if is_female:
                k = 0.7; alpha = -0.329; factor = 1.018
            else:
                k = 0.9; alpha = -0.411; factor = 1
            
            try:
                egfr = 142 * (min(scr/k, 1)**alpha) * (max(scr/k, 1)**-1.200) * (0.9938**current_age) * factor
            except Exception:
                return np.nan
            return egfr
        return np.nan

    df['eGFR'] = df.apply(calc_egfr, axis=1)
    return df, age_col

df, age_col_name = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR - AKILLI FİLTRELEME (GÜNCELLENDİ: GLİKOZ CV EKLENDİ)
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Hasta Gezgini")

@st.cache_data
def categorize_patients(dataframe):
    # 1. Böbrek Riski
    p_risk = dataframe.groupby('subject_id')['eGFR'].min().reset_index()
    
    # 2. Cinsiyet
    p_gender = dataframe.groupby('subject_id')['gender'].first().reset_index()
    
    # 3. HbA1c
    hba1c_data = dataframe[dataframe['lab_name'] == 'HbA1c'].sort_values('charttime')
    p_hba1c = hba1c_data.groupby('subject_id')['valuenum'].last().reset_index().rename(columns={'valuenum': 'last_hba1c'})
    
    # 4. YENİ: Glikoz Dalgalanması (CV)
    # Her hastanın tüm glikoz ölçümlerini alıp CV hesaplıyoruz
    glu_data = dataframe[dataframe['lab_name'] == 'Glucose']
    p_cv = glu_data.groupby('subject_id')['valuenum'].agg(['mean', 'std']).reset_index()
    p_cv['glucose_cv'] = p_cv['std'] / p_cv['mean']
    
    # Hepsini Birleştir
    summary = pd.merge(p_risk, p_gender, on='subject_id', how='left')
    summary = pd.merge(summary, p_hba1c, on='subject_id', how='left')
    summary = pd.merge(summary, p_cv[['subject_id', 'glucose_cv']], on='subject_id', how='left')
    
    # Gruplama Mantıkları
    def assign_risk_group(row):
        if pd.isna(row['eGFR']): return "⚪ Veri Yok"
        elif row['eGFR'] < 60: return "🔴 Yüksek Risk (eGFR < 60)"
        elif row['eGFR'] < 90: return "🟠 Orta Risk (eGFR 60-90)"
        else: return "🟢 Düşük Risk (eGFR > 90)"

    def assign_control_group(row):
        if pd.isna(row['last_hba1c']): return "⚪ Bilinmiyor"
        elif row['last_hba1c'] < 7.0: return "✅ İyi Kontrol (< %7)"
        elif row['last_hba1c'] < 9.0: return "⚠️ Orta Kontrol (%7-9)"
        else: return "⛔ Kötü Kontrol (> %9)"

    # YENİ: CV Gruplama Mantığı
    def assign_cv_group(row):
        if pd.isna(row['glucose_cv']): return "⚪ Yetersiz Veri"
        elif row['glucose_cv'] < 0.20: return "🔹 Stabil (< 0.20)"
        elif row['glucose_cv'] < 0.36: return "🔸 Orta Dalgalı (0.20-0.36)"
        else: return "⚡ Yüksek Dalgalanma (> 0.36)" # En Riskli Grup

    summary['Risk_Grubu'] = summary.apply(assign_risk_group, axis=1)
    summary['Kontrol_Grubu'] = summary.apply(assign_control_group, axis=1)
    summary['CV_Grubu'] = summary.apply(assign_cv_group, axis=1)
    
    return summary

patient_summary = categorize_patients(df)

# --- FİLTRELEME SEÇENEKLERİ (GÜNCELLENDİ) ---
st.sidebar.markdown("### 🎛️ Filtreleme Modu")
filter_mode = st.sidebar.radio(
    "Listeleme Kriteri:",
    [
        "Tüm Hastalar", 
        "Böbrek Riskine Göre", 
        "Glikoz Dalgalanmasına (CV) Göre", # Yeni Seçenek
        "Diyabet Kontrolüne (HbA1c) Göre", 
        "Cinsiyete Göre"
    ]
)

filtered_patients = []

if filter_mode == "Tüm Hastalar":
    filtered_patients = patient_summary['subject_id'].unique()
    st.sidebar.caption(f"Toplam {len(filtered_patients)} hasta.")

elif filter_mode == "Böbrek Riskine Göre":
    groups = sorted(patient_summary['Risk_Grubu'].unique())
    sel = st.sidebar.selectbox("Risk Seviyesi:", groups)
    filtered_patients = patient_summary[patient_summary['Risk_Grubu'] == sel]['subject_id'].unique()

elif filter_mode == "Glikoz Dalgalanmasına (CV) Göre":
    # Yeni Filtre Mantığı
    groups = sorted(patient_summary['CV_Grubu'].unique())
    sel = st.sidebar.selectbox("Dalgalanma Seviyesi:", groups)
    filtered_patients = patient_summary[patient_summary['CV_Grubu'] == sel]['subject_id'].unique()
    
    if "Yüksek Dalgalanma" in sel:
        st.sidebar.error("Bu hastaların şekeri çok kararsızdır. Böbrek hasarı riski en yüksek gruptur.")

elif filter_mode == "Diyabet Kontrolüne (HbA1c) Göre":
    groups = sorted(patient_summary['Kontrol_Grubu'].unique())
    sel = st.sidebar.selectbox("HbA1c Durumu:", groups)
    filtered_patients = patient_summary[patient_summary['Kontrol_Grubu'] == sel]['subject_id'].unique()

elif filter_mode == "Cinsiyete Göre":
    sel = st.sidebar.selectbox("Cinsiyet:", ["F", "M"])
    filtered_patients = patient_summary[patient_summary['gender'] == sel]['subject_id'].unique()

st.sidebar.caption(f"Seçilen kriterde **{len(filtered_patients)}** hasta bulundu.")

# --- HASTA SEÇİMİ ---
if len(filtered_patients) > 0:
    st.sidebar.markdown("---")
    selected_patient_id = st.sidebar.selectbox("📂 Hasta ID Seçin:", filtered_patients)
else:
    st.sidebar.error("Kriterlere uygun hasta yok.")
    st.stop()

# --- VERİ HAZIRLIĞI ---
p_df = df[df['subject_id'] == selected_patient_id].sort_values('charttime')
p_gender = p_df['gender'].iloc[0]
p_age = p_df[age_col_name].iloc[0]

# -----------------------------------------------------------------------------
# 4. ANA EKRAN - HASTA DETAYLARI
# -----------------------------------------------------------------------------
st.title(f"🩺 Hasta Kartı: {selected_patient_id}")

col1, col2, col3, col4 = st.columns(4)

# Son eGFR
egfr_history = p_df.dropna(subset=['eGFR'])
if not egfr_history.empty:
    last_egfr = egfr_history['eGFR'].iloc[-1]
    col1.metric("Son eGFR", f"{last_egfr:.1f}", delta="Riskli" if last_egfr < 60 else "Normal", delta_color="normal" if last_egfr > 60 else "inverse")
else:
    col1.metric("Son eGFR", "Hesaplanamadı")

# Son HbA1c
hba1c_data = p_df[p_df['lab_name'] == 'HbA1c']
if not hba1c_data.empty:
    val = hba1c_data['valuenum'].iloc[-1]
    col2.metric("Son HbA1c", f"%{val}", delta="Yüksek" if val > 7 else "İyi", delta_color="inverse")
else:
    col2.metric("Son HbA1c", "-")

col3.metric("Yaş / Cinsiyet", f"{int(p_age)} / {p_gender}")
col4.metric("Toplam Veri Noktası", len(p_df))

st.divider()

# -----------------------------------------------------------------------------
# 5. GRAFİKLER (ZAMAN SERİLERİ)
# -----------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📈 Korelasyon Analizi", "🧪 Geniş Panel", "📋 Ham Veri"])

with tab1:
    st.subheader("Korelasyon: Glikoz (Şeker) vs. eGFR (Böbrek)")
    st.caption("Sol Eksen (Yeşil): Böbrek Fonksiyonu | Sağ Eksen (Mavi): Kan Şekeri")
    
    egfr_data = p_df.dropna(subset=['eGFR'])
    glu_data = p_df[p_df['lab_name'] == 'Glucose']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not egfr_data.empty:
        fig.add_trace(
            go.Scatter(x=egfr_data['charttime'], y=egfr_data['eGFR'], name="eGFR (Böbrek)",
                       mode='lines+markers', line=dict(color='#2ca02c', width=3), marker=dict(size=8)),
            secondary_y=False)
        fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Kritik Sınır", secondary_y=False)

    if not glu_data.empty:
        fig.add_trace(
            go.Scatter(x=glu_data['charttime'], y=glu_data['valuenum'], name="Glikoz (Şeker)",
                       mode='lines+markers', line=dict(color='#1f77b4', width=2, dash='dot'), marker=dict(size=6, opacity=0.6)),
            secondary_y=True)

    fig.update_layout(title_text="Zaman İçinde Şeker ve Böbrek İlişkisi", height=500, hovermode="x unified")
    fig.update_yaxes(title_text="<b>eGFR</b> (ml/min)", title_font=dict(color="#2ca02c"), secondary_y=False)
    fig.update_yaxes(title_text="<b>Glikoz</b> (mg/dL)", title_font=dict(color="#1f77b4"), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    if not glu_data.empty and not egfr_data.empty:
        g_mean = glu_data['valuenum'].mean()
        g_std = glu_data['valuenum'].std()
        g_cv = g_std / g_mean if g_mean > 0 else 0
        st.info(f"**Analiz:** Hastanın Glikoz CV değeri **{g_cv:.2f}** seviyesindedir.")

with tab2:
    st.markdown("### 🔬 Detaylı Biyobelirteçler")
    available_labs = p_df['lab_name'].unique()
    default_candidates = ['Potassium', 'Hemoglobin', 'Albumin', 'BUN', 'Cholesterol_Total']
    default_labs = [l for l in default_candidates if l in available_labs]
    
    selected_labs = st.multiselect("Grafiğe Eklenecek Testler:", available_labs, default=default_labs)
    
    if selected_labs:
        subset = p_df[p_df['lab_name'].isin(selected_labs)]
        fig_multi = px.scatter(subset, x='charttime', y='valuenum', color='lab_name', 
                               title="Karşılaştırmalı Laboratuvar Değerleri", facet_col='lab_name', facet_col_wrap=2, height=600)
        fig_multi.update_yaxes(matches=None) 
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Lütfen yukarıdan bir test seçin.")

with tab3:
    st.subheader("📋 Hastaya Ait Ham Veri")
    try:
        pivot_df = p_df.pivot_table(index='charttime', columns='lab_name', values='valuenum', aggfunc='first')
        st.dataframe(pivot_df.sort_index(ascending=False), use_container_width=True)
    except Exception as e:
        st.dataframe(p_df)

# -----------------------------------------------------------------------------
# 6. KOHORT ANALİZİ
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🌍 Büyük Resim: Tüm Kohort Analizi")

if st.checkbox("Tüm Veri Setini Analiz Et"):
    with st.spinner("Hastaların verileri özetleniyor..."):
        g_stats = df[df['lab_name'] == 'Glucose'].groupby('subject_id')['valuenum'].agg(['mean', 'std']).reset_index()
        g_stats['glucose_cv'] = g_stats['std'] / g_stats['mean']
        k_stats = df.groupby('subject_id')['eGFR'].min().reset_index().rename(columns={'eGFR': 'min_egfr'})
        cohort_summary = pd.merge(g_stats, k_stats, on='subject_id')
        cohort_summary['Risk_Grubu'] = cohort_summary.apply(lambda x: 'Yüksek Risk (eGFR<60)' if x['min_egfr'] < 60 else 'Düşük Risk', axis=1)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_scat = px.scatter(cohort_summary, x='glucose_cv', y='min_egfr', color='Risk_Grubu',
                                  color_discrete_map={'Yüksek Risk (eGFR<60)': 'red', 'Düşük Risk': 'green'},
                                  hover_data=['subject_id'], title="CV (Dalgalanma) Arttıkça eGFR Düşüyor mu?",
                                  labels={'glucose_cv': 'Glikoz CV (Dalgalanma)', 'min_egfr': 'Minimum eGFR'})
            fig_scat.add_hline(y=60, line_dash="dash", line_color="gray")
            fig_scat.add_vline(x=0.35, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scat, use_container_width=True)
        with c2:
            st.metric("Riskli Hasta Sayısı", len(cohort_summary[cohort_summary['min_egfr'] < 60]))
            st.metric("Toplam Analiz Edilen", len(cohort_summary))