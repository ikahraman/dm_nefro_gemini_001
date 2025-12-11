import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    # CSV dosyasını yükle
    try:
        df = pd.read_csv('type_1_and_renal_damage.csv')
    except FileNotFoundError:
        st.error("❌ 'type_1_and_renal_damage.csv' dosyası bulunamadı! Lütfen dosyayı proje klasörüne ekleyin.")
        st.stop()
        
    # Tarih formatını düzelt
    df['charttime'] = pd.to_datetime(df['charttime'])
    
    # Sayısal değerleri garantiye al (Metin karışmışsa NaN yap)
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    # Yaş Sütununu Otomatik Bul (SQL sorgusuna göre değişebilir)
    age_col = 'age'
    if 'real_age' in df.columns:
        age_col = 'real_age'
    elif 'age_at_test' in df.columns:
        age_col = 'age_at_test'
    
    # --- eGFR HESAPLAMA (CKD-EPI 2021) - GÜVENLİ VERSİYON ---
    def calc_egfr(row):
        # Sadece Kreatinin satırları için hesapla
        if row['lab_name'] == 'Creatinine':
            scr = row['valuenum']
            
            # [FIX] HATA KORUMASI: 0 veya negatif değerleri engelle
            if pd.isna(scr) or scr <= 0:
                return np.nan
            
            current_age = row[age_col]
            is_female = 1 if row['gender'] == 'F' else 0
            
            if is_female:
                k = 0.7; alpha = -0.329; factor = 1.018
            else:
                k = 0.9; alpha = -0.411; factor = 1
            
            # Matematiksel Formül (Try-Except ile ekstra koruma)
            try:
                egfr = 142 * (min(scr/k, 1)**alpha) * (max(scr/k, 1)**-1.200) * (0.9938**current_age) * factor
            except Exception:
                return np.nan
                
            return egfr
        return np.nan

    # Fonksiyonu uygula
    df['eGFR'] = df.apply(calc_egfr, axis=1)
    
    return df, age_col

# Veriyi Yükle
df, age_col_name = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR - HASTA SEÇİMİ VE ÖZET
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Hasta Gezgini")

# Hasta Listesi
unique_patients = df['subject_id'].unique()
st.sidebar.info(f"**Yüklü Hasta Sayısı:** {len(unique_patients)}")

# Arama Kutusu
selected_patient_id = st.sidebar.selectbox("Hasta ID Seçin:", unique_patients)

# Seçilen Hastanın Verisini Filtrele
p_df = df[df['subject_id'] == selected_patient_id].sort_values('charttime')

# Hastanın Demografik Bilgisi (İlk satırdan al)
if not p_df.empty:
    p_gender = p_df['gender'].iloc[0]
    p_age = p_df[age_col_name].iloc[0]
else:
    st.error("Bu hasta için veri bulunamadı.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. ANA EKRAN - HASTA DETAYLARI
# -----------------------------------------------------------------------------
st.title(f"🩺 Hasta Kartı: {selected_patient_id}")

# Üst Bilgi Kartları
col1, col2, col3, col4 = st.columns(4)

# Son eGFR'yi bul
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

tab1, tab2, tab3 = st.tabs(["📈 Ana Trendler", "🧪 Geniş Panel", "📋 Ham Veri"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Böbrek Fonksiyonu (eGFR)")
        if not egfr_history.empty:
            fig_egfr = px.line(egfr_history, x='charttime', y='eGFR', markers=True, 
                               title="eGFR Seyri (Böbrek Süzme Hızı)",
                               labels={'eGFR': 'eGFR (ml/min/1.73m²)'})
            # Kritik Eşik
            fig_egfr.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Yetmezlik Sınırı (60)")
            fig_egfr.update_traces(line_color='#2ca02c') # Yeşil
            st.plotly_chart(fig_egfr, use_container_width=True)
        else:
            st.warning("Bu hasta için yeterli Kreatinin verisi yok, eGFR hesaplanamadı.")
            
    with col_g2:
        st.subheader("Glisemik Kontrol (Glikoz)")
        glu_data = p_df[p_df['lab_name'] == 'Glucose']
        if not glu_data.empty:
            # Glikoz CV Hesabı
            g_mean = glu_data['valuenum'].mean()
            g_std = glu_data['valuenum'].std()
            g_cv = g_std / g_mean if g_mean > 0 else 0
            
            st.caption(f"**Glikoz Dalgalanması (CV): {g_cv:.2f}** (0.36 üzeri yüksek risktir)")
            
            fig_glu = px.line(glu_data, x='charttime', y='valuenum', markers=True,
                              title="Kan Şekeri Seyri",
                              labels={'valuenum': 'Glikoz (mg/dL)'})
            fig_glu.add_hline(y=180, line_dash="dash", line_color="orange", annotation_text="Hiperglisemi")
            fig_glu.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Hipoglisemi")
            st.plotly_chart(fig_glu, use_container_width=True)
        else:
            st.warning("Glikoz verisi yok.")

with tab2:
    st.markdown("### 🔬 Detaylı Biyobelirteçler")
    # Kullanıcının seçebileceği lab testleri
    available_labs = p_df['lab_name'].unique()
    # Varsayılan olarak ilginç olanları seç (Veride varsa)
    default_candidates = ['Potassium', 'Hemoglobin', 'Albumin', 'BUN', 'Cholesterol_Total']
    default_labs = [l for l in default_candidates if l in available_labs]
    
    selected_labs = st.multiselect("Grafiğe Eklenecek Testler:", available_labs, default=default_labs)
    
    if selected_labs:
        subset = p_df[p_df['lab_name'].isin(selected_labs)]
        fig_multi = px.scatter(subset, x='charttime', y='valuenum', color='lab_name', 
                               title="Karşılaştırmalı Laboratuvar Değerleri",
                               facet_col='lab_name', facet_col_wrap=2, # Her test için ayrı küçük grafik
                               height=600)
        fig_multi.update_yaxes(matches=None) # Y eksenlerini serbest bırak (Birimler farklı çünkü)
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Lütfen yukarıdan bir test seçin.")

with tab3:
    st.subheader("📋 Hastaya Ait Ham Veri")
    # Pivot Table ile okunabilir hale getir
    try:
        # Pivot yaparken duplicate charttime'ları yönetmek için 'first' veya 'mean' kullanılır
        pivot_df = p_df.pivot_table(index='charttime', columns='lab_name', values='valuenum', aggfunc='first')
        st.dataframe(pivot_df.sort_index(ascending=False), use_container_width=True)
    except Exception as e:
        st.dataframe(p_df) # Pivot hata verirse düz tablo göster

# -----------------------------------------------------------------------------
# 6. KOHORT ANALİZİ (TÜM HASTALAR) - ALT BÖLÜM
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🌍 Büyük Resim: Tüm Kohort Analizi")

if st.checkbox("Tüm Veri Setini Analiz Et (Biraz zaman alabilir)"):
    
    with st.spinner("Hastaların verileri özetleniyor..."):
        # Her hasta için özet istatistik çıkaralım
        
        # 1. Glikoz İstatistikleri
        g_stats = df[df['lab_name'] == 'Glucose'].groupby('subject_id')['valuenum'].agg(['mean', 'std']).reset_index()
        g_stats['glucose_cv'] = g_stats['std'] / g_stats['mean']
        
        # 2. En Kötü eGFR (Minimum)
        # eGFR zaten hesaplanmıştı df['eGFR']
        k_stats = df.groupby('subject_id')['eGFR'].min().reset_index().rename(columns={'eGFR': 'min_egfr'})
        
        # 3. Birleştir
        cohort_summary = pd.merge(g_stats, k_stats, on='subject_id')
        
        # 4. Risk Etiketlemesi
        cohort_summary['Risk_Grubu'] = cohort_summary.apply(
            lambda x: 'Yüksek Risk (eGFR<60)' if x['min_egfr'] < 60 else 'Düşük Risk', axis=1
        )
        
        # GRAFİK
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Glikoz Kararsızlığı vs. En Kötü Böbrek Değeri")
            fig_scat = px.scatter(cohort_summary, x='glucose_cv', y='min_egfr',
                                  color='Risk_Grubu',
                                  color_discrete_map={'Yüksek Risk (eGFR<60)': 'red', 'Düşük Risk': 'green'},
                                  hover_data=['subject_id'],
                                  title="CV (Dalgalanma) Arttıkça eGFR Düşüyor mu?",
                                  labels={'glucose_cv': 'Glikoz CV (Dalgalanma)', 'min_egfr': 'Minimum eGFR'})
            
            # Kritik Bölgeyi İşaretle
            fig_scat.add_hline(y=60, line_dash="dash", line_color="gray")
            fig_scat.add_vline(x=0.35, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with c2:
            st.write("""
            **Grafik Yorumu:**
            * **X Ekseni:** Şekerin ne kadar dalgalı olduğu. Sağa gittikçe dalgalanma artar.
            * **Y Ekseni:** Böbrek sağlığı. Aşağı gittikçe böbrek kötüleşir.
            * **Hipotez:** Kırmızı noktaların (Hasta böbrekler) grafiğin sağ tarafında (Yüksek CV) yoğunlaşmasını bekliyoruz.
            """)
            riskli_sayi = len(cohort_summary[cohort_summary['min_egfr'] < 60])
            st.metric("Riskli Hasta Sayısı", riskli_sayi)
            st.metric("Toplam Analiz Edilen", len(cohort_summary))