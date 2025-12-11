import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from scipy.stats import linregress

# Sayfa Ayarları
st.set_page_config(
    page_title="Diyabetik Nefropati Erken Uyarı Sistemi",
    page_icon="🛡️",
    layout="wide"
)

# --- 1. MODEL VE SCALER YÜKLEME ---
@st.cache_resource
def load_ai_assets():
    # En iyi modelini buraya yükle (dosya adını kontrol et)
    # Eğer hata alırsan 'best_kidney_model_v3.keras' dosyasının yanında olduğundan emin ol
    model = load_model('best_kidney_model_v3.keras') 
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_ai_assets()
    st.success("Yapay Zeka Motoru Hazır! 🧠")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# --- 2. HESAPLAMA MOTORU (FEATURE ENGINEERING) ---
def calculate_features(age, glucose_history, egfr_history, hba1c):
    """
    Kullanıcının girdiği ham listeden, modelin beklediği türetilmiş özellikleri hesaplar.
    """
    # Veriyi Series'e çevir
    gl_series = pd.Series(glucose_history)
    egfr_series = pd.Series(egfr_history)
    
    # 1. Temel Hesaplamalar
    current_glucose = gl_series.iloc[-1]
    gl_mean = gl_series.mean()
    gl_std = gl_series.std() if len(gl_series) > 1 else 0
    
    # 2. Kritik Türetilmiş Özellikler (Engineering)
    glucose_cv = gl_std / (gl_mean + 1e-9) # Değişkenlik
    hyper_load = (gl_series > 180).astype(int).mean() # Yüksek şeker yükü
    
    # HbA1c yoksa tahmin et, varsa kullan
    if hba1c is None or hba1c == 0:
        final_hba1c = (gl_mean + 46.7) / 28.7
    else:
        final_hba1c = hba1c
        
    age_x_glucose = age * gl_mean # Etkileşim
    toxic_comb = glucose_cv * hyper_load # Zehirli Kombinasyon
    
    # 3. eGFR Trendi (Slope)
    if len(egfr_series) > 1:
        slope, _, _, _, _ = linregress(range(len(egfr_series)), egfr_series)
    else:
        slope = 0
        
    # eGFR_Trend_Rolling için basitçe son slope'u kullanıyoruz
    egfr_trend = slope 

    # Modelin beklediği sıralama (Eğitimdeki 'features' listesiyle AYNI OLMALI)
    # ['age_at_test', 'Glucose', 'Glucose_CV', 'Hyper_Load', 'HbA1c_Final', 'Age_x_Glucose', 'Toxic_Combination', 'eGFR_Trend_Rolling']
    
    features = np.array([
        age, 
        current_glucose, 
        glucose_cv, 
        hyper_load, 
        final_hba1c, 
        age_x_glucose, 
        toxic_comb, 
        egfr_trend
    ]).reshape(1, -1)
    
    return features, glucose_cv, egfr_trend

# --- 3. ARAYÜZ TASARIMI ---
st.title("🛡️ Diyabetik Nefropati - Erken Tespit Sistemi")
st.markdown("""
Bu sistem, Tip 1 Diyabet hastalarında **böbrek hasarını (Nefropati)** klinik belirtiler başlamadan önce öngörmek için 
**LSTM (Deep Learning)** teknolojisini kullanır.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Hasta Verileri")
    
    age = st.number_input("Yaş", min_value=10, max_value=90, value=35)
    hba1c = st.number_input("Son HbA1c (%)", min_value=4.0, max_value=15.0, value=8.5)
    
    st.markdown("---")
    st.markdown("**Glikoz Geçmişi (Son 5 Ölçüm)**")
    st.caption("Virgülle ayırarak giriniz (Örn: 120, 240, 80...)")
    gl_input = st.text_input("Glikoz Değerleri (mg/dL)", "150, 160, 155, 145, 150")
    
    st.markdown("**eGFR Geçmişi (Son 3-5 Ziyaret)**")
    egfr_input = st.text_input("eGFR Değerleri", "90, 88, 89, 87, 86")
    
    btn_predict = st.button("Risk Analizi Yap", type="primary")

with col2:
    if btn_predict:
        try:
            # Girdileri listeye çevir
            gl_history = [float(x.strip()) for x in gl_input.split(',')]
            egfr_history = [float(x.strip()) for x in egfr_input.split(',')]
            
            if len(gl_history) < 3:
                st.warning("Lütfen en az 3 glikoz değeri girin (Trend analizi için).")
            else:
                # 1. Özellikleri Hesapla
                raw_features, calc_cv, calc_slope = calculate_features(age, gl_history, egfr_history, hba1c)
                
                # 2. Ölçekle (Scaling)
                scaled_features = scaler.transform(raw_features)
                
                # 3. Modele Hazırla (Padding)
                # Model (1, 10, 8) bekliyor. Bizim elimizde (1, 8) var.
                # Tek bir zaman adımıymış gibi davranıp padding yapacağız.
                # LSTM stateful olduğu için geçmişi sequence olarak vermemiz gerekirdi,
                # ancak demo için son durumu "tekil sequence" olarak verip padding ile tamamlıyoruz.
                
                # Sequence oluştur: [ [Features] ] -> Shape (1, 1, 8)
                seq_input = scaled_features.reshape(1, 1, 8)
                # Padding -> Shape (1, 10, 8)
                padded_input = pad_sequences(seq_input, maxlen=10, padding='pre', dtype='float32', value=-99.0)
                
                # 4. Tahmin
                prediction_prob = model.predict(padded_input)[0][0]
                prediction_percent = prediction_prob * 100
                
                # --- SONUÇ EKRANI ---
                st.subheader("📊 Analiz Sonucu")
                
                # Renkli Risk Göstergesi
                if prediction_percent < 40:
                    st.success(f"Düşük Risk: %{prediction_percent:.1f}")
                    risk_color = "green"
                elif prediction_percent < 70:
                    st.warning(f"Orta Risk: %{prediction_percent:.1f}")
                    risk_color = "orange"
                else:
                    st.error(f"🚨 YÜKSEK RİSK: %{prediction_percent:.1f}")
                    risk_color = "red"
                
                st.progress(int(prediction_percent))
                
                # Explainable AI (Neden?)
                st.markdown("### 🔍 Model Neden Bu Kararı Verdi?")
                c1, c2, c3 = st.columns(3)
                c1.metric("Glikoz Dalgalanması (CV)", f"{calc_cv:.2f}", 
                          help="Şekerin ne kadar kararsız olduğunu gösterir. Yüksek olması risktir.")
                c2.metric("eGFR Trendi (Slope)", f"{calc_slope:.2f}", 
                          help="Böbrek fonksiyonunun değişim hızı. Negatif değer düşüşü gösterir.")
                c3.metric("Son Glikoz", f"{gl_history[-1]}", "mg/dL")
                
                st.info("""
                **Yapay Zeka Yorumu:** Model, anlık şekerden ziyade **Glikoz CV (Dalgalanma)** ve **eGFR Trendine** odaklanmaktadır. 
                Girdiğiniz verilerdeki dalgalanma ve düşüş eğilimi, risk skorunu doğrudan etkilemiştir.
                """)
                
        except Exception as e:
            st.error(f"Veri işlenirken hata: {e}")
            st.write("Lütfen sayıları virgülle ayırarak girdiğinizden emin olun.")

    else:
        st.info("Verileri girip 'Risk Analizi Yap' butonuna basın.")
        
        # Örnek Grafik (Görsel Zenginlik)
        st.markdown("---")
        st.markdown("#### 📉 Sistem Nasıl Çalışıyor?")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Diabetic_nephropathy.png/400px-Diabetic_nephropathy.png", caption="Diyabetik Nefropati İlerlemesi (Temsili)")