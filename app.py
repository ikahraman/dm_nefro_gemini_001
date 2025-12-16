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
    page_title="Diyabetik Nefropati AI", 
    page_icon="🛡️", 
    layout="wide"
)

# --- 1. MODEL VE SCALER YÜKLEME ---
import os
import streamlit as st
# Diğer importlar... (joblib, tensorflow.keras.models vs.)

@st.cache_resource
def load_assets():
    # ÖNCE DOSYALAR ORADA MI DİYE KONTROL EDELİM
    st.write("📂 Çalışma Dizini:", os.getcwd())
    st.write("📄 Dosya Listesi:", os.listdir('.'))

    # train_and_save.py ile oluşturduğun dosyalar
    try:
        # Önce kütüphanelerin yüklü olup olmadığını test edelim
        import tensorflow as tf
        import joblib
        
        # Dosyaları yüklemeyi dene
        model = tf.keras.models.load_model('model_final.keras') 
        scaler = joblib.load('scaler_final.pkl')
        return model, scaler

    except Exception as e:
        # HATAYI YUTMA, EKRANA YAZ!
        st.error(f"💥 HATA DETAYI: {e}")
        return None, None

model, scaler = load_assets()

# Eğer model yüklenemediyse dur
if model is None:
    st.stop()

# --- 2. HESAPLAMA MOTORU (FEATURE ENGINEERING) ---
def calculate_features(age, glucose_history, egfr_history, hba1c):
    """
    Kullanıcıdan alınan ham listeleri, modelin anladığı matematiksel özelliklere çevirir.
    """
    gl_series = pd.Series(glucose_history)
    egfr_series = pd.Series(egfr_history)
    
    # 1. Temel İstatistikler
    current_glucose = gl_series.iloc[-1]
    gl_mean = gl_series.mean()
    gl_std = gl_series.std() if len(gl_series) > 1 else 0
    
    # 2. Türetilmiş Biyobelirteçler
    # CV: Standart Sapma / Ortalama (Sıfıra bölünmeyi önlemek için +1e-9)
    glucose_cv = gl_std / (gl_mean + 1e-9)
    
    # Hyper Load: 180 üzerindeki ölçümlerin oranı
    hyper_load = (gl_series > 180).astype(int).mean()
    
    # HbA1c Dolgu (Varsa kullan, yoksa tahmin et)
    if hba1c is None or hba1c == 0:
        final_hba1c = (gl_mean + 46.7) / 28.7
    else:
        final_hba1c = hba1c
        
    # Etkileşimler
    age_x_glucose = age * gl_mean
    toxic_comb = glucose_cv * hyper_load
    
    # 3. Trend Analizi (Slope)
    if len(egfr_series) > 1:
        # X ekseni: Zaman (0, 1, 2...), Y ekseni: eGFR
        slope, _, _, _, _ = linregress(range(len(egfr_series)), egfr_series)
    else:
        slope = 0.0
    egfr_trend = slope 

    # --- ÖNEMLİ: EĞİTİMDEKİ SIRALAMA İLE AYNISI ---
    # ['age_at_test', 'Glucose', 'Glucose_CV', 'Hyper_Load', 'HbA1c_Final', 'Age_x_Glucose', 'Toxic_Combination', 'eGFR_Trend_Rolling']
    
    feature_vector = np.array([
        age, 
        current_glucose, 
        glucose_cv, 
        hyper_load, 
        final_hba1c, 
        age_x_glucose, 
        toxic_comb, 
        egfr_trend
    ]).reshape(1, -1)
    
    return feature_vector, glucose_cv, egfr_trend

# --- 3. ARAYÜZ TASARIMI ---
st.title("🛡️ Diyabetik Nefropati - Erken Tespit Sistemi")
st.markdown("""
Bu sistem, **Tip 1 Diyabet** hastalarında böbrek hasarını, klinik belirtiler başlamadan önce öngörmek için 
**LSTM (Derin Öğrenme)** teknolojisini kullanır. Anlık şekerden ziyade **dalgalanmaya (CV)** ve **düşüş hızına (Trend)** odaklanır.
""")

# Yan Panel (Geliştirici Modu)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    debug_mode = st.checkbox("Mühendislik Detaylarını Göster", value=False)
    st.info("Bu mod, modelin arka planda gördüğü sayısal matrisleri gösterir.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Hasta Verileri")
    
    age = st.number_input("Yaş", min_value=10, max_value=90, value=55)
    hba1c = st.number_input("Son HbA1c (%)", min_value=4.0, max_value=15.0, value=8.5, step=0.1)
    
    st.markdown("---")
    st.caption("Verileri virgülle ayırarak giriniz (Örn: 120, 240, 80...)")
    
    # Varsayılan değerler: Yüksek Dalgalanma Senaryosu
    gl_input = st.text_input("Glukoz Geçmişi (Son 5 Ölçüm)", "70, 300, 80, 250, 90")
    
    # Varsayılan değerler: Düşen Trend Senaryosu
    egfr_input = st.text_input("eGFR Geçmişi (Son 3-5 Ziyaret)", "90, 80, 70")
    
    btn_predict = st.button("Risk Analizi Yap", type="primary", use_container_width=True)

with col2:
    if btn_predict:
        try:
            # Girdileri listeye çevir
            gl_history = [float(x.strip()) for x in gl_input.split(',')]
            egfr_history = [float(x.strip()) for x in egfr_input.split(',')]
            
            if len(gl_history) < 3:
                st.warning("⚠️ Doğru trend analizi için en az 3 glukoz değeri girin.")
            else:
                # ---------------------------------------------------------
                # ADIM A: Özellik Hesaplama
                # ---------------------------------------------------------
                raw_features, calc_cv, calc_slope = calculate_features(age, gl_history, egfr_history, hba1c)
                
                # ---------------------------------------------------------
                # ADIM B: Ölçekleme (Standard Scaler)
                # ---------------------------------------------------------
                scaled_features = scaler.transform(raw_features)
                
                # ---------------------------------------------------------
                # ADIM C: Replikasyon (Statik Film Tekniği)
                # Tek bir satırı 10 kez kopyalayarak (1, 10, 8) boyutuna getiriyoruz.
                # Böylece LSTM, "Padding" (boşluk) yüzünden kararsız kalmıyor.
                # ---------------------------------------------------------
                
                # (1, 8) -> (1, 1, 8)
                seq_input = scaled_features.reshape(1, 1, 8)
                
                # (1, 1, 8) -> (1, 10, 8)
                model_input = np.tile(seq_input, (1, 10, 1))
                
                # ---------------------------------------------------------
                # ADIM D: Tahmin
                # ---------------------------------------------------------
                prediction_prob = model.predict(model_input)[0][0]
                prediction_percent = prediction_prob * 100
                
                # ---------------------------------------------------------
                # SONUÇ GÖSTERİMİ
                # ---------------------------------------------------------
                st.subheader("📊 Analiz Sonucu")
                
                if prediction_percent < 40:
                    status_color = "green"
                    status_text = "DÜŞÜK RİSK"
                    alert_type = st.success
                elif prediction_percent < 70:
                    status_color = "orange"
                    status_text = "ORTA RİSK"
                    alert_type = st.warning
                else:
                    status_color = "red"
                    status_text = "YÜKSEK RİSK"
                    alert_type = st.error
                
                alert_type(f"**{status_text}:** %{prediction_percent:.1f}")
                st.progress(int(prediction_percent))
                
                # Explainable AI (Açıklama)
                st.markdown("### 🔍 Model Neden Bu Kararı Verdi?")
                c1, c2, c3 = st.columns(3)
                
                c1.metric(
                    label="Glukoz Dalgalanması (CV)", 
                    value=f"{calc_cv:.2f}",
                    delta="Yüksek Risk" if calc_cv > 0.3 else "Normal",
                    delta_color="inverse",
                    help="0.30'un üzerindeki değerler yüksek kararsızlık (risk) işaretidir."
                )
                
                c2.metric(
                    label="eGFR Trendi (Slope)", 
                    value=f"{calc_slope:.2f}",
                    delta="Düşüş Var" if calc_slope < -2 else "Stabil",
                    delta_color="normal" if calc_slope >= -2 else "inverse",
                    help="Negatif değerler böbrek fonksiyonundaki düşüş hızını gösterir."
                )
                
                c3.metric(label="Son Glukoz", value=f"{gl_history[-1]} mg/dL")
                
                # DEBUG EKRANI
                if debug_mode:
                    st.divider()
                    st.warning("🛠️ MÜHENDİSLİK DETAYLARI (DEBUG)")
                    st.write("**Ham Vektör (Raw Features):**", raw_features)
                    st.write("**Ölçeklenmiş Vektör (Scaled Input):**", scaled_features)
                    st.write(f"**Model Girdisi Şekli:** {model_input.shape}")
                    st.write(f"**Ham Tahmin Olasılığı:** {prediction_prob:.6f}")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
            st.info("Lütfen sayıları virgülle ayırarak (Örn: 100, 120) girdiğinizden emin olun.")

    else:
        # Başlangıçta boş durmasın diye bilgilendirme
        st.info("Sol taraftan verileri girip 'Risk Analizi Yap' butonuna basın.")
        st.markdown("#### 📉 Sistem Nasıl Çalışıyor?")
        st.markdown("""
        1. **Veri Girişi:** Hastanın son glukoz ve eGFR ölçümleri alınır.
        2. **Sinyal İşleme:** Sistem, glukozdaki **dalgalanmayı (CV)** ve böbrekteki **düşüş hızını (Slope)** hesaplar.
        3. **Yapay Zeka:** Bidirectional LSTM modeli, bu örüntüleri 450.000+ hasta verisiyle kıyaslar.
        4. **Sonuç:** Kişiye özel risk skoru üretilir.
        """)