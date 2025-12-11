import pandas as pd
import numpy as np

# 1. CSV Dosyasını Yükle (Google BigQuery'den indirdiğin dosya)
# Dosya adını indirdiğin dosya ile değiştirmeyi unutma
fn=r"Diyabetik Nefropati Tahmin.csv"
file_path = fn
df = pd.read_csv(file_path)

# Tarih formatını düzelt
df['charttime'] = pd.to_datetime(df['charttime'])

print(f"Toplam Veri Sayısı: {len(df)}")
print(df.head())

# 2. CKD-EPI (2021) Formülü ile eGFR Hesaplama
def calculate_egfr(row):
    # Sadece Kreatinin satırları için hesapla
    if row['lab_name'] != 'Creatinine':
        return np.nan
    
    scr = row['valuenum'] # Serum Kreatinin
    
    # --- HATA DÜZELTME (FIX) ---
    # Kreatinin 0 veya negatif olamaz (Tıbbi hata / Veri hatası)
    # Bu durumda hesaplama yapma, NaN dön.
    if scr <= 0:
        return np.nan
    # ---------------------------

    age = row['age_at_test']
    
    # Cinsiyet Kontrolü
    is_female = 1 if row['gender'] == 'F' else 0
    
    if is_female:
        k = 0.7
        alpha = -0.241
        factor = 1.012 
    else: # Male
        k = 0.9
        alpha = -0.302
        factor = 1
        
    # Formül
    egfr = 142 * (min(scr/k, 1)**alpha) * (max(scr/k, 1)**-1.200) * (0.9938**age) * factor
    return egfr
print("\neGFR hesaplanıyor...")
df['eGFR'] = df.apply(calculate_egfr, axis=1)

# 3. Veriyi Pivotla (Model Formatına Getir)
# Hedefimiz: Her hasta için, her tarihte tek bir satır olması.
# Satır: Hasta + Tarih | Sütunlar: Glikoz, Kreatinin, eGFR, vb.

df['date'] = df['charttime'].dt.date

# Pivot Table oluşturuyoruz
pivot_df = df.pivot_table(
    index=['subject_id', 'date', 'gender', 'age_at_test'], # Bu bilgileri koru
    columns='lab_name', 
    values='valuenum', 
    aggfunc='mean' # Aynı gün birden fazla varsa ortalamasını al
).reset_index()

# eGFR'yi ayrı hesaplamıştık, onu da pivot'a eklememiz lazım
# Çünkü eGFR bir 'lab_name' değil, hesaplanan bir sütundu.
egfr_values = df[df['lab_name'] == 'Creatinine'][['subject_id', 'date', 'eGFR']]
# Aynı gün birden fazla eGFR varsa ortalama
egfr_values = egfr_values.groupby(['subject_id', 'date']).mean().reset_index()

# Ana tablo ile birleştir
final_df = pd.merge(pivot_df, egfr_values, on=['subject_id', 'date'], how='left')

print("\n--- İŞLEM TAMAMLANDI ---")
print("Model için hazır veri örneği:")
print(final_df.head())

# İsterseniz sonucu kaydedin
# final_df.to_csv('hazir_diyabet_verisi.csv', index=False)
# --- DÜZELTİLMİŞ KOD BLOĞU (Bunu kopyalayıp eskisinin yerine yapıştırın) ---

# 1. EKSİK VERİLERİ DOLDURMA (Forward Fill)
print("Eksik veriler dolduruluyor (Forward Fill)...")
final_df = final_df.sort_values(['subject_id', 'date'])

# Gruplayarak doldur
# DİKKAT: ffill() sonucu subject_id sütununu düşürür.
final_df_filled = final_df.groupby('subject_id').ffill()

# KRİTİK DÜZELTME: Kaybolan 'subject_id' sütununu orijinal tablodan geri alıyoruz
final_df_filled['subject_id'] = final_df['subject_id']

# Hala baştaki satırlarda boşluk varsa (ilk ölçümden öncesi), onları atalım
# (Not: eGFR hesaplanamayan satırlar işimize yaramaz)
final_df_filled = final_df_filled.dropna(subset=['Glucose', 'Creatinine', 'eGFR'])

# 2. GÜRÜLTÜYÜ AZALTMA (Moving Average / Smoothing)
print("Veri düzleştiriliyor (Smoothing)...")

# eGFR için 3 dönemlik hareketli ortalama
final_df_filled['eGFR_Smooth'] = final_df_filled.groupby('subject_id')['eGFR'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# 3. HEDEF BELİRLEME (Target Engineering)
# Gelecekteki (3 ziyaret sonraki) durumu tahmin etmek
final_df_filled['Target_eGFR_Next_3_Visits'] = final_df_filled.groupby('subject_id')['eGFR_Smooth'].shift(-3)

# Binary Hedef: Gelecekte eGFR 60'ın altına düşecek mi? (Risk Sınıfı)
final_df_filled['Target_Risk_Class'] = (final_df_filled['Target_eGFR_Next_3_Visits'] < 60).astype(int)

# Geleceği olmayan son satırları temizle
model_data = final_df_filled.dropna(subset=['Target_eGFR_Next_3_Visits'])

print("\n--- MODEL İÇİN HAZIR VERİ ---")
# Sütunları kontrol ederek yazdıralım
cols_to_show = ['subject_id', 'date', 'eGFR', 'eGFR_Smooth', 'Target_Risk_Class']
print(model_data[cols_to_show].head(10))

# Kaydet
model_data.to_csv('ml_ready_diabetes_data.csv', index=False)