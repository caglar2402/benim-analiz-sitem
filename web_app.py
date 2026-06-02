import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Sayfa Genişlik ve Tema Ayarları (Siber Karanlık Tema Düzeni)
st.set_page_config(page_title="CST Analitik Otomatik Web v7.0", layout="wide")

# Otomatik Excel Dosya Adı Tanımı
EXCEL_DOSYA_ADI = "veri 2.xlsx"

# CSS ile Arka Planı ve Görsel Kaliteyi Premium Karanlık Yapıyoruz
st.markdown("""
    <style>
    .stApp { background-color: #0A0B0D; color: #E2E8F0; }
    h1, h2, h3 { color: #00ffcc !important; }
    </style>
""", unsafe_allow_html=True)

# 🧠 YAPAY ZEKA MODELİNİ ÖNBELLEĞE ALMA (Excel'i bir kere okur, siteyi her tıklamada dondurmaz)
@st.cache_resource
def yapay_zekayi_otomatik_egit(dosya_yolu):
    if not os.path.exists(dosya_yolu):
        return None
    try:
        df = pd.read_excel(dosya_yolu)
        df.columns = df.columns.astype(str).str.replace(r'\xa0', '', regex=True).str.strip()
        
        # Kolon Eşitleme Filtresi
        kolon_haritasi = {}
        for col in df.columns:
            c_clean = col.upper().replace(' ', '').replace('İ', 'I').replace('_', '').replace('-', '')
            if c_clean == 'MS': map_name = 'MS'
            elif c_clean in ['EVGOL', 'EVSAHIBIGOL', 'EV']: kolon_haritasi[col] = 'Ev_Gol'
            elif c_clean in ['DEPGOL', 'MISAFIRGOL', 'DEPLASMANGOL', 'DEP']: kolon_haritasi[col] = 'Dep_Gol'
            elif ('IY' in c_clean or 'ILKYARI' in c_clean) and ('EV' in c_clean or 'SAHIBI' in c_clean): kolon_haritasi[col] = 'IY_Ev_Gol'
            elif ('IY' in c_clean or 'ILKYARI' in c_clean) and ('DEP' in c_clean or 'MISAFIR' in c_clean): kolon_haritasi[col] = 'IY_Dep_Gol'
            elif c_clean in ['MS1', 'MACSONUCU1']: kolon_haritasi[col] = 'MS1'
            elif c_clean in ['MSX', 'MS0', 'MACSONUCUX', 'MACSONUCU0']: kolon_haritasi[col] = 'MSX'
            elif c_clean in ['MS2', 'MACSONUCU2']: kolon_haritasi[col] = 'MS2'
            elif c_clean in ['VAR', 'KGVAR']: kolon_haritasi[col] = 'Var'
            elif c_clean in ['YOK', 'KGYOK']: kolon_haritasi[col] = 'Yok'
            elif 'EVSAHIBI' in c_clean: kolon_haritasi[col] = 'Ev Sahibi'
            elif 'MISAFIR' in c_clean or 'DEPLASMAN' in c_clean: kolon_haritasi[col] = 'Misafir'
            elif 'LIG' in c_clean: kolon_haritasi[col] = 'Lig'
        
        df.rename(columns=kolon_haritasi, inplace=True)
        
        oran_kolonlari = ['MS1', 'MSX', 'MS2', 'Var', 'Yok']
        for col in oran_kolonlari:
            df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=oran_kolonlari)
        
        df['Kar_Marji'] = (1 / df['MS1']) + (1 / df['MSX']) + (1 / df['MS2']) - 1
        df['Olasilik_Ev'] = (1 / df['MS1']) / (df['Kar_Marji'] + 1)
        df['Olasilik_Dep'] = (1 / df['MS2']) / (df['Kar_Marji'] + 1)
        df['Favori_Makas'] = (df['MS1'] - df['MS2']).abs()
        df['Gol_Potansiyeli'] = df['Var'] / (df['Yok'] + 1e-5)
        df['Log_MS1'] = np.log1p(df['MS1'])
        
        X_kolonlari = ['MS1', 'MSX', 'MS2', 'Var', 'Yok', 'Favori_Makas', 'Gol_Potansiyeli', 'Olasilik_Ev', 'Olasilik_Dep', 'Log_MS1']
        
        df['Ev_Gol'] = pd.to_numeric(df['Ev_Gol'], errors='coerce').fillna(0).astype(int)
        df['Dep_Gol'] = pd.to_numeric(df['Dep_Gol'], errors='coerce').fillna(0).astype(int)
        
        if 'IY_Ev_Gol' not in df.columns:
            df['IY_Ev_Gol'] = (df['Ev_Gol'] * 0.45).apply(lambda x: np.random.poisson(max(0.1, x))).astype(int)
            df['IY_Dep_Gol'] = (df['Dep_Gol'] * 0.40).apply(lambda x: np.random.poisson(max(0.1, x))).astype(int)
            
        X = df[X_kolonlari].values
        y_ev = df['Ev_Gol'].values
        y_dep = df['Dep_Gol'].values
        y_iy_ev = df['IY_Ev_Gol'].values
        y_iy_dep = df['IY_Dep_Gol'].values
        
        y_sonuc = df.apply(lambda r: 0 if r['Ev_Gol'] > r['Dep_Gol'] else (1 if r['Ev_Gol'] == r['Dep_Gol'] else 2), axis=1).values
        
        model_ev = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1).fit(X, y_ev)
        model_dep = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1).fit(X, y_dep)
        model_iy_ev = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X, y_iy_ev)
        model_iy_dep = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X, y_iy_dep)
        
        ham_rf = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
        model_prob = CalibratedClassifierCV(estimator=ham_rf, method='sigmoid', cv=3).fit(X, y_sonuc)
        
        return model_ev, model_dep, model_iy_ev, model_iy_dep, model_prob, df
    except Exception as e:
        st.error(f"Model eğitim hatası: {e}")
        return None

def poisson_monte_carlo_skor(lam_ev, lam_dep):
    lam_ev = max(0.05, lam_ev); lam_dep = max(0.05, lam_dep)
    skor_havuzu, olasilik_havuzu = [], []
    for i in range(5):
        for j in range(5):
            p_ev = (math.pow(lam_ev, i) * math.exp(-lam_ev)) / math.factorial(i)
            p_dep = (math.pow(lam_dep, j) * math.exp(-lam_dep)) / math.factorial(j)
            skor_havuzu.append((i, j))
            olasilik_havuzu.append(p_ev * p_dep)
    olasilik_havuzu = np.array(olasilik_havuzu)
    olasilik_havuzu /= olasilik_havuzu.sum()
    return skor_havuzu[np.random.choice(len(skor_havuzu), p=olasilik_havuzu)]

# --- WEB ARAYÜZÜ ---
st.title("🎯 CST ANALİTİK TERMİNAL v7.0 - AUTOMATIC CLOUD")

# Arka Planda Excel'i Otomatik Kontrol Et ve Eğit
git_modeller = yapay_zekayi_otomatik_egit(EXCEL_DOSYA_ADI)

if git_modeller is None:
    st.error(f"❌ KRİTİK HATA: Bulut sunucusunda '{EXCEL_DOSYA_ADI}' dosyası bulunamadı! Lütfen Excel dosyasını kod ile aynı klasöre yükleyin.")
else:
    model_ev, model_dep, model_iy_ev, model_iy_dep, model_prob, df = git_modeller
    
    # Sol Giriş Paneli
    with st.sidebar:
        st.header("⚡ MAÇ ORANLARINI GİRİN")
        mac_adi = st.text_input("Maç Adı / Kod", "Real Madrid - Barcelona")
        m1_ham = st.number_input("MS 1 Oranı", min_value=1.01, value=1.80, step=0.05)
        mx_ham = st.number_input("MS X Oranı", min_value=1.01, value=3.40, step=0.05)
        m2_ham = st.number_input("MS 2 Oranı", min_value=1.01, value=3.60, step=0.05)
        v = st.number_input("KG VAR Oranı", min_value=1.01, value=1.55, step=0.05)
        yk = st.number_input("KG YOK Oranı", min_value=1.01, value=1.90, step=0.05)
        kasa = st.number_input("Toplam Kasa (TL)", min_value=100, value=10000)
        profil = st.selectbox("Risk Profili", ["Muhafazakar", "Dengeli", "Agresif"], index=1)
        stres = st.slider("Oran Stres Testi (%)", -25, 25, 0)
        
    # HESAPLAMA MOTORU
    stres_yuzde = stres / 100.0
    m1 = max(1.01, m1_ham * (1 + stres_yuzde))
    mx = max(1.01, mx_ham * (1 + (stres_yuzde * 0.5)))
    m2 = max(1.01, m2_ham * (1 + stres_yuzde))
    
    k_marji = (1 / m1) + (1 / mx) + (1 / m2) - 1
    imp_p1 = (1 / m1) / (k_marji + 1)
    imp_p2 = (1 / m2) / (k_marji + 1)
    f_makas = abs(m1 - m2)
    g_potansiyel = v / (yk + 1e-5)
    log_m1 = np.log1p(m1)
    
    yeni_mac = np.array([[m1, mx, m2, v, yk, f_makas, g_potansiyel, imp_p1, imp_p2, log_m1]])
    
    # Monte Carlo Skorları
    g_ev = model_ev.predict(yeni_mac)[0]
    g_dep = model_dep.predict(yeni_mac)[0]
    skor_ev, skor_dep = poisson_monte_carlo_skor(g_ev, g_dep)
    
    g_iy_ev = model_iy_ev.predict(yeni_mac)[0]
    g_iy_dep = model_iy_dep.predict(yeni_mac)[0]
    skor_iy_ev, skor_iy_dep = poisson_monte_carlo_skor(g_iy_ev, g_iy_dep)
    
    olasiliklar = model_prob.predict_proba(yeni_mac)[0]
    p1, px, p2 = olasiliklar[0], olasiliklar[1], olasiliklar[2]
    
    # --- SAĞ PANEL GÖRSEL MATRİS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🔮 İLKYARI TAHMİNİ", value=f"{skor_iy_ev} - {skor_iy_dep}")
        st.caption(f"İY Gol Beklentisi: {g_iy_ev:.2f} / {g_iy_dep:.2f}")
    with col2:
        st.metric(label="🔮 MAÇ SONUCU TAHMİNİ", value=f"{skor_ev} - {skor_dep}")
        st.caption(f"MS Gol Beklentisi: {g_ev:.2f} / {g_dep:.2f}")
    with col3:
        nihai_guven = min(94.5, (max(p1, px, p2) * 80) + 10)
        st.metric(label="🛡️ SİSTEM GÜVEN ENDEKSİ", value=f"%{nihai_guven:.1f}")
        
    st.markdown("---")
    
    # Risk ve Kasa Çubukları
    st.subheader("📊 Olasılıklar ve Yatırım Önerileri")
    risk_carpani = 0.4 if profil == "Muhafazakar" else (1.8 if profil == "Agresif" else 1.0)
    
    def bar_ciz(baslik, ihtimal, oran):
        bulten_ihtimali = 1 / oran
        fark = ihtimal - bulten_ihtimali
        durum = "🔥 DEĞERLİ SİNYAL" if ihtimal > (bulten_ihtimali * 1.08) else "Sınır Altı"
        kasa_orani = min(0.12, max(0.00, fark * 1.4)) * risk_carpani if durum == "🔥 DEĞERLİ SİNYAL" else 0.0
        st.write(f"**{baslik}**: %{ihtimal*100:.1f} | **{durum}** | Öneri: **{kasa * kasa_orani:.0f} TL** (%{kasa_orani*100:.1f})")
        st.progress(float(ihtimal))
        
    bar_ciz("Ev Sahibi (1)", p1, m1)
    bar_ciz("Beraberlik (X)", px, mx)
    bar_ciz("Deplasman (2)", p2, m2)
    
    st.markdown("---")
    
    # Grafik ve Geçmiş İkizler
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.subheader("📈 Gol Dağılım Eğrisi")
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0A0B0D')
        ax.set_facecolor('#11141A')
        goller = np.arange(5)
        ev_p_list = [(math.pow(g_ev, i) * math.exp(-g_ev)) / math.factorial(i) for i in goller]
        dep_p_list = [(math.pow(g_dep, i) * math.exp(-g_dep)) / math.factorial(i) for i in goller]
        ax.plot(goller, ev_p_list, color='#00ffcc', marker='o', linewidth=2, label='Ev')
        ax.plot(goller, dep_p_list, color='#ff4757', marker='s', linewidth=2, label='Dep')
        ax.legend(facecolor='#11141A', labelcolor='white')
        ax.tick_params(colors='white', labelsize=8)
        st.pyplot(fig)
        
    with g_col2:
        st.subheader("🔮 Geçmiş Maç İkizleri")
        gecmis_matris = df[['MS1', 'MSX', 'MS2', 'Var', 'Yok']].values
        mevcut_oranlar = yeni_mac[0][:5]
        mesafeler = np.linalg.norm(gecmis_matris - mevcut_oranlar, axis=1)
        en_yakin_3 = np.argsort(mesafeler)[:3]
        
        for i, idx in enumerate(en_yakin_3, 1):
            satir = df.iloc[idx]
            iy_ev = int(satir['IY_Ev_Gol']) if 'IY_Ev_Gol' in satir else 0
            iy_dep = int(satir['IY_Dep_Gol']) if 'IY_Dep_Gol' in satir else 0
            st.info(f"👉 **İkiz {i}**: {satir['Ev Sahibi']} {int(satir['Ev_Gol'])}-{int(satir['Dep_Gol'])} {satir['Misafir']} (IY: {iy_ev}-{iy_dep}) | [{satir['MS1']:.2f} - {satir['MSX']:.2f} - {satir['MS2']:.2f}]")