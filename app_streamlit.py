import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="NanoPredict · Green Nanoparticle RE% Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #071b2e !important;
    font-family: 'DM Sans', sans-serif;
    color: #cfe8f0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 40% at 20% 15%, #0a2e4a 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, #0f3d2e 0%, transparent 60%),
        #071b2e !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Water wave decoration ── */
.wave-header {
    position: relative;
    text-align: center;
    padding: 2.5rem 1rem 0.5rem;
    overflow: hidden;
}
.wave-header::before {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; right: 0;
    height: 30px;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 500 30' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 15 Q62 5 125 15 Q187 25 250 15 Q312 5 375 15 Q437 25 500 15 L500 30 L0 30Z' fill='rgba(93,202,165,0.06)'/%3E%3C/svg%3E") center/cover;
}

.eyebrow {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #5dcaa5;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    color: #e0f4f8;
    line-height: 1.05;
    margin-bottom: 0.4rem;
}
.hero-title span { color: #5dcaa5; }
.hero-sub {
    font-size: 0.85rem;
    font-weight: 300;
    color: #7aacbd;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
.wave-divider {
    margin: 1rem auto;
    width: 100%;
    max-width: 500px;
    height: 16px;
    opacity: 0.5;
}

/* ── Cards ── */
.card {
    background: rgba(7, 40, 65, 0.82);
    border: 1px solid rgba(93, 202, 165, 0.18);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 0.8rem;
}
.card-title {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5dcaa5;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #5dcaa5;
}
.card-title::after {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    border: 1px dashed #5dcaa5;
    opacity: 0.5;
}

/* ── Widget overrides ── */
.stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
    color: #4a7f96 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(93,202,165,0.2) !important;
    border-radius: 10px !important;
    color: #cfe8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #5dcaa5 !important;
    box-shadow: 0 0 0 2px rgba(93,202,165,0.1) !important;
}
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(93,202,165,0.2) !important;
    border-radius: 10px !important;
    color: #cfe8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #5dcaa5 !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #1d9e75, #5dcaa5) !important;
}
[data-testid="stThumbValue"] {
    background: #1d9e75 !important;
    color: #e0f4f8 !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 600 !important;
}
.stRadio > div {
    flex-direction: row !important;
    gap: 0.6rem;
}
.stRadio > div > label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(93,202,165,0.15) !important;
    border-radius: 9px !important;
    padding: 0.4rem 0.9rem !important;
    color: #7aacbd !important;
    font-size: 0.78rem !important;
    cursor: pointer;
    transition: all 0.2s;
}
.stRadio > div > label:has(input:checked) {
    background: rgba(93,202,165,0.14) !important;
    border-color: #5dcaa5 !important;
    color: #5dcaa5 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1d9e75, #5dcaa5) !important;
    color: #071b2e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.72rem !important;
    box-shadow: 0 4px 20px rgba(29,158,117,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(29,158,117,0.35) !important;
}

/* ── Result display ── */
.result-outer {
    background: rgba(5, 30, 52, 0.9);
    border: 1px solid rgba(93,202,165,0.28);
    border-radius: 18px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-outer::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 50%;
    transform: translateX(-50%);
    width: 220px; height: 70px;
    border-radius: 50%;
    background: rgba(29,158,117,0.08);
    pointer-events: none;
}
.res-context {
    font-size: 0.6rem;
    color: #2a5a70;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.res-label {
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5dcaa5;
    margin-bottom: 0.2rem;
}
.res-value {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 700;
    color: #e0f4f8;
    line-height: 1;
}
.res-unit {
    font-size: 1rem;
    color: #5dcaa5;
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.res-bar-track {
    background: rgba(93,202,165,0.1);
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
    border: 1px solid rgba(93,202,165,0.12);
    max-width: 280px;
    margin: 0 auto 0.8rem;
}
.res-bar {
    height: 100%;
    background: linear-gradient(90deg, #1d9e75, #5dcaa5);
    border-radius: 99px;
    transition: width 0.9s;
}
.res-badge-hi {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.22rem 0.9rem; border-radius: 99px;
    font-size: 0.62rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(29,158,117,0.15);
    color: #5dcaa5;
    border: 1px solid rgba(93,202,165,0.25);
    margin-bottom: 0.7rem;
}
.res-badge-lo {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.22rem 0.9rem; border-radius: 99px;
    font-size: 0.62rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(212,83,126,0.12);
    color: #ed93b1;
    border: 1px solid rgba(237,147,177,0.2);
    margin-bottom: 0.7rem;
}
.res-ci {
    font-size: 0.7rem;
    color: #4a7f96;
    margin-bottom: 1rem;
}
.res-ci span { color: #7aacbd; font-weight: 500; }
.res-pills {
    display: flex; gap: 0.5rem;
    justify-content: center; flex-wrap: wrap;
}
.res-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(93,202,165,0.12);
    border-radius: 9px;
    padding: 0.45rem 0.65rem;
    min-width: 72px; text-align: center;
}
.pill-val {
    font-family: 'Playfair Display', serif;
    font-size: 1rem; font-weight: 600;
    color: #cfe8f0;
}
.pill-lbl {
    font-size: 0.55rem;
    color: #4a7f96;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}
.res-note {
    font-size: 0.63rem;
    color: #2a5a70;
    margin-top: 0.9rem;
    line-height: 1.6;
}

/* Placeholder */
.placeholder-box {
    border: 1px dashed rgba(93,202,165,0.14);
    border-radius: 18px;
    padding: 4rem 1.5rem;
    text-align: center;
}
.ph-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    color: #4a7f96;
    margin-bottom: 0.4rem;
}
.ph-sub {
    font-size: 0.72rem;
    color: #2a5a70;
    line-height: 1.6;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem 0 2rem;
    font-size: 0.63rem;
    color: #1e4a60;
    border-top: 1px solid rgba(93,202,165,0.07);
    margin-top: 1.5rem;
    line-height: 1.8;
}

[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    m  = joblib.load('phase4_separate_models.pkl')
    lp = joblib.load('phase4_label_encoder_pollutant.pkl')
    ln = joblib.load('phase4_label_encoder_np.pkl')
    return m, lp, ln

try:
    models, le_poll, le_np = load_models()
except FileNotFoundError as e:
    st.error(f"Missing model file: {e}")
    st.stop()

TRAIN_COLS = [
    'log_Size','Optimum pH','log_Dose','log_InitConc',
    'pH_x_Pollutant','Dose_Conc_ratio','Size_x_NP',
    'Pollutant Category','NP Family','Synthesis Type'
]

def predict_re(pollutant, np_family, synthesis, ph, dose, init_conc, particle_size):
    df = pd.DataFrame([{
        'Pollutant Category': pollutant, 'NP Family': np_family,
        'Synthesis Type': synthesis, 'Particle Size (nm)': particle_size,
        'Optimum pH': ph, 'Dose (g/L)': dose, 'Init. Conc. (mg/L)': init_conc
    }])
    df['log_Size']     = np.log1p(df['Particle Size (nm)'])
    df['log_Dose']     = np.log1p(df['Dose (g/L)'])
    df['log_InitConc'] = np.log1p(df['Init. Conc. (mg/L)'])
    df.drop(columns=['Particle Size (nm)','Dose (g/L)','Init. Conc. (mg/L)'], inplace=True)
    try:
        df['Pollutant_encoded'] = le_poll.transform([pollutant])[0]
    except Exception:
        df['Pollutant_encoded'] = 0
    df['pH_x_Pollutant']  = df['Optimum pH'] * df['Pollutant_encoded']
    df['Dose_Conc_ratio'] = df['log_Dose'] / (df['log_InitConc'] + 1e-6)
    try:
        df['NP_encoded'] = le_np.transform([np_family])[0]
    except Exception:
        df['NP_encoded'] = 0
    df['Size_x_NP'] = df['log_Size'] * df['NP_encoded']
    for col in TRAIN_COLS:
        if col not in df.columns:
            df[col] = (pollutant  if col == 'Pollutant Category' else
                       np_family  if col == 'NP Family' else
                       synthesis  if col == 'Synthesis Type' else 0.0)
    df = df[TRAIN_COLS]
    model = (models['Heavy Metals'] if pollutant == 'Heavy Metals' and 'Heavy Metals' in models else
             models['Dyes']         if pollutant == 'Dyes'         and 'Dyes'         in models else
             models.get('all', list(models.values())[0]))
    pred_t  = model.predict(df)[0]
    pred_re = 100 - (np.exp(pred_t) - 1)
    return float(np.clip(pred_re, 0, 100))


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="wave-header">
  <div class="eyebrow">Green Nanotechnology · Wastewater Treatment · AFRETEC 2025</div>
  <div class="hero-title">Nano<span>Predict</span></div>
  <div class="hero-sub">
    Biosynthesised nanoparticle removal efficiency predictor —
    ML model trained on 103 adsorption studies from the literature.
  </div>
</div>
<svg class="wave-divider" viewBox="0 0 500 16" xmlns="http://www.w3.org/2000/svg">
  <path d="M0 8 Q62 2 125 8 Q187 14 250 8 Q312 2 375 8 Q437 14 500 8"
        fill="none" stroke="rgba(93,202,165,0.25)" stroke-width="1"/>
</svg>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # Panel 1 — Pollutant & NP
    st.markdown('<div class="card"><div class="card-title">Pollutant &amp; Nanoparticle</div>', unsafe_allow_html=True)

    POLL_MAP = {
        'Heavy Metals (Pb, Cd, Hg, As...)':       'Heavy Metals',
        'Organic Dyes (MB, MO, Crystal Violet...)': 'Dyes',
        'Pharmaceuticals & Micropollutants':       'Pharmaceuticals',
        'Nutrients / Inorganics (NO₃, PO₄)':      'Nutrients/Inorganics',
        'Pathogens / Microbial':                   'Pathogens',
        'Precious Metals / Other':                 'Precious Metals/Other',
    }
    NP_MAP = {
        'Magnetite (Fe₃O₄) — most studied':  'Magnetite (Fe₃O₄)',
        'Composite / Functionalised NPs':    'Composite/Other',
        'Ferrite NPs (MFe₂O₄)':             'Ferrite NPs',
        'ZnO Nanoparticles':                 'ZnO NPs',
        'Zero-valent Iron (Fe⁰)':            'Iron NPs (Fe⁰)',
        'Bimetallic Fe NPs':                 'Bimetallic Fe NPs',
        'Copper-based NPs':                  'Copper-Based NPs',
    }

    poll_display = st.selectbox("Pollutant type", list(POLL_MAP.keys()))
    np_display   = st.selectbox("Nanoparticle family", list(NP_MAP.keys()))
    synth_raw    = st.radio("Biosynthesis route",
                            ["Green / Biosynthesised", "Chemical / Conventional"])
    synthesis    = "Green" if "Green" in synth_raw else "Non-Green"
    st.markdown('</div>', unsafe_allow_html=True)

    # Panel 2 — Conditions
    st.markdown('<div class="card"><div class="card-title">Adsorption Conditions</div>', unsafe_allow_html=True)
    ph = st.slider("Solution pH", 2.0, 12.0, 6.0, 0.1)
    c1, c2 = st.columns(2)
    with c1:
        dose      = st.number_input("Adsorbent dose (g/L)",        0.01, 20.0,   1.0, 0.1, format="%.2f")
        psize     = st.number_input("Particle size (nm)",          1,    250,    25,  1)
    with c2:
        init_conc = st.number_input("Init. concentration (mg/L)", 1,    1500,   40,  5)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("Predict Removal Efficiency", use_container_width=True)


with right:
    if predict_btn:
        pollutant = POLL_MAP[poll_display]
        np_family = NP_MAP[np_display]
        with st.spinner(""):
            re = predict_re(pollutant, np_family, synthesis, ph, dose, init_conc, psize)

        lo  = max(0,   re - 11)
        hi  = min(100, re + 11)
        isHi = re >= 80
        badge_cls = "res-badge-hi" if isHi else "res-badge-lo"
        badge_txt = "High performance (≥ 80%)" if isHi else "Below threshold (< 80%)"

        st.markdown(f"""
        <div class="result-outer">
          <div class="res-context">{pollutant} · {np_family} · {synthesis}</div>
          <div class="res-label">Predicted Removal Efficiency</div>
          <div class="res-value">{re:.1f}</div>
          <div class="res-unit">%</div>
          <div class="res-bar-track">
            <div class="res-bar" style="width:{re:.1f}%"></div>
          </div>
          <div class="{badge_cls}">{badge_txt}</div>
          <div class="res-ci">
            95% prediction interval &nbsp;<span>[{lo:.0f}% – {hi:.0f}%]</span>
          </div>
          <div class="res-pills">
            <div class="res-pill"><div class="pill-val">0.78</div><div class="pill-lbl">AUC-ROC</div></div>
            <div class="res-pill"><div class="pill-val">5.77%</div><div class="pill-lbl">Ext. MAE</div></div>
            <div class="res-pill"><div class="pill-val">103</div><div class="pill-lbl">Studies</div></div>
            <div class="res-pill"><div class="pill-val">17</div><div class="pill-lbl">Ext. val.</div></div>
          </div>
          <div class="res-note">
            Literature-trained ensemble model (n=103 records).<br>
            Experimental validation recommended for critical applications.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder-box">
          <svg width="48" height="48" viewBox="0 0 48 48" style="margin:0 auto 12px;display:block">
            <circle cx="24" cy="24" r="8" fill="rgba(93,202,165,0.2)"/>
            <circle cx="24" cy="24" r="16" fill="none" stroke="rgba(93,202,165,0.2)" stroke-width="1" stroke-dasharray="3 2"/>
            <circle cx="24" cy="24" r="23" fill="none" stroke="rgba(93,202,165,0.1)" stroke-width="0.8"/>
          </svg>
          <div class="ph-title">Awaiting parameters</div>
          <div class="ph-sub">
            Configure pollutant type, nanoparticle family<br>
            and adsorption conditions, then predict.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <strong style="color:#2a5a70">NanoPredict</strong> · Capstone EGR4402 ·
  Amina Yassmine Nadane · Supervised by Dr. Nadia Arrousse<br>
  External validation MAE = 5.77% across 17 independent literature studies ·
  <span style="color:#1e3a50">AFRETEC-aligned 2025</span>
</div>
""", unsafe_allow_html=True)