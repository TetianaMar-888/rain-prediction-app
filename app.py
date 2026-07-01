import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain in Australia?",
    page_icon="🌧️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1b2a; }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 100%);
        border: 1px solid #1e4976;
        border-radius: 16px;
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.2rem; color: #e8f4fd; margin-bottom: .3rem; }
    .hero p  { color: #7fb3d3; font-size: 1rem; }

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #4a90d9;
        margin: 1.8rem 0 .6rem;
    }

    .result-box {
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-rain   { background: #0a2540; border: 1px solid #1e6cb8; }
    .result-norain { background: #0a2014; border: 1px solid #1a6b3a; }
    .result-box h2 { font-size: 2rem; margin: 0; }
    .result-box p  { margin: .5rem 0 0; font-size: 1rem; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #1565c0, #1e88e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: .75rem 2.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
        cursor: pointer;
        transition: opacity .2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: .88; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("aussie_rain.joblib")   # ← шлях до файлу в репо

try:
    bundle = load_model()
except FileNotFoundError:
    st.error("❌ Файл `aussie_rain.joblib` не знайдено. Переконайся, що він лежить поруч з `app.py`.")
    st.stop()

model        = bundle["model"]
imputer      = bundle["imputer"]
scaler       = bundle["scaler"]
encoder      = bundle["encoder"]
numeric_cols = bundle["numeric_cols"]
cat_cols     = bundle["categorical_cols"]
encoded_cols = bundle["encoded_cols"]
input_cols   = bundle["input_cols"]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌧️ Rain Tomorrow?</h1>
  <p>Enter weather data to find out if it will rain tomorrow</p>
</div>
""", unsafe_allow_html=True)

# Location options
locations = [c.replace("Location_", "") for c in encoded_cols if c.startswith("Location_")]
wind_dirs = [c.replace("WindGustDir_", "") for c in encoded_cols
             if c.startswith("WindGustDir_") and c != "WindGustDir_nan"]
wind_dirs_with_na = ["(not observed)"] + wind_dirs

# ── Section: Location ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
location = st.selectbox("Weather station", locations, label_visibility="collapsed")

# ── Section: Temperature ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">🌡️ Temperature (°C)</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
min_temp  = c1.number_input("Min Temp",   value=12.0, step=0.5)
max_temp  = c2.number_input("Max Temp",   value=24.0, step=0.5)
temp_9am  = c3.number_input("Temp 9am",   value=16.0, step=0.5)
temp_3pm  = c4.number_input("Temp 3pm",   value=22.0, step=0.5)

# ── Section: Rain & Evaporation ────────────────────────────────────────────────
st.markdown('<div class="section-label">💧 Rain & Evaporation</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
rainfall    = c1.number_input("Rainfall (mm)",     value=0.0,  step=0.1, min_value=0.0)
evaporation = c2.number_input("Evaporation (mm)",  value=4.8,  step=0.1, min_value=0.0)
sunshine    = c3.number_input("Sunshine (hrs)",    value=7.0,  step=0.1, min_value=0.0, max_value=14.0)
rain_today  = st.radio("Did it rain today?", ["No", "Yes"], horizontal=True)

# ── Section: Wind ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">💨 Wind</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
wind_gust_dir = c1.selectbox("Gust direction",    wind_dirs_with_na)
wind_dir_9am  = c2.selectbox("Wind dir at 9am",   wind_dirs_with_na)
wind_dir_3pm  = c3.selectbox("Wind dir at 3pm",   wind_dirs_with_na)

c1, c2, c3 = st.columns(3)
wind_gust_speed = c1.number_input("Gust speed (km/h)", value=35.0, step=1.0, min_value=0.0)
wind_speed_9am  = c2.number_input("Wind speed 9am",    value=14.0, step=1.0, min_value=0.0)
wind_speed_3pm  = c3.number_input("Wind speed 3pm",    value=20.0, step=1.0, min_value=0.0)

# ── Section: Humidity & Pressure ──────────────────────────────────────────────
st.markdown('<div class="section-label">🔵 Humidity & Pressure</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
humidity_9am  = c1.slider("Humidity 9am (%)",  0, 100, 70)
humidity_3pm  = c2.slider("Humidity 3pm (%)",  0, 100, 55)
pressure_9am  = c3.number_input("Pressure 9am (hPa)", value=1017.0, step=0.1)
pressure_3pm  = c4.number_input("Pressure 3pm (hPa)", value=1015.0, step=0.1)

# ── Section: Cloud ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">☁️ Cloud cover (oktas 0–8)</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
cloud_9am = c1.slider("Cloud 9am", 0, 8, 4)
cloud_3pm = c2.slider("Cloud 3pm", 0, 8, 5)

# ── Predict ────────────────────────────────────────────────────────────────────
def map_dir(val):
    return np.nan if val == "(not observed)" else val

if st.button("🔍 Predict tomorrow's rain"):

    raw = pd.DataFrame([{
        "Location":       location,
        "MinTemp":        min_temp,
        "MaxTemp":        max_temp,
        "Rainfall":       rainfall,
        "Evaporation":    evaporation,
        "Sunshine":       sunshine,
        "WindGustDir":    map_dir(wind_gust_dir),
        "WindGustSpeed":  wind_gust_speed,
        "WindDir9am":     map_dir(wind_dir_9am),
        "WindDir3pm":     map_dir(wind_dir_3pm),
        "WindSpeed9am":   wind_speed_9am,
        "WindSpeed3pm":   wind_speed_3pm,
        "Humidity9am":    humidity_9am,
        "Humidity3pm":    humidity_3pm,
        "Pressure9am":    pressure_9am,
        "Pressure3pm":    pressure_3pm,
        "Cloud9am":       cloud_9am,
        "Cloud3pm":       cloud_3pm,
        "Temp9am":        temp_9am,
        "Temp3pm":        temp_3pm,
        "RainToday":      rain_today,
    }])

    # Preprocessing
    raw_numeric = raw[numeric_cols].copy()
    raw_numeric = pd.DataFrame(
        imputer.transform(raw_numeric),
        columns=numeric_cols
    )
    raw_numeric = pd.DataFrame(
        scaler.transform(raw_numeric),
        columns=numeric_cols
    )

    raw_cat = raw[cat_cols].copy()
    encoded = encoder.transform(raw_cat)
    enc_df  = pd.DataFrame(encoded, columns=encoded_cols)

    X = pd.concat([raw_numeric.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)

    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    rain_prob   = proba[1] * 100
    no_rain_prob = proba[0] * 100

    if pred == "Yes" or pred == 1:
        st.markdown(f"""
        <div class="result-box result-rain">
          <h2>🌧️ Rain expected</h2>
          <p>Probability of rain: <strong>{rain_prob:.1f}%</strong> &nbsp;·&nbsp; No rain: {no_rain_prob:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-norain">
          <h2>☀️ No rain expected</h2>
          <p>Probability of no rain: <strong>{no_rain_prob:.1f}%</strong> &nbsp;·&nbsp; Rain: {rain_prob:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
