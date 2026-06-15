import streamlit as st
import joblib

from tabs.eda     import render_eda
from tabs.models  import render_models
from tabs.predict import render_predict

st.set_page_config(page_title="Доверие к институтам", layout="centered")

# ── Загрузка модели ───────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load('models.joblib')

bundle = load_bundle()

# ── Хедер ─────────────────────────────────────────────────────────────────────
st.markdown("## Прогноз доверия к государственным институтам")

tab_eda, tab_models, tab_predict = st.tabs(["EDA", "Модели", "Прогноз"])

with tab_eda:
    render_eda()

with tab_models:
    render_models()

with tab_predict:
    render_predict(bundle)