import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Прогноз доверия", layout="centered")

if 'answers' not in st.session_state:
    st.session_state.answers = {}

def save_answer(q_key, wkey):
    val = st.session_state.get(wkey)
    if val is not None:
        st.session_state.answers[q_key] = val

bundle       = joblib.load('models.joblib')
full_model   = bundle['full_model']
lite_model   = bundle['lite_model']
full_feats   = bundle['full_features']
lite_feats   = bundle['lite_features']
medians      = bundle['train_medians']
countries    = bundle['countries']
NUMERIC_FEATURES = bundle['lite_numeric']

COUNTRY_NAMES = {
    "AL":"Albania",
    "AT":"Austria",
    "BE":"Belgium",
    "BG":"Bulgaria",
    "CH":"Switzerland",
    "CY":"Cyprus",
    "CZ":"Czechia",
    "DE":"Germany",
    "DK":"Denmark",
    "EE":"Estonia",
    "ES":"Spain",
    "FI":"Finland",
    "FR":"France",
    "GB":"United Kingdom",
    "GE":"Georgia",
    "GR":"Greece",
    "HR":"Croatia",
    "HU":"Hungary",
    "IE":"Ireland",
    "IS":"Iceland",
    "IL":"Israel",
    "IT":"Italy",
    "LT":"Lithuania",
    "LU":"Luxembourg",
    "LV":"Latvia",
    "ME":"Montenegro",
    "MK":"North Macedonia",
    "NL":"Netherlands",
    "NO":"Norway",
    "PL":"Poland",
    "PT":"Portugal",
    "RO":"Romania",
    "RS":"Serbia",
    "RU":"Russian Federation",
    "SE":"Sweden",
    "SI":"Slovenia",
    "SK":"Slovakia",
    "TR":"Turkey",
    "UA":"Ukraine",
    "XK":"Kosovo",
}
NAME_TO_CODE = {v: k for k, v in COUNTRY_NAMES.items()}

QUESTIONS = [
    {
        'key': 'cntry',
        'label': 'В какой стране вы живёте?',
        'type': 'select',
        'options': sorted(COUNTRY_NAMES.get(c, c) for c in countries),
        'default_display': 'Норвегия',
    },
    {
        'key': 'gndr',
        'label': 'Ваш пол',
        'type': 'radio',
        'options': [1, 2],
        'labels': {1: 'Мужской', 2: 'Женский'},
    },
    {
        'key': 'stfgov',
        'label': 'Насколько вы довольны работой правительства?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Совсем недоволен', 'max_label': 'Полностью доволен',
    },
    {
        'key': 'stfdem',
        'label': 'Насколько вы довольны состоянием демократии в стране?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Совсем недоволен', 'max_label': 'Полностью доволен',
    },
    {
        'key': 'stfeco',
        'label': 'Насколько вы довольны экономической ситуацией?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Совсем недоволен', 'max_label': 'Полностью доволен',
    },
    {
        'key': 'stfedu',
        'label': 'Насколько вы довольны системой образования?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Совсем недоволен', 'max_label': 'Полностью доволен',
    },
    {
        'key': 'stfhlth',
        'label': 'Насколько вы довольны системой здравоохранения?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Совсем недоволен', 'max_label': 'Полностью доволен',
    },
    {
        'key': 'ppltrst',
        'label': 'Насколько вы согласны с тем, что большинству людей ли доверять?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Нельзя доверять', 'max_label': 'Можно доверять',
    },
    {
        'key': 'pplhlp',
        'label': 'Насколько вы согласны с тем, что большинство людей готовы помогать другим?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Думают только о себе', 'max_label': 'Готовы помогать',
    },
    {
        'key': 'happy',
        'label': 'Насколько вы счастливы?',
        'type': 'slider', 'min': 0, 'max': 10, 'step': 1,
        'min_label': 'Крайне несчастлив', 'max_label': 'Крайне счастлив',
    },
    {
        'key': 'polintr',
        'label': 'Насколько вы интересуетесь политикой?',
        'type': 'radio',
        'options': [1, 2, 3, 4],
        'labels': {1: 'Очень интересуюсь',   2: 'Довольно интересуюсь',
                   3: 'Мало интересуюсь',    4: 'Совсем не интересуюсь'},
    },
]

N = len(QUESTIONS)

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'finished' not in st.session_state:
    st.session_state.finished = False

for q in QUESTIONS:
    wkey = f"w_{q['key']}"
    if wkey not in st.session_state:
        if q['key'] in st.session_state.answers:
            st.session_state[wkey] = st.session_state.answers[q['key']]
        elif q['type'] == 'select':
            st.session_state[wkey] = q.get('default_display', q['options'][0])
        elif q['type'] == 'radio':
            default_val = int(medians.get(q['key'], q['options'][0]))
            st.session_state[wkey] = default_val if default_val in q['options'] else q['options'][0]
        else:
            st.session_state[wkey] = int(medians.get(q['key'], q.get('min', 0)))

def get_current_row():
    row = medians.copy()
    for q in QUESTIONS:
        key = q['key']
        if key in st.session_state.answers:
            val = st.session_state.answers[key]
            if key == 'cntry':
                val = NAME_TO_CODE.get(val, val)
            row[key] = val
    return row

def get_predictions():
    row      = get_current_row()
    lite_row = pd.DataFrame([row])[lite_feats]
    full_row = pd.DataFrame([row])[full_feats]
    pred_lite = float(np.clip(lite_model.predict(lite_row)[0], 0, 10))
    pred_full = float(np.clip(full_model.predict(full_row)[0], 0, 10))
    return round(pred_lite, 2), round(pred_full, 2)

def trust_label(v):
    if v < 2.5: return "Очень низкое"
    if v < 4.0: return "Низкое"
    if v < 7.0: return "Среднее"
    if v < 8.0: return "Высокое"
    return "Очень высокое"

def trust_color(v):
    if v < 3:   return "#e74c3c"
    if v < 5:   return "#f39c12"
    if v < 6.5: return "#2ecc71"
    return "#1a9e6e"

def gauge_chart(value, color):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        number = {
            'font': {'size': 48, 'color': color},
            'suffix': '',
        },
        gauge = {
            'axis': {
                'range': [0, 10],
                'tickwidth': 1,
                'tickcolor': '#ccc',
                'tickvals': [0, 2.5, 5, 7.5, 10],
                'ticktext': ['0', '2.5', '5', '7.5', '10'],
                'tickfont': {'size': 11, 'color': '#999'},
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0,   3],   'color': '#fdecea'},
                {'range': [3,   5],   'color': '#fff3e0'},
                {'range': [5,   6.5], 'color': '#e8f5e9'},
                {'range': [6.5, 10],  'color': '#d0f0e8'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': value,
            },
        },
    ))
    fig.update_layout(
        height  = 220,
        margin  = dict(t=20, b=0, l=20, r=20),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font = {'family': 'sans-serif'},
    )
    return fig

st.markdown("## Прогноз доверия к государственным институтам")

pred_lite, pred_full = get_predictions()
color_lite = trust_color(pred_lite)
color_full = trust_color(pred_full)

st.markdown(
    f"<div style='text-align:center; font-size:13px; color:#888;",
    unsafe_allow_html=True
)
st.plotly_chart(gauge_chart(pred_lite, color_lite),
                use_container_width=True, config={'displayModeBar': False})
st.markdown(
    f"<div style='text-align:center; margin-top:-15px'>"
    f"<span style='font-size:16px; font-weight:600; color:{color_lite}'>"
    f"{trust_label(pred_lite)}</span><br>"
    f"</div>",
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div style='border: 1px solid 44; border-radius: 12px;
                padding: 1rem; text-align: center; background: 0d;
                margin-top: 2rem'>
        <div style='font-size: 12px; color: #888; margin-bottom: 4px'>
            Полная модель
        </div>
        <div style='font-size: 38px; font-weight: 600;
                    color: {color_full}; line-height: 1'>
            {pred_full:.1f}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



st.divider()

if st.session_state.finished:
    st.markdown("### Ваши ответы")
    for q in QUESTIONS:
        val = st.session_state.answers.get(q['key'], '—')
        if q['type'] == 'radio':
            val = q['labels'].get(val, val)
        if q['key'] == 'cntry':
            val = st.session_state.answers.get('cntry', '—')
        st.markdown(f"**{q['label']}** — {val}")

    st.divider()
    if st.button("← Пройти заново", use_container_width=True):
        for q in QUESTIONS:
            wkey = f"w_{q['key']}"
            if wkey in st.session_state:
                del st.session_state[wkey]
        st.session_state.step     = 0
        st.session_state.finished = False
        st.session_state.answers  = {}
        st.rerun()

else:
    step = st.session_state.step
    q    = QUESTIONS[step]
    wkey = f"w_{q['key']}"

    st.progress(step / N)
    st.markdown(
        f"<div style='font-size:13px; color:#888; margin-bottom:0.75rem'>"
        f"Вопрос {step + 1} из {N}</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"### {q['label']}")
    if q.get('hint'):
        st.caption(q['hint'])

    if q['type'] == 'slider':
        st.slider(
            label            = q['label'],
            min_value        = q['min'],
            max_value        = q['max'],
            step             = q['step'],
            format           = q.get('format'),
            key              = wkey,
            label_visibility = 'hidden',
            on_change        = save_answer,
            args             = (q['key'], wkey),
        )
        if 'min_label' in q:
            cl, cr = st.columns(2)
            cl.caption(q['min_label'])
            cr.markdown(
                f"<div style='text-align:right;font-size:12px;color:#888'>"
                f"{q['max_label']}</div>",
                unsafe_allow_html=True
            )

    elif q['type'] == 'radio':
        st.radio(
            label            = q['label'],
            options          = q['options'],
            format_func      = lambda x, _q=q: _q['labels'].get(x, str(x)),
            key              = wkey,
            label_visibility = 'hidden',
            horizontal       = len(q['options']) <= 3,
            on_change        = save_answer,
            args             = (q['key'], wkey),
        )

    elif q['type'] == 'select':
        st.selectbox(
            label            = q['label'],
            options          = q['options'],
            key              = wkey,
            label_visibility = 'hidden',
            on_change        = save_answer,
            args             = (q['key'], wkey),
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 3])

    with col_back:
        if step > 0:
            if st.button("← Назад", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()

    with col_next:
        is_last = (step == N - 1)
        if st.button(
            "Узнать результат →" if is_last else "Далее →",
            use_container_width=True,
            type="primary",
        ):
            if is_last:
                st.session_state.finished = True
            else:
                st.session_state.step += 1
            st.rerun()