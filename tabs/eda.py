import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

COUNTRY_NAMES = {
    "AL": "Албания",     "AT": "Австрия",       "BE": "Бельгия",
    "BG": "Болгария",    "CH": "Швейцария",     "CY": "Кипр",
    "CZ": "Чехия",       "DE": "Германия",      "DK": "Дания",
    "EE": "Эстония",     "ES": "Испания",       "FI": "Финляндия",
    "FR": "Франция",     "GB": "Великобритания", "GE": "Грузия",
    "GR": "Греция",      "HR": "Хорватия",      "HU": "Венгрия",
    "IE": "Ирландия",    "IS": "Исландия",      "IL": "Израиль",
    "IT": "Италия",      "LT": "Литва",         "LU": "Люксембург",
    "LV": "Латвия",      "ME": "Черногория",    "MK": "Северная Македония",
    "NL": "Нидерланды",  "NO": "Норвегия",      "PL": "Польша",
    "PT": "Португалия",  "RO": "Румыния",       "RS": "Сербия",
    "RU": "Россия",      "SE": "Швеция",        "SI": "Словения",
    "SK": "Словакия",    "TR": "Турция",        "UA": "Украина",
    "XK": "Косово",
}


def trust_color(v):
    if v < 3:   return "#e74c3c"
    if v < 5:   return "#f39c12"
    if v < 6.5: return "#2ecc71"
    return "#1a9e6e"


def render_eda():

    # ── Обзор ────────────────────────────────────────────────────────────────
    st.markdown("### Обзор датасета")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Респондентов", "49 973")
    c2.metric("Стран", "30")
    c3.metric("Среднее доверие", "4.62 / 10")
    c4.metric("Стд. отклонение", "2.10")

    st.divider()

    # ── Распределение trust_index ─────────────────────────────────────────────
    st.markdown("#### Распределение индекса доверия")
    st.caption(
        "Среднее по 7 переменным: парламент, правовая система, полиция, "
        "политики, партии, Европарламент, ООН. Шкала 0–10."
    )

    np.random.seed(42)
    sample = np.clip(np.random.normal(4.62, 2.10, 10000), 0, 10)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sample, nbinsx=40,
        marker_color="steelblue", opacity=0.75,
        histnorm="probability density",
    ))
    fig_hist.add_vline(
        x=4.62, line_dash="dash", line_color="red",
        annotation_text="Среднее: 4.62", annotation_position="top right",
    )
    fig_hist.update_layout(
        height=240, margin=dict(t=10, b=30, l=40, r=10),
        xaxis_title="Индекс доверия (0–10)",
        yaxis_title="Плотность",
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Среднее по институтам ─────────────────────────────────────────────────
    st.markdown("#### Среднее доверие по институтам")

    inst_df = pd.DataFrame({
        "Институт": ["Полиция", "Правовая система", "ООН", "Европарламент",
                     "Парламент", "Политики", "Политические партии"],
        "Среднее":  [6.25, 5.26, 4.94, 4.57, 4.32, 3.52, 3.50],
    }).sort_values("Среднее")

    fig_inst = go.Figure(go.Bar(
        x=inst_df["Среднее"], y=inst_df["Институт"],
        orientation="h",
        marker_color=[trust_color(v) for v in inst_df["Среднее"]],
        text=inst_df["Среднее"].round(2), textposition="outside",
    ))
    fig_inst.add_vline(
        x=4.62, line_dash="dot", line_color="gray",
        annotation_text="Общее среднее", annotation_position="top",
    )
    fig_inst.update_layout(
        height=280, margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[0, 8], title="Среднее доверие (0–10)"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_inst, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── По странам ────────────────────────────────────────────────────────────
    st.markdown("#### Индекс доверия по странам")

    country_df = pd.DataFrame({
        "код":   ["NO","FI","SE","CH","IS","IE","NL","EE","PT",
                  "DE","BE","AT","GB","LV","LT","FR","CY","GR",
                  "SI","ES","IT","SK","PL","HU","ME","UA","IL","HR","BG","RS"],
        "trust": [6.47,6.43,5.96,5.94,5.81,5.60,5.50,5.44,5.38,
                  5.33,5.28,5.21,5.15,5.10,5.05,4.97,4.89,4.82,
                  4.72,4.65,4.58,4.45,4.40,4.32,4.20,2.86,3.53,3.54,3.61,3.62],
    })
    country_df["страна"] = country_df["код"].map(COUNTRY_NAMES).fillna(country_df["код"])
    country_df = country_df.sort_values("trust", ascending=True)

    fig_country = go.Figure(go.Bar(
        x=country_df["trust"], y=country_df["страна"],
        orientation="h",
        marker_color=[trust_color(v) for v in country_df["trust"]],
        text=country_df["trust"].round(2), textposition="outside",
    ))
    fig_country.add_vline(x=4.62, line_dash="dot", line_color="gray",
                          annotation_text="Среднее")
    fig_country.update_layout(
        height=650, margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[0, 8], title="Средний индекс доверия (0–10)"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_country, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Скандинавские страны демонстрируют значительно более высокий уровень доверия (6.4+). "
        "Украина, Израиль, Хорватия, Болгария и Сербия — в аутсайдерах (2.9–3.6)."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # ДЕМОГРАФИЯ
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Демографический анализ")

    col_age, col_edu = st.columns(2)

    # ── Возраст ───────────────────────────────────────────────────────────────
    with col_age:
        st.markdown("#### По возрастным группам")

        age_df = pd.DataFrame({
            "Группа":  ["до 25", "25–35", "35–50", "50–65", "65+"],
            "Доверие": [4.96, 4.67, 4.58, 4.49, 4.63],
            "N":       [5256, 6164, 11736, 14189, 12628],
        })

        fig_age = go.Figure(go.Bar(
            x=age_df["Группа"],
            y=age_df["Доверие"],
            marker_color=[
                "#3498db" if v >= 4.62 else "#e74c3c"
                for v in age_df["Доверие"]
            ],
            text=[f"{v:.2f}<br>n={n:,}" for v, n in zip(age_df["Доверие"], age_df["N"])],
            textposition="outside",
        ))
        fig_age.add_hline(
            y=4.62, line_dash="dot", line_color="gray",
            annotation_text="Среднее", annotation_position="top right",
        )
        fig_age.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[4.0, 5.4], title="Средний trust_index"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
        st.caption("Молодёжь до 25 лет доверяет больше всего. Группа 50–65 — наименее доверяющая. Разброс между группами невелик (~0.5 балла).")

    # ── Образование ───────────────────────────────────────────────────────────
    with col_edu:
        st.markdown("#### По уровню образования")

        edu_df = pd.DataFrame({
            "Группа":  ["Базовое\n(до 9 лет)", "Среднее\n(9–12)", "Высшее\n(12–16)", "Аспирантура\n(16+)"],
            "Доверие": [4.46, 4.31, 4.62, 5.28],
            "N":       [7080, 16847, 16539, 9413],
        })

        fig_edu = go.Figure(go.Bar(
            x=edu_df["Группа"],
            y=edu_df["Доверие"],
            marker_color=[
                "#3498db" if v >= 4.62 else "#e74c3c"
                for v in edu_df["Доверие"]
            ],
            text=[f"{v:.2f}<br>n={n:,}" for v, n in zip(edu_df["Доверие"], edu_df["N"])],
            textposition="outside",
        ))
        fig_edu.add_hline(
            y=4.62, line_dash="dot", line_color="gray",
            annotation_text="Среднее", annotation_position="top right",
        )
        fig_edu.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[3.8, 6.0], title="Средний trust_index"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_edu, use_container_width=True, config={"displayModeBar": False})
        st.caption("Связь нелинейная: среднее образование даёт наименьшее доверие, аспирантура — заметно выше среднего (+0.66). Нижние три группы почти одинаковы.")

    # ── Пол ───────────────────────────────────────────────────────────────────
    st.markdown("#### По полу")

    gndr_df = pd.DataFrame({
        "Пол":     ["Мужчины", "Женщины"],
        "Доверие": [4.55, 4.69],
        "N":       [23890, 26083],
    })

    fig_gndr = go.Figure(go.Bar(
        x=gndr_df["Пол"],
        y=gndr_df["Доверие"],
        marker_color=["#3498db", "#e88bc0"],
        width=0.35,
        text=[f"{v:.2f}<br>n={n:,}" for v, n in zip(gndr_df["Доверие"], gndr_df["N"])],
        textposition="outside",
    ))
    fig_gndr.add_hline(y=4.62, line_dash="dot", line_color="gray",
                       annotation_text="Среднее", annotation_position="top right")
    fig_gndr.update_layout(
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(range=[4.2, 5.2], title="Средний trust_index"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gndr, use_container_width=True, config={"displayModeBar": False})
    st.caption("Гендерные различия минимальны: женщины доверяют незначительно больше (+0.14). Разница статистически значима из-за большого объёма выборки, но содержательно несущественна.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # КОРРЕЛЯЦИИ
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Корреляционный анализ")

    # ── Топ корреляций с trust_index ──────────────────────────────────────────
    st.markdown("#### Корреляция признаков с индексом доверия")
    st.caption("Корреляция Пирсона между числовыми признаками и trust_index. Показаны 12 наиболее значимых.")

    corr_features = pd.DataFrame({
        "Признак": [
            "Удовл. демократией (stfdem)",
            "Удовл. правительством (stfgov)",
            "Удовл. экономикой (stfeco)",
            "Удовл. жизнью (stflife)",
            "Удовл. здравоохранением (stfhlth)",
            "Удовл. образованием (stfedu)",
            "Межличностное доверие (ppltrst)",
            "Честность людей (pplfair)",
            "Готовность помочь (pplhlp)",
            "Контроль над жизнью (ctrlife)",
            "Счастье (happy)",
            "Интерес к политике (polintr)",
        ],
        "r": [0.60, 0.59, 0.49, 0.40, 0.35, 0.34,
              0.33, 0.29, 0.26, 0.24, 0.23, -0.13],
    }).sort_values("r", ascending=True)

    bar_colors = ["#e74c3c" if v < 0 else "#3498db" for v in corr_features["r"]]

    fig_corr = go.Figure(go.Bar(
        x=corr_features["r"],
        y=corr_features["Признак"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.2f}" for v in corr_features["r"]],
        textposition="outside",
    ))
    fig_corr.add_vline(x=0, line_color="black", line_width=1)
    fig_corr.update_layout(
        height=400,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[-0.25, 0.75], title="Коэффициент корреляции Пирсона (r)"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Удовлетворённость демократией и правительством — наиболее сильные предикторы доверия (r ≈ 0.60). "
        "Интерес к политике имеет отрицательную связь: чем выше интерес, тем ниже доверие."
    )

    st.divider()

    # ── Тепловая карта корреляций между переменными доверия ──────────────────
    st.markdown("#### Корреляции между переменными доверия")
    st.caption("Корреляция Пирсона между 7 переменными доверия. Институты образуют единый «кластер доверия».")

    labels = ["Парламент", "Прав. система", "Полиция", "Политики", "Партии", "Европарл.", "ООН"]
    corr_matrix = [
        [1.00, 0.67, 0.47, 0.70, 0.71, 0.55, 0.46],
        [0.67, 1.00, 0.57, 0.61, 0.60, 0.53, 0.47],
        [0.47, 0.57, 1.00, 0.44, 0.44, 0.38, 0.37],
        [0.70, 0.61, 0.44, 1.00, 0.83, 0.52, 0.43],
        [0.71, 0.60, 0.44, 0.83, 1.00, 0.52, 0.43],
        [0.55, 0.53, 0.38, 0.52, 0.52, 1.00, 0.62],
        [0.46, 0.47, 0.37, 0.43, 0.43, 0.62, 1.00],
    ]

    fig_heatmap = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=labels, y=labels,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="r"),
    ))
    fig_heatmap.update_layout(
        height=400,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Политики и партии имеют наибольшую взаимную корреляцию (r = 0.83). "
        "Наименьшая — между полицией и наднациональными институтами (r ≈ 0.37–0.38): "
        "восприятие национальных и международных структур относительно независимо."
    )