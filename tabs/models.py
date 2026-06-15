import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_models():
    st.markdown("### Сравнение моделей")
    st.caption(
        "Все модели обучены на 99 признаках + страна (one-hot encoding)."
    )

    models_df = pd.DataFrame([
        {'Модель': 'Baseline: общее среднее',     'RMSE': 2.0921, 'MAE': 1.7160, 'R²': -0.0000, 'CV R²': None,   'тип': 'baseline'},
        {'Модель': 'Baseline: медиана по стране', 'RMSE': 1.9168, 'MAE': 1.5202, 'R²':  0.1605, 'CV R²': None,   'тип': 'baseline'},
        {'Модель': 'Decision Tree',               'RMSE': 1.5688, 'MAE': 1.2289, 'R²':  0.4377, 'CV R²': 0.4215, 'тип': 'other'},
        {'Модель': 'KNN',                         'RMSE': 1.4796, 'MAE': 1.1639, 'R²':  0.4998, 'CV R²': 0.4511, 'тип': 'other'},
        {'Модель': 'SVR',                         'RMSE': 1.3465, 'MAE': 1.0535, 'R²':  0.5857, 'CV R²': 0.5278, 'тип': 'other'},
        {'Модель': 'ElasticNet',                  'RMSE': 1.3230, 'MAE': 1.0247, 'R²':  0.6000, 'CV R²': 0.5605, 'тип': 'linear'},
        {'Модель': 'Lasso',                       'RMSE': 1.3124, 'MAE': 1.0146, 'R²':  0.6065, 'CV R²': None,   'тип': 'linear'},
        {'Модель': 'Ridge',                       'RMSE': 1.3122, 'MAE': 1.0144, 'R²':  0.6066, 'CV R²': None,   'тип': 'linear'},
        {'Модель': 'Линейная регрессия',          'RMSE': 1.3122, 'MAE': 1.0143, 'R²':  0.6066, 'CV R²': 0.5565, 'тип': 'linear'},
        {'Модель': 'Extra Trees',                 'RMSE': 1.2921, 'MAE': 1.0039, 'R²':  0.6186, 'CV R²': None,   'тип': 'ensemble'},
        {'Модель': 'Random Forest',               'RMSE': 1.3395, 'MAE': 1.0396, 'R²':  0.5900, 'CV R²': 0.5484, 'тип': 'ensemble'},
        {'Модель': 'HistGradientBoosting',        'RMSE': 1.2741, 'MAE': 0.9833, 'R²':  0.6291, 'CV R²': 0.5764, 'тип': 'ensemble'},
        {'Модель': 'Gradient Boosting',           'RMSE': 1.2784, 'MAE': 0.9879, 'R²':  0.6266, 'CV R²': 0.5771, 'тип': 'ensemble'},
        {'Модель': 'XGBoost',                     'RMSE': 1.2405, 'MAE': 0.9514, 'R²':  0.6484, 'CV R²': None,   'тип': 'ensemble'},
        {'Модель': 'LightGBM',                    'RMSE': 1.2546, 'MAE': 0.9660, 'R²':  0.6404, 'CV R²': None,   'тип': 'ensemble'},
        {'Модель': 'CatBoost',                  'RMSE': 1.2345, 'MAE': 0.9485, 'R²':  0.6518, 'CV R²': None,   'тип': 'best'},
    ]).sort_values('R²', ascending=False)

    # ── Таблица ───────────────────────────────────────────────────────────────
    def highlight_best(row):
        if '★' in row['Модель']:
            return ['background-color: #d0f0e8'] * len(row)
        if 'Baseline' in row['Модель']:
            return ['background-color: #fdecea'] * len(row)
        return [''] * len(row)

    display_df = models_df[['Модель', 'RMSE', 'MAE', 'R²', 'CV R²']].copy()
    display_df['RMSE'] = display_df['RMSE'].round(4)
    display_df['MAE']  = display_df['MAE'].round(4)
    display_df['R²']   = display_df['R²'].round(4)
    display_df['CV R²'] = display_df['CV R²'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else '—'
    )

    st.dataframe(
        display_df,
        use_container_width=True, hide_index=True, height=560,
    )

    st.divider()

    # ── График R² ────────────────────────────────────────────────────────────
    st.markdown("#### Сравнение R²")

    plot_df = models_df[~models_df['Модель'].str.contains('Baseline')].copy()
    plot_df = plot_df.sort_values('R²', ascending=True)
    bar_colors = [
        '#1a9e6e' if t == 'best'
        else '#3498db' if t == 'ensemble'
        else '#95a5a6'
        for t in plot_df['тип']
    ]

    fig_r2 = go.Figure(go.Bar(
        x=plot_df['R²'], y=plot_df['Модель'],
        orientation='h', marker_color=bar_colors,
        text=plot_df['R²'].round(4), textposition='outside',
    ))
    fig_r2.update_layout(
        height=460, margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(range=[0.35, 0.72], title='R²'),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_r2, use_container_width=True,
                    config={'displayModeBar': False})

    st.divider()

    # ── Лайт vs полная ───────────────────────────────────────────────────────
    st.markdown("#### Лайт-модель vs Полная модель")
    st.caption("Лайт-модель использует только 10 признаков из опросника и применяется на вкладке «Прогноз».")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Лайт-модель** — Линейная регрессия, 10 признаков + страна")
        st.metric("RMSE", "1.3703")
        st.metric("MAE",  "1.0620")
        st.metric("R²",   "0.5710")
    with c2:
        st.markdown("**Полная модель** — CatBoost, 99 признаков + страна")
        st.metric("RMSE", "1.2345")
        st.metric("MAE",  "0.9485")
        st.metric("R²",   "0.6518")

    st.caption(
        "Потеря качества при переходе от полной к лайт-модели: −0.081 R² (12%). "
    )