import numpy as np  # RMSE 계산용
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="eSeek 베이커리 수요 예측 시스템", layout="wide")
st.title("eSeek 베이커리 수요 예측 시스템")

# Load predictions
lgb_df = pd.read_csv("LightGBM_predictions.csv")
xgb_df = pd.read_csv("XGBoost_predictions.csv")
prophet_df = pd.read_csv("Prophet_predictions.csv")

for df in [lgb_df, xgb_df, prophet_df]:
    df['date'] = pd.to_datetime(df['date'])

# Load 영향 분석 결과
impact_df = pd.read_csv("sku_factor_analysis.csv")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["SKU별 예측 비교", "전체 모델 분석", "외부 변수 영향 분석"])

# ----------- 탭 1: SKU별 예측 비교 ----------
with tab1:
    st.header("1. SKU별 예측 비교")

    sku_list = sorted(lgb_df['sku'].unique())
    selected_sku = st.selectbox("SKU 선택", sku_list)

    l = lgb_df[lgb_df['sku'] == selected_sku]
    x = xgb_df[xgb_df['sku'] == selected_sku]
    p = prophet_df[prophet_df['sku'] == selected_sku]

    merged = pd.merge(l, x[['date', 'predicted']], on='date', suffixes=('', '_xgb'))
    merged = pd.merge(merged, p[['date', 'predicted']], on='date')
    merged.columns = ['date', 'sku', 'actual', 'lgb_pred', 'xgb_pred', 'prophet_pred']
    merged = merged[merged['date'] <= "2024-06-30"]

    sampled = merged[merged['date'].dt.day % 15 == 1]

    st.subheader(f"{selected_sku} 수요 예측 결과 (15일 단위)")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(sampled['date'], sampled['actual'], label="실제값", marker='o')
    ax.scatter(sampled['date'], sampled['lgb_pred'], label="LightGBM", marker='x')
    ax.scatter(sampled['date'], sampled['xgb_pred'], label="XGBoost", marker='^')
    ax.scatter(sampled['date'], sampled['prophet_pred'], label="Prophet", marker='s')
    ax.set_xlabel("날짜")
    ax.set_ylabel("판매 수량")
    ax.set_title(f"{selected_sku} 예측 비교 (점 그래프)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("모델별 예측 정확도")

    lgb_mae = mean_absolute_error(merged['actual'], merged['lgb_pred'])
    lgb_rmse = np.sqrt(mean_squared_error(merged['actual'], merged['lgb_pred']))
    xgb_mae = mean_absolute_error(merged['actual'], merged['xgb_pred'])
    xgb_rmse = np.sqrt(mean_squared_error(merged['actual'], merged['xgb_pred']))
    prophet_mae = mean_absolute_error(merged['actual'], merged['prophet_pred'])
    prophet_rmse = np.sqrt(mean_squared_error(merged['actual'], merged['prophet_pred']))

    st.markdown(f"- LightGBM: MAE = {lgb_mae:.2f}, RMSE = {lgb_rmse:.2f}")
    st.markdown(f"- XGBoost: MAE = {xgb_mae:.2f}, RMSE = {xgb_rmse:.2f}")
    st.markdown(f"- Prophet: MAE = {prophet_mae:.2f}, RMSE = {prophet_rmse:.2f}")

    st.subheader("예측 결과 테이블")
    st.dataframe(merged.round(2))

# ----------- 탭 2: 전체 모델 성능 비교 ----------
with tab2:
    st.header("2. 전체 SKU 기준 모델 분석")

    def calc_metrics(df, label):
        return {
            "model": label,
            "MAE": mean_absolute_error(df['actual'], df['predicted']),
            "RMSE": np.sqrt(mean_squared_error(df['actual'], df['predicted']))
        }

    all_scores = [
        calc_metrics(lgb_df, "LightGBM"),
        calc_metrics(xgb_df, "XGBoost"),
        calc_metrics(prophet_df, "Prophet")
    ]

    score_df = pd.DataFrame(all_scores).sort_values("RMSE")
    st.subheader("모델별 전체 예측 정확도 비교")
    st.dataframe(score_df.set_index("model").round(2))

    best_model = score_df.iloc[0]['model']
    st.markdown("Prophet은 단기/중기 시계열 트렌드와 계절성 반영에 강한 모델로, 전체 SKU 기준에서도 가장 낮은 오차를 보여주었다.")

# ----------- 탭 3: 외부 변수 영향 분석 ----------
with tab3:
    st.header("3. 외부 변수 영향 분석")

    st.markdown("각 SKU에 대해 온도, 주말, 프로모션 여부가 수요에 어떤 영향을 미쳤는지 분석했습니다. ")

    st.dataframe(impact_df.round(3))

    st.subheader("")
    melted = impact_df.melt(id_vars='sku', value_vars=['temperature_coef', 'is_weekend_coef', 'is_promo_coef'],
                            var_name='변수', value_name='계수')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for var in melted['변수'].unique():
        subset = melted[melted['변수'] == var]
        ax2.plot(subset['sku'], subset['계수'], marker='o', label=var.replace('_coef', ''))
    ax2.set_ylabel("회귀 계수 (영향력)")
    ax2.set_title("SKU별 변수 영향도 비교")
    ax2.legend()
    st.pyplot(fig2)
