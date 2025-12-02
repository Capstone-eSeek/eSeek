import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, Any, List
from pathlib import Path

# Scikit-learn 및 예측 모델
from sklearn.ensemble import HistGradientBoostingRegressor
from pandas.api.types import is_categorical_dtype

# 외부 유틸리티 함수 import
from python.prediction_utils import (
    build_calendar_features,
    attach_events,
    fetch_weather_daily,
    make_prediction_grid,
    safe_merge,
    expand_events,
    setup_column_transformer,
    TIMEZONE
)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 지점별 설정 (predict_module 내부에서 필요한 하드코딩 값)
LAT, LON = 33.46, 126.31  # 제주 애월 좌표(대략)
HOLIDAY_WINDOW = 2

# 제주 애월점 이벤트 (원본 코드 내용 유지)
JEJU_EVENTS = [
    {"name": "world_donut_day", "start": "2024-06-07", "end": "2024-06-07", "type": "free_glazed"},
    {"name": "picnic_mat",      "start": "2024-08-01", "end": "2024-08-15", "type": "gwp_threshold"},
    {"name": "lucky_box",       "start": "2025-05-22", "end": "2025-05-31", "type": "closing_promo"},
    {"name": "closing_50",      "start": "2025-06-01", "end": "2099-12-31", "type": "closing_discount"},
]
EVENTS_DF = expand_events(JEJU_EVENTS)


def predict_next_week(store_name: str, base_date: date, horizon: int, model: Any, transformer: Any) -> pd.DataFrame:
    """
    FastAPI /forecast 엔드포인트에서 호출되는 핵심 예측 함수.
    
    Args:
        store_name (str): 지점 이름 (여기서는 "제주애월점" 고정).
        base_date (date): 예측을 시작할 기준 날짜 (과거 데이터의 마지막 날짜 다음 날).
        horizon (int): 예측 기간 (일수, 보통 7일).
        
    Returns:
        pd.DataFrame: '날짜', '상품명', '예측수량', '주문량_ceil' 컬럼을 포함하는 예측 결과.
    """
    print(f"--- [INFO] 제주애월점 예측 시작: {base_date}부터 {horizon}일 ---")
    
    # ===== 0. 데이터 로드 및 전처리 =====
    
    # 파일 경로 하드코딩 
    POS_FILE = "aewol_POS.csv" 
    POS_PATH = DATA_DIR / POS_FILE

    if not POS_PATH.exists():
        raise FileNotFoundError(f"POS 데이터 파일이 없습니다: {POS_PATH}")

    df = pd.read_csv(POS_PATH, encoding="utf-8-sig")

    df = df.rename(columns={"일자": "날짜"})
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["수량"] = pd.to_numeric(df["수량"], errors="coerce")

    # 결측/이상치 제거 및 정렬
    df = df.dropna(subset=["날짜", "상품명", "수량"]).sort_values("날짜").reset_index(drop=True)
    df = df.groupby(["날짜","상품명"], as_index=False)["수량"].sum()
    
    # 예측 기간 설정 (FastAPI 요청 기반)
    forecast_start = base_date.isoformat()
    forecast_end = (base_date + timedelta(days=horizon - 1)).isoformat()


    # ===== 1. 학습용 피처 생성 (원본 로직 유지) =====
    
    # 1) 캘린더 피처
    df_train = build_calendar_features(df, date_col="날짜", holiday_window=HOLIDAY_WINDOW)

    # 2) 이벤트 피처
    df_train = attach_events(df_train, EVENTS_DF)

    # 3) 과거 날씨 병합
    hist_start = pd.to_datetime(df_train["날짜"].min()).date().isoformat()
    hist_end   = pd.to_datetime(df_train["날짜"].max()).date().isoformat()
    weather_hist = fetch_weather_daily(LAT, LON, hist_start, hist_end, timezone=TIMEZONE)

    df_train = safe_merge(df_train, weather_hist, on="날짜", how="left")
    
    # 같은 '날짜' 그룹 내부에서만 결측 채움 (날씨)
    weather_cols = ["temp_max","precip"]
    if set(weather_cols).issubset(df_train.columns):
        df_train[weather_cols] = df_train.groupby("날짜")[weather_cols].transform("max")
    df_train = df_train.sort_values(["날짜","상품명"]).reset_index(drop=True)
    
    
    # ===== 2. 예측 그리드 생성 및 피처 부착 =====
    
    # 0) 제품 목록 정의 (학습 데이터 기반)
    products = sorted(df_train["상품명"].dropna().unique().tolist())
    
    # 1) 예측 그리드
    df_pred = make_prediction_grid(products, forecast_start, forecast_end)
    
    # 2) 캘린더/이벤트 피처
    df_pred = build_calendar_features(df_pred, date_col="날짜", holiday_window=HOLIDAY_WINDOW)
    df_pred = attach_events(df_pred, EVENTS_DF)
    
    # 3) 예측 구간 날씨
    weather_fore = fetch_weather_daily(LAT, LON, forecast_start, forecast_end, timezone=TIMEZONE)
    
    # 4) 안전 병합 및 결측 채움 (날씨)
    df_pred = safe_merge(df_pred, weather_fore, on="날짜", how="left")
    df_pred = df_pred.sort_values(["날짜","상품명"]).reset_index(drop=True)
    if set(weather_cols).issubset(df_pred.columns):
        df_pred[weather_cols] = df_pred.groupby("날짜")[weather_cols].transform("max")

    # 5) 이벤트 결측 처리 (이벤트 피처가 전부 0일 때 컬럼이 없을 수 있음)
    event_cols = [c for c in df_pred.columns if c.startswith("event_") and c not in ["event_any"]]
    if event_cols:
        df_pred[event_cols] = df_pred[event_cols].fillna(0).astype(int)
    df_pred["event_any"] = df_pred["event_any"].fillna(0).astype(int)


    # ===== 3. HGBR 학습 및 예측 (원본 로직 유지) =====
    
    # 1) 피처 세트 정의
    base_feats = ['상품명','day','month','is_holiday','is_holiday_window',
                    'holiday_weight','temp_max','precip','event_any']
    
    # df_train에 존재하는 event_cols만 포함하도록 업데이트
    final_event_cols = [c for c in df_train.columns if c.startswith('event_') and c not in ['event_any']]
    feature_cols = list(set(base_feats + final_event_cols) & set(df_train.columns))

    TARGET   = '수량'
    
    # 2) 데이터 준비
    train_use = df_train.dropna(subset=feature_cols + [TARGET]).copy()
    X_train = train_use[feature_cols].copy()
    y_train = train_use[TARGET].clip(lower=0).astype(float)
    X_pred  = df_pred[feature_cols].copy()

    # 3) Column Transformer 설정 및 변환
    # ct = setup_column_transformer(X_train, feature_cols)
    #X_train_tr = ct.transform(X_train)
    X_pred_tr  = transformer.transform(X_pred)

    if hasattr(X_pred_tr, "toarray"):
       # X_train_tr = X_train_tr.toarray()
        X_pred_tr  = X_pred_tr.toarray()

    # 4) 모델 학습
    # model = HistGradientBoostingRegressor(
    #    loss='poisson', learning_rate=0.05, max_depth=None, max_iter=1500,
    #    l2_regularization=0.0, early_stopping=False, random_state=42
    # )
    # model.fit(X_train_tr, y_train)

    # 5) 예측
    pred = np.maximum(model.predict(X_pred_tr), 0)

    # ===== 4. 결과 DataFrame 반환 =====
    df_out = df_pred[['날짜','상품명']].copy()
    df_out['예측수량']    = pred
    df_out['주문량_ceil'] = np.ceil(pred).astype(int)
    
    df_out['날짜'] = df_out['날짜'].dt.strftime('%Y-%m-%d')
    
    print(f"--- [INFO] 제주애월점 예측 완료. 레코드 수: {len(df_out)} ---")
    
    return df_out.sort_values(['날짜', '상품명'])

# (참고: API 호출 환경에서는 이 main 블록이 실행되지 않음. 테스트용임!)
if __name__ == "__main__":
    # 독립 테스트를 위한 블록
    base_date_test = date(2025, 10, 24)
    result_df = predict_next_week("제주애월점", base_date_test, 7)
    print("\n[독립 테스트 결과]")
    print(result_df.head())
