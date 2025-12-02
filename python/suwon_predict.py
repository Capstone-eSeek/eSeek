import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Scikit-learn 및 예측 모델
from catboost import CatBoostRegressor, Pool
from pandas.api.types import is_categorical_dtype

# ⭐️ 외부 유틸리티 함수 import ⭐️
from python.prediction_utils import (
    build_calendar_features,
    attach_events,
    fetch_weather_daily,
    make_prediction_grid,
    safe_merge,
    expand_events,
    TIMEZONE
)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ===== 수원점 설정 =====
LAT, LON = 37.28, 127.05  # 수원 타임빌라스 좌표(대략)
HOLIDAY_WINDOW = 2

# 요일 카테고리 고정 (피처 생성 시 사용)
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ========= 수원점 이벤트 정의 =========
SUWON_EVENTS = [
    # 1) 2024.07.05 오픈 행사
    {"name": "suwon_opening_free_glazed", "start": "2024-07-05", "end": "2024-07-05", "type": "free_glazed_777"},
    {"name": "suwon_opening_pvc_bag",     "start": "2024-07-05", "end": "2024-07-07", "type": "gwp_threshold_50k"},
    # 2) 2024.10.24 타임빌라스 그랜드 오픈 (글레이즈 300개 무료)
    {"name": "suwon_timevillas_grand_open", "start": "2024-10-24", "end": "2024-10-24", "type": "free_glazed_300"},
    # 3) 2025.05.22~06.03 럭키박스 (랜덤 제공)
    {"name": "lucky_box", "start": "2025-05-22", "end": "2025-06-03", "type": "lucky_box_random"},
    # 4) 2025.06.04~(현재) 도넛 50% 할인 (마감 1시간 한정)
    {"name": "closing_50", "start": "2025-06-04", "end": "2099-12-31", "type": "closing_discount_hour_only"},
]
EVENTS_DF = expand_events(SUWON_EVENTS)


# ========= 상품 태그 및 상호작용 피처 함수 (원본 코드 기반) =========

def build_product_tags(df: pd.DataFrame, name_col="상품명") -> pd.DataFrame:
    """상품명 기반으로 태그 플래그 생성"""
    out = df.copy()
    if name_col not in out.columns:
        return out
    s = out[name_col].astype(str).str.lower()
    def has(p): return s.str.contains(p, regex=True, na=False)

    out["is_donut"]      = has(r"도넛|donut").astype(int)
    out["is_glazed"]     = has(r"글레이즈|glaze").astype(int)
    out["is_americano"]  = has(r"아메리카노|americano").astype(int)
    out["is_beverage"]   = has(r"라떼|cold ?brew|콜드브루|에이드|주스|티|coffee|커피|americano|아메리카노").astype(int)
    out["is_round_mini"] = has(r"라운드\s*미니|round\s*mini|미니").astype(int)
    
    # 상호작용에 사용될 수 있는 필드
    out["is_drink"]      = out["is_beverage"] 
    out["is_popular"]    = out["is_glazed"] 
    out["is_set"]        = 0
    return out

def add_event_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """수원점 전용 이벤트와 상품 태그의 상호작용 피처를 생성"""
    out = df.copy()

    # 50% 할인: 마감 1시간 한정 → 일별 합계에선 약하게 가중치로 근사
    if "event_closing_50" in out.columns:
        out["event_closing_50_hour_weight"] = out["event_closing_50"] * 0.35  # 원본 코드 가중치 유지
        out["evt50_x_donut"] = out["event_closing_50"] * out.get("is_donut", 0)

    # 오픈/그랜드 오픈: 글레이즈 무료 제공 → 글레이즈 강화
    if "event_suwon_opening_free_glazed" in out.columns:
        out["evt_open_x_glazed"] = out["event_suwon_opening_free_glazed"] * out.get("is_glazed", 0)
    if "event_suwon_timevillas_grand_open" in out.columns:
        out["evt_timevillas_x_glazed"] = out["event_suwon_timevillas_grand_open"] * out.get("is_glazed", 0)

    # PVC 백 증정: 전반적 수요 신호
    if "event_suwon_opening_pvc_bag" in out.columns:
        out["evt_pvc_any"] = out["event_suwon_opening_pvc_bag"]

    # 럭키박스(랜덤 제공): 특정 SKU 집중이 아니라 도넛 전반 상승으로 가정
    if "event_lucky_box" in out.columns:
        out["evt_lucky_random_x_donut"] = out["event_lucky_box"] * out.get("is_donut", 0)

    return out

def _finalize_feature_block(df_out: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임을 최종 모델 입력 형식에 맞게 정리"""
    
    weather_cols = ["temp_max","precip"]
    if set(weather_cols).issubset(df_out.columns):
        df_out[weather_cols] = df_out.groupby("날짜")[weather_cols].transform("max")

    if "event_any" not in df_out.columns:
        df_out["event_any"] = 0
    ev_cols = [c for c in df_out.columns if c.startswith("event_") and c not in ["event_any"]]
    if ev_cols:
        df_out[ev_cols] = df_out[ev_cols].fillna(0).astype(int)

    if "day" in df_out.columns:
        df_out["day"] = pd.Categorical(df_out["day"], categories=WEEKDAY_ORDER, ordered=True)
    
    # CatBoost를 위해 모든 범주형 컬럼을 문자열로 통일
    cat_cols_to_str = ['상품명', 'day'] + ev_cols 
    for c in cat_cols_to_str:
        if c in df_out.columns:
            df_out[c] = df_out[c].astype(str)

    return df_out.sort_values(["날짜","상품명"]).reset_index(drop=True)

def make_features(df_in: pd.DataFrame, start_date: str, end_date: str, mode="train") -> pd.DataFrame:
    """학습 및 예측 데이터에 공통 피처를 부착"""
    df_out = build_calendar_features(df_in, date_col="날짜", holiday_window=HOLIDAY_WINDOW)
    df_out = attach_events(df_out, EVENTS_DF)          
    df_out = build_product_tags(df_out)                
    df_out = add_event_interactions(df_out)            
    df_out["day"] = pd.to_datetime(df_out["날짜"]).dt.day_name()
    
    # 날씨 데이터 로드
    weather = fetch_weather_daily(LAT, LON, start_date, end_date, timezone=TIMEZONE)
    df_out = safe_merge(df_out, weather, on="날짜", how="left")

    return _finalize_feature_block(df_out)


# ----------------------------------------------------
# predict_next_week: API 호출용 메인 예측 함수 
# ----------------------------------------------------
def predict_next_week(store_name: str, base_date: date, horizon: int, model: Any, transformer: Any ) -> pd.DataFrame:
    """
    CatBoost 모델을 사용하여 수원점의 다음 주 수요를 예측합니다.
    (수원점 모델은 CatBoost 일반(log1p 없음)으로 가정하고 구현)
    """
    print(f"--- [INFO] 수원타임빌리지점 예측 시작: {base_date}부터 {horizon}일 ---")
    
    # ===== 0. 데이터 로드 및 전처리 =====
    POS_FILE = "suwon_POS.csv" 
    POS_PATH = DATA_DIR / POS_FILE

    if not POS_PATH.exists():
        raise FileNotFoundError(f"POS 데이터 파일이 없습니다: {POS_PATH}")

    df = pd.read_csv(POS_PATH, encoding="utf-8-sig")

    df = df.rename(columns={"일자": "날짜"})
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["수량"] = pd.to_numeric(df["수량"], errors="coerce")
    df = df.dropna(subset=["날짜", "상품명", "수량"]).sort_values("날짜").reset_index(drop=True)
    df = df.groupby(["날짜","상품명"], as_index=False)["수량"].sum()
    
    # 예측 기간 설정
    forecast_start = base_date.isoformat()
    forecast_end = (base_date + timedelta(days=horizon - 1)).isoformat()
    
    # 학습 기간 설정 (과거 데이터 전체)
    hist_start = pd.to_datetime(df["날짜"].min()).date().isoformat()
    hist_end   = pd.to_datetime(df["날짜"].max()).date().isoformat()

    
    # ===== 1. 학습용 피처 생성 =====
    df_train = make_features(df, hist_start, hist_end, mode="train")
    
    # 2) 제품 목록 정의 (학습 데이터 기반)
    products = sorted(df_train["상품명"].dropna().unique().tolist())
    
    # ===== 2. 예측 그리드 생성 및 피처 부착 =====
    df_pred = make_prediction_grid(products, forecast_start, forecast_end)
    df_pred = make_features(df_pred, forecast_start, forecast_end, mode="predict")
    
    
    # ===== 3. CatBoost 학습 및 예측 =====
    
    # 1) 피처 세트 정의 (원본 코드 기반 + 중복 제거)
    base_feats = ['상품명','day','month','is_holiday','is_holiday_window','holiday_weight','temp_max','precip', 'event_any']
    
    # 상호작용 및 이벤트 플래그 피처 
    event_flag_cols = [c for c in df_train.columns if c.startswith('event_') and c not in ['event_any', 'event_closing_50_hour_weight']]
    event_days_cols = [c for c in df_train.columns if c.startswith('event_') and c.endswith('_days_since_start')]
    
    interaction_cols = [
        "is_donut","is_glazed","is_beverage","is_americano","is_round_mini",
        "evt_open_x_glazed", "evt_timevillas_x_glazed", "evt_pvc_any",
        "evt_lucky_random_x_donut", "evt50_x_donut", "event_closing_50_hour_weight"
    ]
    
    # 중복 제거 및 최종 피처 목록 구성
    feature_candidates_set = set(base_feats) | set(event_flag_cols) | set(event_days_cols) | set(interaction_cols)
    feature_cols = sorted(list(feature_candidates_set & set(df_train.columns)))
    
    # ⭐️ CatBoost 정렬 보장: 순서 중요한 컬럼을 맨 앞으로 ⭐️
    order_dependent = ['상품명', 'day', 'month'] 
    final_feature_cols = [c for c in order_dependent if c in feature_cols]
    final_feature_cols.extend([c for c in feature_cols if c not in order_dependent])
    feature_cols = final_feature_cols

    # 2) 데이터 준비 (로그 변환 없음)
    TARGET = '수량'
    
    # X_train = df_train[feature_cols].copy()
    # y_train = df_train[TARGET].astype(float).clip(lower=0) # ⭐️ 로그 변환 없음 ⭐️

    X_pred  = df_pred[feature_cols].copy()

    # 3) 범주형 지정 
    cat_cols = [c for c in ['상품명','day'] if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    
    # 4) Pool 구성
    # train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    pred_pool  = Pool(X_pred,           cat_features=cat_idx)
    
    # 5) 모델 정의/학습 
    # model = CatBoostRegressor(
    #     loss_function='RMSE', eval_metric='R2', depth=8, learning_rate=0.05,
    #     iterations=5000, l2_leaf_reg=3.0, random_seed=42, verbose=0
    # )
    # model.fit(train_pool)
    
    # 6) 예측 (역변환 없음 + 음수 방지)
    pred = model.predict(pred_pool)
    pred = np.maximum(pred, 0.0)

    # ===== 4. 결과 DataFrame 반환 =====
    df_out = df_pred[['날짜','상품명']].copy()
    df_out['예측수량']    = pred
    df_out['주문량_ceil'] = np.ceil(pred).astype(int)
    
    # API 응답을 위해 날짜 포맷팅
    df_out['날짜'] = df_out['날짜'].dt.strftime('%Y-%m-%d')
    
    print(f"--- [INFO] 수원타임빌리지점 예측 완료. 레코드 수: {len(df_out)} ---")
    
    return df_out.sort_values(['날짜', '상품명'])

# (참고: API 호출 환경에서는 이 main 블록이 실행되지 않음. 테스트용임!)
if __name__ == "__main__":
    # 독립 테스트를 위한 블록
    base_date_test = date(2025, 10, 24)
    result_df = predict_next_week("수원점", base_date_test, 7)
    print("\n[독립 테스트 결과]")
    print(result_df.head())
