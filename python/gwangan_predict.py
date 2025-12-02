import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Scikit-learn 및 예측 모델
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pandas.api.types import is_categorical_dtype

# 외부 유틸리티 함수 import
from prediction_utils import (
    build_calendar_features,
    attach_events,
    fetch_weather_daily,
    make_prediction_grid,
    safe_merge,
    expand_events,
    setup_column_transformer, # (CatBoost는 OHE 대신 자체 인코딩 사용하므로 필요 없음)
    TIMEZONE
)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ===== 광안리점 설정 =====
LAT, LON = 35.15, 129.11  # 부산 광안리 좌표(대략)
HOLIDAY_WINDOW = 2

# 요일 카테고리 고정 (피처 생성 시 사용)
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ========= 광안리점 전체 이벤트 정의 =========
GWANGAN_EVENTS = [
    # 현장 행사
    {"name": "gwangalli_americano_gwp_20k", "start": "2023-07-10", "end": "2023-09-10", "type": "gwp_threshold_20k_americano"},
    {"name": "gwangalli_fireworks_ecobag_gwp", "start": "2023-11-03", "end": "2023-11-05", "type": "gwp_threshold_20k_ecobag"},
    {"name": "gwangalli_fireworks_picnicmat_buy_americano_free", "start": "2023-11-04", "end": "2023-11-04", "type": "buy_picnic_mat_get_americano"},
    {"name": "gwangalli_ecobag_gwp_20k", "start": "2023-12-18", "end": "2023-12-22", "type": "gwp_threshold_20k_ecobag"},
    {"name": "gwangalli_wdd_free_glazed_per_person", "start": "2024-06-07", "end": "2024-06-07", "type": "world_donut_day_free_glazed_per_person"},
    {"name": "gwangalli_wdd_free_lucky_donut", "start": "2024-06-07", "end": "2024-06-07", "type": "free_lucky_donut"},
    {"name": "gwangalli_picnic_mat_gwp_50k", "start": "2024-08-01", "end": "2024-08-15", "type": "gwp_threshold_50k_picnic_mat"},
    {"name": "gwangalli_picnic_mat_50pct_with_donut_or_drink", "start": "2024-08-01", "end": "2024-08-15", "type": "picnic_mat_50pct_with_donut_or_drink"},
    {"name": "gwangalli_fireworks_2024_blanket_gwp_50k", "start": "2024-11-09", "end": "2024-11-09", "type": "gwp_threshold_50k_blanket"},
    {"name": "gwangalli_lucky_box_random", "start": "2025-05-22", "end": "2025-06-03", "type": "lucky_box_random"},
    {"name": "gwangalli_lucky_box_choice", "start": "2025-06-04", "end": "2099-12-31", "type": "lucky_box_choice"},
    {"name": "gwangalli_closing_50", "start": "2025-06-04", "end": "2099-12-31", "type": "closing_discount_hour_only"},
    # 배달 행사
    {"name": "gwangalli_delivery_coldbrew_gwp_50k", "start": "2024-12-24", "end": "2024-12-25", "type": "delivery_gwp_threshold_50k_coldbrew"},
]
EVENTS_DF = expand_events(GWANGAN_EVENTS)


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
    out["is_drink"]      = out["is_beverage"] # is_drink는 is_beverage와 동일하게 처리 (CatBoost 모델에 필요)
    
    # 임시 is_popular/is_set 필드 (모델 학습 시 사용되었다면)
    out["is_popular"] = out["is_glazed"] # 임시 휴리스틱
    out["is_set"]     = 0
    return out

def add_event_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """이벤트와 상품 태그의 상호작용 피처를 생성"""
    out = df.copy()
    
    def any_flag(cols):
        present = [c for c in cols if c in out.columns]
        if not present: return pd.Series(0, index=out.index)
        return out[present].sum(axis=1).clip(0, 1)

    # ---------- 마감 1시간 50% 할인 ----------
    closing50_any = any_flag(["event_gwangalli_closing_50"])
    out["event_closing_50_any"] = closing50_any
    out["event_closing_50_hour_weight"] = closing50_any * 0.35 # 원본 코드의 가중치 유지
    out["evt50_x_donut"] = closing50_any * out.get("is_donut", 0)

    # ---------- 글레이즈 푸시 계열 ----------
    glazed_push_any = any_flag(["event_gwangalli_wdd_free_glazed_per_person"])
    if "is_glazed" in out.columns:
        out["evt_glazed_push"] = glazed_push_any * out["is_glazed"]

    # ---------- GWP (금액 기준 증정) ----------
    gwp20_any = any_flag(["event_gwangalli_fireworks_ecobag_gwp", "event_gwangalli_ecobag_gwp_20k", "event_gwangalli_americano_gwp_20k"])
    out["evt_gwp20_any"] = gwp20_any

    gwp50_any = any_flag(["event_gwangalli_picnic_mat_gwp_50k", "event_gwangalli_fireworks_2024_blanket_gwp_50k", "event_gwangalli_delivery_coldbrew_gwp_50k"])
    out["evt_gwp50_any"] = gwp50_any

    # ---------- 럭키박스 ----------
    lucky_random_any = any_flag(["event_gwangalli_lucky_box_random"])
    out["evt_lucky_random_x_donut"] = lucky_random_any * out.get("is_donut", 0)

    lucky_choice_any = any_flag(["event_gwangalli_lucky_box_choice"])
    base_choice_pref = (out.get("is_popular", 0) if "is_popular" in out.columns 
                        else (out.get("is_glazed", 0) * 0.7 + out.get("is_donut", 0) * 0.3))
    out["evt_lucky_choice_pref"] = lucky_choice_any * base_choice_pref

    # ---------- 피크닉 매트 50% 할인 ----------
    picnic50_any = any_flag(["event_gwangalli_picnic_mat_50pct_with_donut_or_drink"])
    if "is_drink" in out.columns:
        out["evt_picnic50_x_drink"] = picnic50_any * out["is_drink"]
    else:
        out["evt_picnic50_any_weak"] = picnic50_any * 0.2

    # ---------- 피크닉 매트 구매 시 아메리카노 증정 ----------
    buy_mat_get_coffee_any = any_flag(["event_gwangalli_fireworks_picnicmat_buy_americano_free"])
    if "is_drink" in out.columns:
        out["evt_buy_mat_get_coffee_x_drink"] = buy_mat_get_coffee_any * out["is_drink"]
    else:
        out["evt_buy_mat_get_coffee_any_weak"] = buy_mat_get_coffee_any * 0.2

    return out

def _finalize_feature_block(df_out: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임을 최종 모델 입력 형식에 맞게 정리"""
    
    # 날씨 결측 채움 (같은 날짜 안에서)
    weather_cols = ["temp_max","precip"]
    if set(weather_cols).issubset(df_out.columns):
        df_out[weather_cols] = df_out.groupby("날짜")[weather_cols].transform("max")

    # 이벤트 기본값/타입 정리
    if "event_any" not in df_out.columns:
        df_out["event_any"] = 0
    ev_cols = [c for c in df_out.columns if c.startswith("event_") and c not in ["event_any"]]
    if ev_cols:
        df_out[ev_cols] = df_out[ev_cols].fillna(0).astype(int)

    # 요일 카테고리 고정
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
    df_out["day"] = pd.to_datetime(df_out["날짜"]).dt.day_name() # 요일 재계산
    
    # 날씨 데이터 로드
    weather = fetch_weather_daily(LAT, LON, start_date, end_date, timezone=TIMEZONE)
    df_out = safe_merge(df_out, weather, on="날짜", how="left")

    return _finalize_feature_block(df_out)


# ========= 예측 함수 (API 호출용) =========
def predict_next_week(store_name: str, base_date: date, horizon: int) -> pd.DataFrame:
    """
    CatBoost log1p 모델을 사용하여 광안리점의 다음 주 수요를 예측합니다.
    """
    print(f"--- [INFO] 부산광안리점 예측 시작: {base_date}부터 {horizon}일 ---")
    
    # ===== 0. 데이터 로드 및 전처리 =====
    POS_FILE = "gwangan_POS.csv" 
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
    
    # 1) 피처 세트 정의 (원본 코드 기반)
    # df_train에 존재하는 event_cols 및 extra_cols 모두 포함
    base_feats = ['상품명','day','month','is_holiday','is_holiday_window','holiday_weight','temp_max','precip','event_any']
    
    # event_*_days_since_start 컬럼을 포함한 모든 이벤트 관련 피처
    event_flag_cols = [c for c in df_train.columns if c.startswith('event_') and not c.endswith('_days_since_start')]
    event_days_cols = [c for c in df_train.columns if c.startswith('event_') and c.endswith('_days_since_start')]
    
    extra_cols = [
        "is_donut","is_glazed","is_beverage","is_americano","is_round_mini","is_drink", "is_popular", "is_set", # 상품 태그
        "evt_glazed_push", "evt_gwp20_any", "evt_gwp50_any", "evt_lucky_random_x_donut", "evt_lucky_choice_pref",
        "evt_picnic50_x_drink", "evt_picnic50_any_weak", "evt_buy_mat_get_coffee_x_drink", "evt_buy_mat_get_coffee_any_weak",
        "evt50_x_donut", "event_closing_50_hour_weight" # 상호작용 피처
    ]
    
    # base_feats에서 event_any를 제외하고 시작
    base_feats_clean = [c for c in base_feats if c != 'event_any'] 
    
    # 모든 피처 후보 통합
    feature_candidates_set = set(base_feats_clean) | set(event_flag_cols) | set(event_days_cols) | set(extra_cols)
    
    # 누락된 event_any를 명시적으로 추가 (CatBoost는 순서가 중요하지 않지만, CatBoostPool이 기대함)
    if 'event_any' in df_train.columns:
        feature_candidates_set.add('event_any')

    # df_train에 존재하는 최종 피처 목록 (원본 순서 유지 없이 set 기반)
    feature_cols = sorted(list(feature_candidates_set & set(df_train.columns)))
    
    # CatBoost는 feature_cols의 순서를 참조하므로, 순서가 중요한 컬럼을 맨 앞으로 뺍니다.
    order_dependent = ['상품명', 'day', 'month', 'is_holiday', 'is_holiday_window']
    final_feature_cols = [c for c in order_dependent if c in feature_cols]
    final_feature_cols.extend([c for c in feature_cols if c not in order_dependent])
    
    feature_cols = final_feature_cols
    
    # 2) 데이터 준비 및 로그 변환
    TARGET = '수량'
    
    X_train = df_train[feature_cols].copy()
    y_train = np.log1p(df_train[TARGET].astype(float).clip(lower=0)) # ⭐️ log1p 변환 ⭐️

    X_pred  = df_pred[feature_cols].copy() # 예측 그리드도 동일 피처 사용

    # 3) 범주형 지정 (CatBoost는 OHE 불필요)
    cat_cols = [c for c in ['상품명','day'] if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    
    # 4) Pool 구성
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    pred_pool  = Pool(X_pred,           cat_features=cat_idx)
    
    # 5) 모델 정의/학습 (원본 설정 유지)
    model = CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='R2',
        depth=8,
        learning_rate=0.05,
        iterations=5000,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0 # API 환경에서는 로그 출력하지 않음
    )
    model.fit(train_pool)
    
    # 6) 예측 (역변환 + 음수 방지)
    pred_log = model.predict(pred_pool)
    pred = np.expm1(pred_log) # ⭐️ np.expm1으로 복원 ⭐️
    pred = np.maximum(pred, 0.0)

    # ===== 4. 결과 DataFrame 반환 =====
    df_out = df_pred[['날짜','상품명']].copy()
    df_out['예측수량']    = pred
    df_out['주문량_ceil'] = np.ceil(pred).astype(int)
    
    # API 응답을 위해 날짜 포맷팅
    df_out['날짜'] = df_out['날짜'].dt.strftime('%Y-%m-%d')
    
    print(f"--- [INFO] 부산광안리점 예측 완료. 레코드 수: {len(df_out)} ---")
    
    return df_out.sort_values(['날짜', '상품명'])

# (참고: API 호출 환경에서는 이 main 블록이 실행되지 않음. 테스트용임!)
if __name__ == "__main__":
    # 독립 테스트를 위한 블록
    base_date_test = date(2025, 10, 24)
    result_df = predict_next_week("부산광안리점", base_date_test, 7)
    print("\n[독립 테스트 결과]")
    print(result_df.head())