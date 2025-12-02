import pandas as pd
import numpy as np
import requests
import holidays
from datetime import timedelta, date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, List

# 전역 설정 (필요에 따라 각 지점 파일에서 오버라이드 가능)
TIMEZONE = "Asia/Seoul"

# ===== 날짜/공휴일 관련 유틸리티 =====
def _to_date(x):
    """Pandas Timestamp나 문자열을 datetime.date 객체로 변환"""
    return pd.to_datetime(x).date()

def _holiday_dates(dmin, dmax):
    """기간 내 한국 공휴일 목록 반환"""
    yrs = range(dmin.year, dmax.year + 1)
    hol = holidays.KR(years=yrs)
    return sorted({d for d in hol if dmin <= d <= dmax})

def build_calendar_features(df, date_col="날짜", holiday_window=2):
    """캘린더 및 공휴일 피처 생성"""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day_of_week"] = out[date_col].dt.dayofweek # 0=월, 6=일
    out["day"] = out[date_col].dt.day_name()
    
    dmin = _to_date(out[date_col].min()) - timedelta(days=holiday_window+7)
    dmax = _to_date(out[date_col].max()) + timedelta(days=holiday_window+7)
    holiday_days = _holiday_dates(dmin, dmax)

    # 공휴일 피처 초기화: holiday_days 유무와 관계없이 생성 보장 
    out["is_holiday"] = 0
    out["days_to_nearest_holiday"] = np.nan
    out["is_holiday_window"] = 0
    
    # 삼각 가중치 기본값
    out["holiday_weight"] = 0.0

    if not holiday_days:
        return out

    def signed_min_dist(ts):
        d = _to_date(ts)
        diffs = [(d - h).days for h in holiday_days]
        j = int(np.argmin(np.abs(diffs)))
        return diffs[j]

    signed_dist = out[date_col].map(signed_min_dist)
    out["days_to_nearest_holiday"] = signed_dist
    out["is_holiday"] = (out["days_to_nearest_holiday"] == 0).astype(int)
    out["is_holiday_window"] = (out["days_to_nearest_holiday"].abs() <= holiday_window).astype(int)

    # 삼각 가중치: |d|=0 → 1.0, |d|=window → 0.0
    w = (holiday_window - out["days_to_nearest_holiday"].abs()).clip(lower=0) / max(holiday_window, 1)
    out["holiday_weight"] = w

    return out

# ===== 이벤트 관련 유틸리티 =====
def expand_events(events: List[Dict]) -> pd.DataFrame:
    """이벤트 리스트를 날짜별 DataFrame으로 확장"""
    rows = []
    for ev in events:
        name = ev["name"]; ev_type = ev.get("type","")
        cur = _to_date(ev["start"]); end = _to_date(ev["end"])
        while cur <= end:
            rows.append({"날짜": pd.to_datetime(cur), "event_name": name, "event_type": ev_type})
            cur += timedelta(days=1)
    return pd.DataFrame(rows).sort_values("날짜").reset_index(drop=True)

def build_event_features(events_df, start_date=None, end_date=None):
    """이벤트 DataFrame을 와이드 포맷 피처로 변환"""
    # 원본 코드 로직 유지 (wide pivot table 생성)
    df_ev = events_df.copy()
    if df_ev.empty:
        return pd.DataFrame(columns=["날짜","event_any"])
        
    df_ev["날짜"] = pd.to_datetime(df_ev["날짜"])
    if start_date is not None:
        df_ev = df_ev[df_ev["날짜"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_ev = df_ev[df_ev["날짜"] <= pd.to_datetime(end_date)]
        
    wide = (
        df_ev.assign(val=1)
             .pivot_table(index="날짜", columns="event_name", values="val", aggfunc="max", fill_value=0)
             .rename(columns=lambda c: f"event_{c}")
             .reset_index()
    )
    wide["event_any"] = (wide.drop(columns=["날짜"]).sum(axis=1) > 0).astype(int)
    return wide

def attach_events(df_in: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """이벤트 피처를 메인 데이터프레임에 병합"""
    base = df_in.copy()
    base["날짜"] = pd.to_datetime(base["날짜"])
    
    # 1. 와이드 이벤트 피처 생성 및 병합
    wide = build_event_features(events_df)
    if wide.empty:
        base["event_any"] = 0
        return base
        
    out = base.merge(wide, on="날짜", how="left")
    event_cols = [c for c in out.columns if c.startswith("event_") and c not in ["event_any"]]
    
    # 2. 결측 처리 및 경과일수 피처 (원본 코드 로직 유지)
    out[event_cols] = out[event_cols].fillna(0).astype(int)
    out["event_any"] = out["event_any"].fillna(0).astype(int)
    
    # 경과일수 (Days Since Start)
    for c in event_cols:
        name = c.replace("event_", "")
        first_day = events_df.loc[events_df["event_name"]==name, "날짜"].min()
        if pd.isna(first_day): continue
        first_day = pd.to_datetime(first_day)
        col = f"{c}_days_since_start"
        out[col] = (out["날짜"] - first_day).dt.days.clip(lower=0)
        out.loc[out[c]==0, col] = 0
        
    return out

# ===== 데이터 처리/병합 유틸리티 =====
def make_prediction_grid(products: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """예측 기간 동안의 날짜×상품 그리드를 생성"""
    dates = pd.date_range(start_date, end_date, freq="D")
    return pd.MultiIndex.from_product([dates, products], names=["날짜","상품명"]).to_frame(index=False)

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how='left') -> pd.DataFrame:
    """컬럼 중복을 안전하게 처리하며 병합"""
    if right is None or len(right)==0:
        return left
    dup_cols = set(left.columns).intersection(set(right.columns)) - {on}
    right_ren = right.rename(columns={c: f"{c}__r" for c in dup_cols})
    merged = left.merge(right_ren, on=on, how=how)
    for c in dup_cols:
        rc = f"{c}__r"
        if rc in merged:
            merged[c] = merged[rc].where(~merged[rc].isna(), merged[c])
            merged.drop(columns=[rc], inplace=True)
    return merged

# ===== 날씨 API 유틸리티 (원본 코드 유지) =====
def _weather_df_from_payload(payload):
    if not payload or "daily" not in payload:
        return pd.DataFrame(columns=["날짜","temp_max","precip"])
    d = payload["daily"]
    return pd.DataFrame({
        "날짜": pd.to_datetime(d.get("time", [])),
        "temp_max": d.get("temperature_2m_max", [np.nan]*len(d.get("time", []))),
        "precip":   d.get("precipitation_sum", [np.nan]*len(d.get("time", []))),
    })

def fetch_weather_daily(lat: float, lon: float, start_date: str, end_date: str, timezone=TIMEZONE) -> pd.DataFrame:
    """과거와 미래 날씨를 Open-Meteo API에서 가져옴 (자동 분기 처리)"""
    sd = pd.to_datetime(start_date).date()
    ed = pd.to_datetime(end_date).date()

    today = pd.Timestamp.now(tz=timezone).date()
    hist_last = min(ed, today - timedelta(days=2))

    frames = []

    # 1) 과거(ERA5)
    if sd <= hist_last:
        url_hist = "https://archive-api.open-meteo.com/v1/era5"
        params_hist = {
            "latitude": lat, "longitude": lon,
            "start_date": sd.isoformat(), "end_date": hist_last.isoformat(),
            "daily": ["temperature_2m_max","precipitation_sum"],
            "timezone": timezone,
        }
        try:
            r = requests.get(url_hist, params=params_hist, timeout=30)
            r.raise_for_status()
            frames.append(_weather_df_from_payload(r.json()))
        except Exception as e:
            print(f"Warning: Failed to fetch historical weather ({sd} to {hist_last}): {e}")

    # 2) 미래 또는 잔여 구간(forecast)
    fore_start = max(sd, hist_last + timedelta(days=1))
    if fore_start <= ed:
        url_fore = "https://api.open-meteo.com/v1/forecast"
        params_fore = {
            "latitude": lat, "longitude": lon,
            "start_date": fore_start.isoformat(), "end_date": ed.isoformat(),
            "daily": ["temperature_2m_max","precipitation_sum"],
            "timezone": timezone,
        }
        try:
            r = requests.get(url_fore, params=params_fore, timeout=30)
            r.raise_for_status()
            frames.append(_weather_df_from_payload(r.json()))
        except Exception as e:
            print(f"Warning: Failed to fetch forecast weather ({fore_start} to {ed}): {e}")

    if not frames:
        return pd.DataFrame(columns=["날짜","temp_max","precip"])

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["날짜"]).sort_values("날짜")
    return out

# ===== 피처 변환기 유틸리티 =====
def setup_column_transformer(df_train: pd.DataFrame, feature_cols: List[str]):
    """학습 데이터 기반으로 ColumnTransformer (OHE)를 설정"""
    cat_cols = [c for c in ['상품명','day'] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    except TypeError: # 구 버전 sklearn 호환성
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)

    transformers = []
    if cat_cols:
        transformers.append(('cat', ohe, cat_cols))
    if num_cols:
        transformers.append(('num', 'passthrough', num_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.0
    )
    
    # OHE 안정화를 위해 범주형 -> 문자열 캐스팅
    for c in cat_cols:
        df_train[c] = df_train[c].astype(str).str.strip()

    ct.fit(df_train[feature_cols])
    return ct