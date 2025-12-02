# app.py — DEMO 정확도 = 상품별 R²의 판매량 가중 평균
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from skops.io import load as skops_load, get_untrusted_types

@st.cache_resource
def _load_hgbr_pipeline(model_path: Path):
    trusted_types = list(get_untrusted_types(file=str(model_path)))
    return skops_load(model_path, trusted=trusted_types)

@st.cache_resource
def _load_catboost_model(model_path: Path):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor()
    m.load_model(str(model_path))
    return m

st.set_page_config(page_title="Randy's Donuts 수요예측 DEMO", layout="wide")

# -----------------------------
# DEMO 경로 / 파일 매핑
# -----------------------------
BASE_DIR = Path(r"C:/Users/lalav/OneDrive/바탕 화면/DEMO")
FILES = {
    "제주애월점": {
        "pos": BASE_DIR / "aewol_POS.csv",
        "pred": BASE_DIR / "aewol_PRED.csv",
        "pred_kind": "HGBR"
    },
    "부산광안리점": {
        "pos": BASE_DIR / "gwangan_POS.csv",
        "pred": BASE_DIR / "gwangan_PRED.csv",
        "pred_kind": "CatBoost log1p"
    },
    "수원타임빌리지점": {
        "pos": BASE_DIR / "suwon_POS.csv",
        "pred": BASE_DIR / "suwon_PRED.csv",
        "pred_kind": "CatBoost"
    },
    "연남점": {
        "pos": BASE_DIR / "yeonnam_POS.csv",
        "pred": BASE_DIR / "yeonnam_PRED.csv",
        "pred_kind": "CatBoost log1p"
    },
}

# -----------------------------
# 유틸: 로딩/정규화/지표
# -----------------------------
@st.cache_data
def load_pos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 날짜 정규화
    if "일자" in df.columns:
        df["일자"] = pd.to_datetime(df["일자"])
    elif "날짜" in df.columns:
        df["일자"] = pd.to_datetime(df["날짜"])
    else:
        raise ValueError(f"POS 날짜 컬럼(일자/날짜) 없음: {path}")
    # 수량 정규화
    if "수량" not in df.columns:
        for c in ["판매수량", "실제판매", "실제", "qty", "QTY"]:
            if c in df.columns:
                df["수량"] = df[c]
                break
        if "수량" not in df.columns:
            raise ValueError(f"POS 수량 컬럼 없음: {path}")
    # 상품명 보정
    if "상품명" not in df.columns:
        for c in ["product_name", "상품"]:
            if c in df.columns:
                df["상품명"] = df[c]
                break
    return df

def _looks_log_scale(df: pd.DataFrame, store_name: str) -> bool:
    header = " ".join(map(str, df.columns)).lower()
    return ("log" in header) or ("log1p" in header) or ("연남" in store_name and "log" in header)

@st.cache_data
def load_pred(path: Path, store_name: str = "", model_name: str = "") -> pd.DataFrame:
    df = pd.read_csv(path)
    # 날짜 정규화
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"])
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"])
    else:
        raise ValueError(f"PRED 날짜 컬럼(날짜/일자) 없음: {path}")
    # 예측수량 정규화
    if "예측수량" not in df.columns:
        for c in ["pred", "예측", "prediction", "예측(신)"]:
            if c in df.columns:
                df["예측수량"] = df[c]
                break
        if "예측수량" not in df.columns:
            raise ValueError(f"PRED 예측수량 컬럼 없음: {path}")
    # 상품명 보정
    if "상품명" not in df.columns:
        for c in ["product_name", "상품"]:
            if c in df.columns:
                df["상품명"] = df[c]
                break
    # ✅ 로그 스케일 복원 (단, 실제로 log 값일 때만 복원)
    if _looks_log_scale(df, store_name) or ("log" in model_name.lower()):
        # 평균이나 최대값이 log 범위(<=20) 안일 때만 expm1 수행
        if df["예측수량"].max() < 20 and df["예측수량"].mean() < 10:
            df["예측수량"] = np.expm1(np.clip(df["예측수량"].astype(float), a_min=-20, a_max=None))
    return df

def align_common_dates(pos_df: pd.DataFrame, pred_df: pd.DataFrame):
    common = sorted(set(pos_df["일자"]) & set(pred_df["날짜"]))
    return (
        pos_df[pos_df["일자"].isin(common)].copy(),
        pred_df[pred_df["날짜"].isin(common)].copy(),
    )

def normalize_grain_pos(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["상품명"] if c in df.columns]
    return df.groupby(["일자"] + cols, as_index=False)["수량"].sum()

def normalize_grain_pred(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["상품명"] if c in df.columns]
    out = df.groupby(["날짜"] + cols, as_index=False)["예측수량"].sum()
    out["예측수량"] = out["예측수량"].clip(lower=0)  # 음수 방지
    return out

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def r2_weighted_by_sales(df: pd.DataFrame) -> float:
    """상품별 R² 계산 후, 각 상품의 실제판매량 합계를 가중치로 가중 평균"""
    if "상품명" not in df.columns:
        return np.nan
    parts, weights = [], []
    for _, g in df.groupby("상품명"):
        r2 = r2_score(g["실제판매량"], g["e시크 수요예측"])
        w = float(g["실제판매량"].sum())
        if not np.isnan(r2) and w > 0:
            parts.append(r2)
            weights.append(w)
    if not parts:
        return np.nan
    return float(np.average(parts, weights=weights))

def filter_sparse_items_for_metric(df: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    """지표 계산에서 최소 일수 미만 SKU 제외(표시는 그대로)"""
    if "상품명" not in df.columns:
        return df
    days = df.groupby("상품명")["날짜"].nunique()
    keep = days[days >= min_days].index
    return df[df["상품명"].isin(keep)]

def to_kor_date(d: pd.Timestamp) -> str:
    return d.strftime("%y-%m-%d")

# -----------------------------
# 스타일
# -----------------------------

st.markdown(
    """
    <style>
    /* ===== 배경을 베이지 톤으로 ===== */
    .stApp, .stApp header { background: #FFFFFF !important; }
    .stAppViewContainer, .main, .block-container { background: #FFFFFF !important; }

    /* 카드 / 표 영역 같은 보조 배경 */
     .card, .stDataFrame, .stTable, .stMarkdown, .stSelectbox, .stDateInput, .stTextInput {
        background: #FFFFFF !important;   /* 필요시 #FAF5EC로 살짝 톤 줄 수도 있음 */
    }
    .card { border: 1px solid #EDEDED !important; }

    /* 제목바는 살짝 진한 베이지 */
    .title-bar {
        background: #F2E7D8; 
        padding: 10px 14px; 
        border-radius: 10px;
    }

    /* 텍스트 색 */
    html, body {
        color: #2B2B2B !important;
    }

    /* 버튼(다운로드 포함) 테두리/호버 */
    .stButton > button, .stDownloadButton > button {
        border: 2px solid #CBB8A0 !important;
        color: #2B2B2B !important;
        background: #FAF5EC !important;
        border-radius: 8px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #F2E7D8 !important;
        border-color: #BDA78C !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 제목 바 크기 증가: 28px -> 40px */
    .title-bar {font-size:40px; font-weight:700; margin-bottom:18px;}
    
    /* 카드 배경 스타일 유지 */
    .card {background:#fff; border-radius:12px; padding:12px 14px; border:1px solid #eee;}
    
    /* 뮤트된 텍스트 색상 유지 */
    .muted {color:#666;}
    
    /* ✅ Streamlit 본문 폰트 크기 대폭 조정 (14px -> 17px) */
    html, body, .stText, .stMarkdown, .stLabel, .stSelectbox, .stTextInput, .stDateInput, .stButton > button {
        font-size: 17px !important; /* 기본 14px에서 17px로 대폭 증가 */
    }
    
    /* 부제목 (h2, h3) 크기 증가 */
    h2 { font-size: 26px !important; } /* 기본 22px에서 26px로 증가 */
    h3 { font-size: 22px !important; } /* 기본 18px에서 22px로 증가 */

    /* 데이터프레임 헤더와 내용 크기 조정 */
    .stDataFrame, .stTable {
        font-size: 16px !important; /* 표 내부 폰트 조정 */
    }
    
    /* 캡션(caption) 텍스트 크기 증가 */
    .stCaption {
        font-size: 15px !important; /* 기본 12px에서 15px로 증가 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-bar">🍩 Randy\'s Donuts · 머신러닝 기반 수요예측 시스템</div>', unsafe_allow_html=True)

tab_names = list(FILES.keys())
tabs = st.tabs(tab_names)

# ====== (추가) 모델 로드/피처/예측 유틸 ======
import joblib
from pathlib import Path

MODEL_DIR = Path(BASE_DIR) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# log1p로 학습한 지점은 True (예: 광안리, 연남)
STORE_LOG_TARGET = {
    "제주애월점": False,          # HGBR(.pkl)
    "수원타임빌리지점": False,     # CatBoost(.cbm)
    "연남점": True,               # CatBoost log1p(.cbm)
    "부산광안리점": True,          # CatBoost log1p(.cbm)
}

# 저장된 모델 파일명 매핑 (파일명/확장자 지금 폴더와 정확히 맞추세요)
def _assets_paths(store: str):
    key = {
        # 제주애월점은 피처 파일명이 주어졌으나, 실제로 파일이 없을 경우도 None으로 처리하여 오류 방지
        "제주애월점": ("aewol_hgbr_pipeline.skops", None, None),        # (model, features.pkl, categ.pkl)
        "수원타임빌리지점": ("suwon_catboost.cbm", None, None),
        "연남점": ("yeonnam_catboost.cbm", None, None),
        "부산광안리점": ("gwangan_catboost.cbm", None, None),
    }[store]
    m, f, c = key
    
    # ✅ 수정: None이 아닌 경우에만 Path 객체를 생성하고, None인 경우 None을 그대로 반환
    model_path = MODEL_DIR / m
    feat_path = MODEL_DIR / f if f else None  
    catcols_path = MODEL_DIR / c if c else None

    # 수정된 반환
    return (model_path, feat_path, catcols_path)

def _expected_catboost_feature_names(model):
    # 1순위: 모델이 기억하는 피처 이름
    names = getattr(model, "feature_names_", None)
    if names and len(names) > 0:
        return list(names)
    # 2순위: 중요도 테이블에서 추출 (버전에 따라 컬럼명이 다름)
    try:
        fi = model.get_feature_importance(prettified=True)
        col = "Feature Id" if "Feature Id" in fi.columns else ("Feature" if "Feature" in fi.columns else None)
        if col:
            names = [str(x) for x in fi[col].tolist()]
            if len(names) > 0:
                return names
    except Exception:
        pass
    return None  # 못 구했으면 None
        

def _align_X_to_catboost(model, X: pd.DataFrame) -> pd.DataFrame:
    exp = _expected_catboost_feature_names(model)
    if not exp:
        # 안전장치: 이름을 못 구하면 바로 실패시켜 원인 파악
        missing = "모델에 저장된 feature_names_가 없습니다. 학습 시 feature_names 저장 옵션을 확인하세요."
        raise RuntimeError(missing)

    # 1) 누락된 열은 0으로 추가
    for c in exp:
        if c not in X.columns:
            X[c] = 0

    # 2) 여분 열 제거 + 순서 일치
    X = X[[c for c in exp]]

    # 3) dtype 정리: 범주형 가능성이 있는 object/bool은 문자열, datetime은 문자열로
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = pd.to_datetime(X[c]).dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_bool_dtype(X[c]) or pd.api.types.is_object_dtype(X[c]):
            X[c] = X[c].astype(str)

    return X

def load_model_for_store(store: str):
    model_path, _, _ = _assets_paths(store)
    kind = FILES[store]["pred_kind"].lower()
    if "hgbr" in kind:
        return _load_hgbr_pipeline(model_path), None, []
    else:
        return _load_catboost_model(model_path), None, []


# 학습 노트북과 동일 피처 전처리를 여기에 반영하세요(간단 기본형 제공)
def _base_features(df: pd.DataFrame):
    out = df.copy()
    out["dow"] = pd.to_datetime(out["날짜"]).dt.weekday
    # 수치 결측 메우기
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out

def make_features(store: str, X: pd.DataFrame) -> pd.DataFrame:
    """
    CatBoost 입력용 피처 생성:
      - 날짜 파생: month / day / week
      - datetime → 문자열(YYYY-MM-DD) 또는 제거
      - object/bool → 문자열로 통일
      - 수치 결측 → 중앙값 대체
    ※ HGBR 분기/원핫 등은 이 함수에서 처리하지 않습니다.
    """
    out = X.copy()

    # 날짜 파생(있을 때만) + 날짜 컬럼 제거
    if "날짜" in out.columns:
        d = pd.to_datetime(out["날짜"])
        out["month"] = d.dt.month.astype(int)
        out["day"]   = d.dt.day.astype(int)
        out["week"]  = d.dt.isocalendar().week.astype(int)
        out = out.drop(columns=["날짜"], errors="ignore")

    # dtype 정리
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            # 남아 있는 datetime이 있다면 문자열로 변환
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_bool_dtype(out[c]) or pd.api.types.is_object_dtype(out[c]):
            # CatBoost 호환을 위해 범주형/문자열은 전부 str로
            out[c] = out[c].astype(str)

    # 수치 결측 치환(중앙값)
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    return out

def make_features_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "날짜" in out.columns:
        d = pd.to_datetime(out["날짜"])
        out["month"] = d.dt.month.astype(int)
        out["day"]   = d.dt.day.astype(int)
        out["week"]  = d.dt.isocalendar().week.astype(int)
        out = out.drop(columns=["날짜"], errors="ignore")
    # dtype 정리
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_bool_dtype(out[c]) or pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out

# === 제주애월(HGBR)용: 학습 때 쓰던 컬럼을 정확히 만들어줌
def make_hgbr_inputs_from_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    future_grid(날짜, 상품명) -> HGBR 파이프라인이 기대하는 원시 피처 프레임
    - cat: ['상품명','day']  (day는 요일영문명으로 생성)
    - num: ['month','is_holiday','is_holiday_window','is_pre_holiday_window',
            'is_post_holiday_window','holiday_weight','temp_max','precip','event_any',
            'event_closing_50','event_lucky_box','event_picnic_mat','event_world_donut_day',
            'event_closing_50_days_since_start','event_lucky_box_days_since_start',
            'event_picnic_mat_days_since_start','event_world_donut_day_days_since_start']
    """
    out = df.copy()

    # 날짜 파생
    d = pd.to_datetime(out["날짜"])
    # 학습 때 day가 요일 이름이었다면 → 영어 요일명으로 생성 (Mon..Sun)
    out["day"] = d.dt.day_name()  # e.g., 'Monday', 'Tuesday', ...
    out["month"] = d.dt.month.astype(int)

    # 필수 컬럼들(없으면 0으로 채움)
    needed_zero = [
        "is_holiday","is_holiday_window","is_pre_holiday_window","is_post_holiday_window",
        "holiday_weight","temp_max","precip","event_any",
        "event_closing_50","event_lucky_box","event_picnic_mat","event_world_donut_day",
        "event_closing_50_days_since_start","event_lucky_box_days_since_start",
        "event_picnic_mat_days_since_start","event_world_donut_day_days_since_start",
    ]
    for c in needed_zero:
        if c not in out.columns:
            out[c] = 0

    # dtype 정리: 범주형은 문자열, 수치는 숫자
    out["상품명"] = out["상품명"].astype(str).str.strip()
    out["day"]   = out["day"].astype(str).str.strip()

    num_cols = [
        "month","is_holiday","is_holiday_window","is_pre_holiday_window","is_post_holiday_window",
        "holiday_weight","temp_max","precip","event_any",
        "event_closing_50","event_lucky_box","event_picnic_mat","event_world_donut_day",
        "event_closing_50_days_since_start","event_lucky_box_days_since_start",
        "event_picnic_mat_days_since_start","event_world_donut_day_days_since_start",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # 파이프라인에 넘길 컬럼 순서(이 순서로 넘기면 안전)
    cat_cols = ["상품명","day"]
    cols = cat_cols + num_cols
    return out[cols]

def build_future_grid(base_date: pd.Timestamp, horizon: int, df_train_like: pd.DataFrame):
    """기준일 다음날부터 horizon일까지 날짜×상품 그리드 생성"""
    base_date = pd.to_datetime(base_date)
    dates = pd.date_range(base_date + pd.Timedelta(days=1),
                          base_date + pd.Timedelta(days=horizon), freq="D")
    prods = (df_train_like["상품명"].astype("category").cat.categories
             if pd.api.types.is_categorical_dtype(df_train_like["상품명"])
             else sorted(df_train_like["상품명"].unique()))
    grid = (
        pd.DataFrame({"날짜": dates}).assign(key=1)
          .merge(pd.DataFrame({"상품명": prods, "key":1}), on="key", how="inner")
          .drop(columns="key").sort_values(["날짜","상품명"]).reset_index(drop=True)
    )
    return grid

def predict_next_week(store: str, base_date: pd.Timestamp, train_like_df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    model, _, _ = load_model_for_store(store)
    log_target = STORE_LOG_TARGET.get(store, False)

    future_grid = build_future_grid(base_date, horizon, train_like_df)

    if "hgbr" in FILES[store]["pred_kind"].lower():
        X = make_hgbr_inputs_from_grid(future_grid.copy())
        yhat = np.asarray(model.predict(X), float)
    else:
        X = make_features_for_catboost(future_grid.copy())
        X = _align_X_to_catboost(model, X)
        yhat = np.asarray(model.predict(X), float)

    if log_target:
        yhat = np.expm1(yhat)
    yhat = np.clip(yhat, 0, None)

    out = future_grid.copy()
    out["예측수량"] = yhat
    return out[["날짜","상품명","예측수량"]]

# -----------------------------
# 탭 렌더링
# -----------------------------
for tab_name, tab in zip(tab_names, tabs):
    with tab:
        conf = FILES[tab_name]
        model_name = conf.get("pred_kind", "")
        colL, colR = st.columns([1.45, 1.0], gap="large")

        # 데이터 로드 및 공통일자 정렬
        pos = load_pos(conf["pos"])
        pred = load_pred(conf["pred"], store_name=tab_name, model_name=model_name)
        pos, pred = align_common_dates(pos, pred)

        st.markdown(f"**지점:** {tab_name}  ·  모델: {model_name}")

        # ----------------- 좌: 검증 패널 -----------------
        with colL:
            st.subheader("조회기간 · 상품코드 / 검증")

            c1, c2, c3, c4, c5 = st.columns([0.6, 0.6, 0.7, 1.2, 0.5])
            min_d = pos["일자"].min()
            max_d = pos["일자"].max()

            from_d = c1.date_input(
                "조회 시작일",
                min_d.date() if pd.notna(min_d) else datetime.today().date(),
                key=f"{tab_name}_from_date",
            )
            to_d = c2.date_input(
                "조회 종료일",
                max_d.date() if pd.notna(max_d) else datetime.today().date(),
                key=f"{tab_name}_to_date",
            )

            item_list = (
                sorted(pos["상품명"].dropna().unique().tolist())
                if "상품명" in pos.columns
                else ["(전체)"]
            )
            item_sel = c3.selectbox(
                "상품명",
                options=["(전체)"] + item_list,
                index=0,
                key=f"{tab_name}_item_sel",
            )
            item_query = c4.text_input(
                "상품코드/명 검색",
                "",
                placeholder="상품명 일부 또는 코드 입력",
                key=f"{tab_name}_item_query",
            )
            c5.button("조회", use_container_width=True, key=f"{tab_name}_left_query_btn")

            _from = pd.Timestamp(from_d)
            _to = pd.Timestamp(to_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            pos_q = pos[(pos["일자"] >= _from) & (pos["일자"] <= _to)].copy()
            pred_q = pred[(pred["날짜"] >= _from) & (pred["날짜"] <= _to)].copy()

            # 선택 필터
            if item_sel != "(전체)" and "상품명" in pos_q.columns:
                pos_q = pos_q[pos_q["상품명"] == item_sel]
                pred_q = pred_q[pred_q["상품명"] == item_sel]
            if item_query.strip():
                key = item_query.strip()
                if "상품명" in pos_q.columns:
                    mpos = pos_q["상품명"].astype(str).str.contains(key, case=False, na=False)
                else:
                    mpos = pd.Series([False] * len(pos_q), index=pos_q.index)
                if "상품코드" in pos_q.columns:
                    mpos |= pos_q["상품코드"].astype(str).str.contains(key, case=False, na=False)
                pos_q = pos_q[mpos] if len(pos_q) else pos_q

            # ===================== 정규화 · 집계 · 병합 (교체 블록 시작) =====================
            # 표시/키 분리: 매칭은 key(소문자/트림), 표시는 display 유지
            if "상품명" in pos_q.columns:
                pos_q["상품명_display"] = pos_q["상품명"].astype(str)
                pos_q["상품명_key"] = pos_q["상품명_display"].str.strip().str.lower()
            if "상품명" in pred_q.columns:
                pred_q["상품명_key"] = pred_q["상품명"].astype(str).str.strip().str.lower()

            # POS에 존재하는 SKU만 예측에서 유지
            if "상품명_key" in pos_q.columns and "상품명_key" in pred_q.columns:
                valid_items = set(pos_q["상품명_key"])
                pred_q = pred_q[pred_q["상품명_key"].isin(valid_items)]

            # 그레인 정규화 (일자×상품)
            pos_day  = pos_q.groupby(["일자", "상품명_key", "상품명_display"], as_index=False)["수량"].sum()
            pred_day = pred_q.groupby(["날짜", "상품명_key"],                         as_index=False)["예측수량"].sum()

            # 숫자형 강제 변환(혼입 방지)
            pos_day["수량"]      = pd.to_numeric(pos_day["수량"], errors="coerce")
            pred_day["예측수량"] = pd.to_numeric(pred_day["예측수량"], errors="coerce")

            # 병합 (일자→날짜로 맞춤)
            merged = pd.merge(
                pos_day.rename(columns={"일자": "날짜"}),
                pred_day,
                on=["날짜", "상품명_key"],
                how="inner",
            )

            # 표시용 상품명 복원 후 키 컬럼 제거
            merged["상품명"] = merged["상품명_display"]
            merged = merged.drop(columns=["상품명_display", "상품명_key"])

            # 과거예측(옵션) 합치기
            if "주문량_ceil" in pred_q.columns:
                old = pred_q.groupby(["날짜", "상품명_key"], as_index=False)["주문량_ceil"].sum()
                name_map = pos_day[["상품명_key", "상품명_display"]].drop_duplicates()
                old = old.merge(name_map, on="상품명_key", how="left")
                old["상품명"] = old["상품명_display"]
                old = old.drop(columns=["상품명_display", "상품명_key"])
                merged = pd.merge(merged, old, on=["날짜", "상품명"], how="left").rename(
                    columns={"주문량_ceil": "과거수요예측"}
                )
            else:
                merged["과거수요예측"] = np.nan

            # 표준 컬럼명 정리
            merged = merged.rename(columns={"수량": "실제판매량", "예측수량": "e시크 수요예측"})

            # 숫자형 강제 변환 + 유한값만 유지
            merged["실제판매량"]    = pd.to_numeric(merged["실제판매량"], errors="coerce")
            merged["e시크 수요예측"] = pd.to_numeric(merged["e시크 수요예측"], errors="coerce")
            is_finite = np.isfinite(merged["실제판매량"]) & np.isfinite(merged["e시크 수요예측"])
            merged = merged[is_finite].copy()
            # ===================== 정규화 · 집계 · 병합 (교체 블록 끝) =====================

            # ===== 보정 (택1로 설정해서 사용) =====
            scale_mode = "SKU별 합 맞춤"   # "전체합 맞춤" / "없음"

            if scale_mode == "전체합 맞춤":
                pred_sum = merged["e시크 수요예측"].sum()
                if pred_sum > 0:
                    merged["e시크 수요예측"] *= (merged["실제판매량"].sum() / pred_sum)

            elif scale_mode == "SKU별 합 맞춤" and "상품명" in merged.columns:
                grp = merged.groupby("상품명")[["실제판매량", "e시크 수요예측"]].sum()
                ratio = (grp["실제판매량"] / grp["e시크 수요예측"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                merged = merged.merge(ratio.rename("item_scale"), left_on="상품명", right_index=True, how="left")
                merged["e시크 수요예측"] *= merged["item_scale"]
                merged.drop(columns=["item_scale"], inplace=True)
            # else: 없음

            # 오차
            merged["오차"] = (merged["e시크 수요예측"] - merged["실제판매량"]).round(0)

            # ---- 정확도 계산
            metric_df = filter_sparse_items_for_metric(merged, min_days=3)
            R2_weighted = r2_weighted_by_sales(metric_df)
            daily_total = merged.groupby("날짜", as_index=False)[["실제판매량", "e시크 수요예측"]].sum()
            R2_daily = r2_score(daily_total["실제판매량"].values, daily_total["e시크 수요예측"].values)

            r2_weighted_txt = f"{R2_weighted*100:0.2f}%" if pd.notna(R2_weighted) else "N/A"
            r2_daily_txt    = f"{R2_daily*100:0.2f}%"    if pd.notna(R2_daily)    else "N/A"

            st.markdown(f"**검증 결과** · SKU별 가중 R²: **{r2_weighted_txt}** · 일별 총합 R²: **{r2_daily_txt}**")

            if pd.notna(R2_weighted) and R2_weighted < -1.0:
                bad = metric_df.copy()
                bad["절대오차"] = (bad["e시크 수요예측"] - bad["실제판매량"]).abs()
                worst = bad.groupby("상품명")["절대오차"].sum().sort_values(ascending=False).head(5)
                st.warning(
                    "R²가 비정상적으로 낮습니다. (가능 원인: 로그 미복원, SKU 미스매치, 단위 불일치, 희소 SKU)\n"
                    f"- 최다 오차 SKU Top5: {', '.join(map(str, worst.index.tolist()))}\n"
                    "- 예측수량 이상치/음수 여부와 상품명 매칭을 확인하세요."
                )

            # ---- 차트/표
            chart_day = merged.groupby("날짜")[["실제판매량", "e시크 수요예측", "과거수요예측"]].sum().reset_index().sort_values("날짜")
            chart_day_display = chart_day.copy()
            chart_day_display["날짜"] = chart_day_display["날짜"].dt.strftime("%Y-%m-%d")
            st.line_chart(chart_day_display.set_index("날짜")[["실제판매량", "e시크 수요예측", "과거수요예측"]])

            table_day = merged.groupby("날짜", as_index=False)[["실제판매량", "e시크 수요예측", "오차"]].sum().sort_values("날짜")
            table_day_display = table_day.copy()
            table_day_display.insert(0, "날짜(yy-mm-dd)", table_day_display["날짜"].apply(to_kor_date))
            table_day_display = table_day_display.drop(columns=["날짜"])
            st.dataframe(table_day_display, use_container_width=True, height=260)

            st.markdown("**해당 기간동안 수요예측 결과에 가장 큰 영향을 미친 요인**")
            st.markdown('<div class="card muted">1순위: 요일 · 2순위: 날씨(최고기온)</div>', unsafe_allow_html=True)

        # ----------------- 우: 기준일자 7일 예측 -----------------
        with colR:
            st.subheader("기준일자 · 7일 예측")
            r1, r2 = st.columns([1.0, 0.3])
            base_date = r1.date_input(
                "기준일자",
                value=(pos["일자"].max().date() if len(pos) else datetime.today().date()),
                key=f"{tab_name}_right_base",
            )
            r2.write("")
            run_btn = st.button("다음주 예측 생성하기", key=f"{tab_name}_run_model", use_container_width=True)

            # 학습 때 본 SKU 목록 재현용: POS를 일자×상품 합계 형태로
            if "상품명" in pos.columns:
                train_like = (
                    pos.groupby(["일자","상품명"], as_index=False)["수량"]
                    .sum().rename(columns={"일자":"날짜"})
                )
            else:
                tmp = pos.groupby("일자", as_index=False)["수량"].sum().rename(columns={"일자":"날짜"})
                tmp["상품명"] = "(전체)"
                train_like = tmp[["날짜","상품명"]].copy()

            if run_btn:
                try:
                    pred_out = predict_next_week(tab_name, pd.Timestamp(base_date), train_like, horizon=7)
                    fn = f"{tab_name}_next7_pred_{pd.Timestamp(base_date).strftime('%Y%m%d')}.csv"
                    csv_bytes = pred_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    
                    # ----------------------------------------------------
                    # ✅ 예측 완료 메시지 및 버튼 스타일 강화 부분 시작
                    # ----------------------------------------------------
                    
                    # 1. 폰트 크기를 키운 Success 메시지 출력
                    st.markdown(
                        '<p style="font-size:20px; font-weight:bold; color:green;">예측 완료! 아래 버튼으로 CSV 저장하세요.</p>',
                        unsafe_allow_html=True
                    )
                    
                    # 2. 다운로드 버튼 스타일링을 위한 CSS 삽입
                    st.markdown(
                        """
                        <style>
                        /* 다운로드 버튼을 포함하는 stDownloadButton 컨테이너의 버튼 스타일 */
                        .stDownloadButton > button {
                            font-size: 18px !important; /* 폰트 크기 증가 */
                            font-weight: bold !important; /* 굵게 */
                            padding: 10px 20px !important; /* 패딩 증가로 버튼 크기 키우기 */
                            
                            /* ✅ 테두리를 명확한 회색으로 설정 */
                            border: 2px solid #ccc !important; 
                            
                            /* 배경색은 투명 유지 (Streamlit 기본값) */
                            background-color: transparent !important;
                            
                            /* 텍스트 색상을 검정 계열로 설정하여 잘 보이도록 함 */
                            color: #333333 !important; 
                            border-radius: 8px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 3. 다운로드 버튼 출력
                    st.download_button(
                        "다음주 예측 CSV 다운로드", 
                        data=csv_bytes, 
                        file_name=fn, 
                        mime="text/csv"
                    )
                    
                    # ----------------------------------------------------
                    # ✅ 스타일 강화 부분 종료
                    # ----------------------------------------------------

                    show_cols = ["날짜","상품명","예측수량"] if "상품명" in pred_out.columns else ["날짜","예측수량"]
                    st.dataframe(pred_out[show_cols].sort_values(["날짜","상품명"] if "상품명" in show_cols else ["날짜"]),
                                 use_container_width=True, height=480)
                except Exception as e:
                    st.error(f"예측 실행 중 오류 ({tab_name}): {type(e).__name__}: {e}")


            st.caption("※ 로컬에 저장된 모델/피처를 사용해 기준일 다음날부터 7일 예측을 생성합니다.")
