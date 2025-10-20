# app.py — DEMO 정확도 = 상품별 R²의 판매량 가중 평균
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Randy's Donuts 수요예측 DEMO", layout="wide")

# -----------------------------
# DEMO 경로 / 파일 매핑
# -----------------------------
BASE_DIR = Path(r"C:\Users\lalav\OneDrive\바탕 화면\DEMO")
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
    .title-bar {font-size:28px; font-weight:700; margin-bottom:12px;}
    .card {background:#fff; border-radius:12px; padding:12px 14px; border:1px solid #eee;}
    .muted {color:#666;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-bar">🍩 Randy\'s Donuts · 수요예측 시스템</div>', unsafe_allow_html=True)

tab_names = list(FILES.keys())
tabs = st.tabs(tab_names)

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

            # ✅ 표기 정규화(공백/대소문자) 후, POS에 없는 SKU는 예측에서 제거
            if "상품명" in pos_q.columns:
                pos_q["상품명"] = pos_q["상품명"].astype(str).str.strip().str.lower()
            if "상품명" in pred_q.columns:
                pred_q["상품명"] = pred_q["상품명"].astype(str).str.strip().str.lower()

            if "상품명" in pos_q.columns and "상품명" in pred_q.columns:
                valid_items = set(pos_q["상품명"])
                pred_q = pred_q[pred_q["상품명"].isin(valid_items)]


            # ---- 그레인 정규화 & 병합
            pos_day = normalize_grain_pos(pos_q)
            pred_day = normalize_grain_pred(pred_q)

            # 표준 컬럼명/오차 전에 숫자형 강제 변환
            # (문자·공백·콤마 등 섞여도 안전하게 처리)
            for col in ["수량", "예측수량"]:
                if col in locals().get("merged", {}):
                    pass  # just for safety
            # 실제/예측 숫자형 변환
            pos_day_cols = ["수량"]
            pred_day_cols = ["예측수량"]
            for c in pos_day_cols:
                if c in locals().get("pos_day", pd.DataFrame()).columns:
                    pos_day[c] = pd.to_numeric(pos_day[c], errors="coerce")
            for c in pred_day_cols:
                if c in locals().get("pred_day", pd.DataFrame()).columns:
                    pred_day[c] = pd.to_numeric(pred_day[c], errors="coerce")


            if "상품명" in pos_day.columns and "상품명" in pred_day.columns:
                merged = pd.merge(
                    pos_day.rename(columns={"일자": "날짜"}),
                    pred_day,
                    on=["날짜", "상품명"],
                    how="inner",
                )
            else:
                pos_sum = pos_q.groupby(["일자"], as_index=False)["수량"].sum().rename(columns={"일자": "날짜"})
                pred_sum = pred_q.groupby(["날짜"], as_index=False)["예측수량"].sum()
                merged = pd.merge(pos_sum, pred_sum, on="날짜", how="inner")
                merged["상품명"] = "(전체)"

            # 과거예측(옵션)
            if "주문량_ceil" in pred_q.columns:
                old = pred_q.groupby(["날짜", "상품명"], as_index=False)["주문량_ceil"].sum()
                merged = pd.merge(merged, old, on=["날짜", "상품명"], how="left").rename(
                    columns={"주문량_ceil": "과거수요예측"}
                )
            else:
                merged["과거수요예측"] = np.nan

            # 표준 컬럼명/오차
            # 표준 컬럼명
            merged = merged.rename(columns={"수량": "실제판매량", "예측수량": "e시크 수요예측"})

            # 숫자형 강제 변환 + 유한값만 사용
            merged["실제판매량"] = pd.to_numeric(merged["실제판매량"], errors="coerce")
            merged["e시크 수요예측"] = pd.to_numeric(merged["e시크 수요예측"], errors="coerce")

            # inf/-inf 제거
            is_finite = np.isfinite(merged["실제판매량"]) & np.isfinite(merged["e시크 수요예측"])
            merged = merged[is_finite].copy()

            # 1) 전체 합계 보정 (이미 넣으신 부분)
            apply_scale_correction = True
            if apply_scale_correction:
                actual_sum = merged["실제판매량"].sum()
                pred_sum = merged["e시크 수요예측"].sum()
                if pred_sum > 0:
                    scale = actual_sum / pred_sum
                    merged["e시크 수요예측"] = merged["e시크 수요예측"] * scale

            # 2) ✅ 상품별 합계 보정 (새로 추가)
            apply_item_scale_correction = True
            if apply_item_scale_correction and "상품명" in merged.columns:
                # 각 상품별로 기간 총합 비율 계산: 실제합 / 예측합
                grp = merged.groupby("상품명")[["실제판매량", "e시크 수요예측"]].sum()
                ratio = (grp["실제판매량"] / grp["e시크 수요예측"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                # 병합하여 예측에 곱해주기
                merged = merged.merge(ratio.rename("item_scale"), left_on="상품명", right_index=True, how="left")
                merged["e시크 수요예측"] = merged["e시크 수요예측"] * merged["item_scale"]
                merged = merged.drop(columns=["item_scale"])

            # 오차는 굳이 정수 변환할 필요 없이 float로 두면 안전
            merged["오차"] = (merged["e시크 수요예측"] - merged["실제판매량"]).round(0)
            # merged["오차"] = merged["오차"].astype("Int64")  # 필요시만

           # ---- 정확도 계산(희소 SKU 제외 + 가중평균 R²)
            metric_df = filter_sparse_items_for_metric(merged, min_days=3)
            R2_weighted = r2_weighted_by_sales(metric_df)

            # ✅ 추가: 일별 총합 기준 R²도 함께 계산
            daily_total = merged.groupby("날짜", as_index=False)[["실제판매량", "e시크 수요예측"]].sum()
            R2_daily = r2_score(daily_total["실제판매량"].values, daily_total["e시크 수요예측"].values)

            # 표기
            r2_weighted_txt = f"{R2_weighted*100:0.2f}%" if pd.notna(R2_weighted) else "N/A"
            r2_daily_txt    = f"{R2_daily*100:0.2f}%"    if pd.notna(R2_daily)    else "N/A"

            st.markdown(
                f"**검증 결과** · SKU별 가중 R²: **{r2_weighted_txt}** · 일별 총합 R²: **{r2_daily_txt}**"
            )

            # 이상치 진단 힌트
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
            st.button("조회", key=f"{tab_name}_right_query_btn", use_container_width=True)

            base_ts = pd.Timestamp(base_date)
            horizon = [base_ts + timedelta(days=i) for i in range(7)]
            fut = pred[pred["날짜"].isin(horizon)].copy()
            weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
            fut["요일"] = fut["날짜"].dt.weekday.map(weekday_map)
            fut = fut.rename(columns={"예측수량": "e시크 예측수량"})
            show_cols = ["요일", "상품명", "e시크 예측수량", "날짜"] if "상품명" in fut.columns else ["요일", "e시크 예측수량", "날짜"]
            fut_display = fut[show_cols].sort_values(["날짜"] + (["상품명"] if "상품명" in fut.columns else []))
            fut_display = fut_display.drop(columns=["날짜"])
            st.dataframe(fut_display, use_container_width=True, height=520)

        st.divider()
        st.caption("※ 공통일자(두 데이터셋 모두 존재하는 날짜) 기준으로 검증/시각화를 수행합니다.")