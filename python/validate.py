import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import date
from fastapi import HTTPException
from pathlib import Path
from sklearn.metrics import r2_score

# models.py에서 필요한 모델들을 import 해야 함
# 이미 backend/main.py에서 api 엔드포인트 상단에서 경로에 추가했기 때문에 바로 from models해도 됨
from backend.models import (
    ValidationQueryRequest,
    ValidationResponse,
    MetricsData,
    DailyAggregate,
    QueryPeriod
) # [DEBUG] 2. backend.models 임포트 성공 확인
print("--- [VALIDATE DEBUG] 2. backend.models 임포트 성공. ---")
# 경로 설정(데이터 파일)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================
# 1. 유틸리티 함수 (데이터 로드 및 처리)
# ============================================
def load_data(filename: str, date_column: str) -> pd.DataFrame:
    """CSV 파일을 로드하고 지정된 컬럼을 datetime 객체로 변환합니다."""
    file_path = DATA_DIR / filename
    
    if not file_path.exists():
        # 파일이 없으면 500 오류를 발생시켜 API로 전달합니다.
        raise FileNotFoundError(f"데이터 파일이 없습니다: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
         # encoding 문제 발생 시 다른 인코딩 시도
        df = pd.read_csv(file_path, encoding='euc-kr')

    # 날짜 컬럼 처리 (데이터 처리 전 필수)
    if date_column in df.columns:
        # 기존 코드에서 POS는 '일자', PRED는 '날짜'를 사용하므로 '날짜'로 통일
        df.rename(columns={date_column: '날짜'}, inplace=True)
        df['날짜'] = pd.to_datetime(df['날짜'])
    else:
        raise KeyError(f"데이터 파일 '{filename}'에 필수 날짜 컬럼 '{date_column}'이 없습니다.")
        
    return df

def merge_pos_pred(pos_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """POS와 예측 데이터를 병합하고 오차를 계산합니다."""
    
    # 컬럼 이름 통일 및 준비
    pos_agg = pos_df.rename(columns={"수량": "실제판매량"})
    pred_agg = pred_df

    merged = pos_agg.merge(pred_agg, on=['날짜','상품명'], how='left')

    merged = merged.drop_duplicates(subset=['날짜', '상품명'])
    # => merged 컬럼: 상품명, 상품코드, 실제판매량, 날짜, 예측수량, 주문량_ceil

    # 오차 계산
    merged['error'] = merged['주문량_ceil'] - merged['실제판매량']
    merged['abs_error'] = merged['error'].abs()
    merged['squared_error'] = merged['error']**2
    merged['ape'] = np.where(merged['실제판매량'] != 0, merged['abs_error']/merged['실제판매량'], np.nan)
    merged['smape'] = 100 * 2 * merged['abs_error'] / (merged['주문량_ceil'].abs() + merged['실제판매량'].abs())

    merged = merged.sort_values(['날짜','상품명'])
    
    # 날짜가 가장 맨 왼쪽에 오도록 정렬
    final_columns = [
        '날짜',
        '상품명',
        '실제판매량',
        '주문량_ceil',
        '예측수량',
        'error',
        'abs_error',
        'squared_error',
        'ape',
        'smape'
    ]

    present_columns = [col for col in final_columns if col in merged.columns]
    
    return merged[present_columns]


def calculate_r2_weighted(merged_df: pd.DataFrame) -> float:
    """병합된 데이터 전체에 대해 가중 R-squared를 계산합니다."""
    y_true = merged_df['실제판매량'].values
    y_pred = merged_df['주문량_ceil'].values
    
    # R² 계산을 위한 최소 조건 확인 (분산이 0이면 계산 불가)
    if len(y_true) < 2 or np.sum((y_true - y_true.mean()) ** 2) == 0:
        return 0.0
        
    r2 = r2_score(y_true, y_pred)
    
    return max(0.0, r2) 

def calculate_r2_daily(merged_df: pd.DataFrame) -> float:
    """일별 R-squared를 계산하고 평균을 반환합니다."""
    daily_r2_list = []
    
    for _, group in merged_df.groupby('날짜'):
        y_true = group['실제판매량'].values
        y_pred = group['주문량_ceil'].values
        
        # 일별 R² 계산을 위한 최소 조건 (최소 2개 이상의 데이터 및 분산 > 0)
        if len(y_true) > 1 and np.sum((y_true - y_true.mean()) ** 2) > 0:
            r2 = r2_score(y_true, y_pred)
            daily_r2_list.append(max(0.0, r2))

    if not daily_r2_list:
        return 0.0

    return np.mean(daily_r2_list)

def calculate_avg_prediction(
    pos_df: pd.DataFrame, 
    query_start_date: date, 
    days_prior: int = 7
) -> pd.DataFrame:
    """
    조회 기간 직전 N일의 평균 판매량을 계산하여 예측값으로 사용합니다.

    :param pos_df: 전체 POS(실제 판매) 데이터프레임
    :param query_start_date: 조회 기간 시작일 (date 객체)
    :param days_prior: 평균 계산에 사용할 직전 일수
    :return: '날짜', '상품명', 'avg_prediction'을 포함하는 데이터프레임
    """
    
    end_date_for_avg = pd.Timestamp(query_start_date) - pd.Timedelta(days=1)
    start_date_for_avg = end_date_for_avg - pd.Timedelta(days=days_prior - 1)

    # 1. 평균 계산에 사용할 기간 필터링
    # '날짜'가 '일자' 컬럼으로 로드된 원본 pos_df를 사용해야 함
    avg_period_df = pos_df[
        (pos_df['날짜'] >= start_date_for_avg) & 
        (pos_df['날짜'] <= end_date_for_avg)
    ].copy()
    
    if avg_period_df.empty:
        print(f"--- [INFO] 평균 계산을 위한 이전 {days_prior}일 데이터가 부족합니다. 0으로 처리합니다. ---")
        # 데이터가 없으면 상품별 평균을 0으로 처리
        avg_sales_by_product = {}
    else:
        # 2. 상품별 일평균 실제판매량 계산
        # 일별 합산 후, 일수로 나누어 기간 평균 계산
        daily_sales = avg_period_df.groupby(['날짜', '상품명'])['수량'].sum().reset_index()
        
        # 상품별 평균 (총합 / 사용된 날짜 수)
        avg_sales_by_product = daily_sales.groupby('상품명')['수량'].mean().round(0).to_dict()

    # 3. 조회 기간 전체에 평균값을 적용할 데이터프레임 생성
    query_dates = pos_df['날짜'].unique()
    query_products = pos_df['상품명'].unique()
    
    # 조회 기간의 모든 날짜-상품 조합 생성
    from itertools import product
    all_combinations = list(product(query_dates, query_products))
    
    avg_pred_df = pd.DataFrame(all_combinations, columns=['날짜', '상품명'])
    
    # 상품별 평균값을 적용
    avg_pred_df['avg_prediction'] = avg_pred_df['상품명'].map(avg_sales_by_product).fillna(0).astype(float)
    
    return avg_pred_df

# ============================================
# 2. 메인 검증 로직 함수 (calculate_r2)
# ============================================
def calculate_r2(query: ValidationQueryRequest, config: Dict[str, Any]) -> ValidationResponse:
    """실제 데이터 로드, 병합, R2 계산 및 응답 형식을 처리합니다."""
    
    print(f"--- [DEBUG] 데이터 로드 및 검증 시작 (지점: {query.store_name}) ---")

    # 1. 데이터 로드
    # config에서 파일 이름 가져와서 load_data 함수에 전달
    pos_df = load_data(config["pos_file"], date_column="일자")
    pred_df = load_data(config["pred_file"], date_column="날짜")

    # 2. 조회 기간 및 상품 필터링
    start = pd.Timestamp(query.start_date)
    end = pd.Timestamp(query.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    pos_df = pos_df[(pos_df["날짜"] >= start) & (pos_df["날짜"] <= end)]
    pred_df = pred_df[(pred_df["날짜"] >= start) & (pred_df["날짜"] <= end)]

    if query.product_name != "(전체)" and "상품명" in pos_df.columns:
        pos_df = pos_df[pos_df["상품명"] == query.product_name]
        pred_df = pred_df[pred_df["상품명"] == query.product_name]

    # 3. 데이터 병합 및 오차 계산
    merged = merge_pos_pred(pos_df, pred_df)

    if len(merged) == 0:
        raise HTTPException(status_code=404, detail="조회 기간에 일치하는 데이터가 없습니다")
    
    # 4. 평균 기반 예측
    pos_raw = load_data(config["pos_file"], date_column="일자")
    avg_pred_df = calculate_avg_prediction(
        pos_df = pos_raw,
        query_start_date = query.start_date,
        days_prior=7 # 직전 7일 평균 사용 
    )
    print(avg_pred_df)

    # merged 데이터프레임에 평균 예측값 병합
    merged = merged.merge(avg_pred_df, on=['날짜', '상품명'], how='left')
    # avg_prediction 컬럼에 값이 없으면 0으로 채움 (데이터 부족 등으로)
    merged['avg_prediction'] = merged['avg_prediction'].fillna(0)

    # 5. 성능 지표 계산
    r2_weighted = calculate_r2_weighted(merged)
    r2_daily = calculate_r2_daily(merged)

    print("r2_weighted", r2_weighted)
    print("r2_daily", r2_daily)

    # 6. 응답 형식 가공(일별 집계)
    # 최종 예측값과 실제 판매량을 일별로 합산하여 데이터 만들기
    daily_data = merged.groupby("날짜").agg({
        "실제판매량": "sum",
        "주문량_ceil": "sum",
        "error": "sum",
        "avg_prediction": "sum"
    }).reset_index()
    
    chart_data = [
        DailyAggregate(
            date=row["날짜"].strftime("%Y-%m-%d"),
            actual_sales=int(row["실제판매량"]),
            model_prediction=float(row["주문량_ceil"]),
            avg_prediction=float(row["avg_prediction"]),
            error=float(row["error"])
        )
        for _, row in daily_data.iterrows()
    ]

    return ValidationResponse(
        store_name=query.store_name,
        model_type=config["model_type"],
        query_period=QueryPeriod(
            start=query.start_date.strftime("%Y-%m-%d"),
            end=query.end_date.strftime("%Y-%m-%d")
        ),
        metrics=MetricsData(
            r2_weighted=r2_weighted,
            r2_daily=r2_daily,
            r2_weighted_display=f"{r2_weighted*100:.2f}%" if r2_weighted else "N/A",
            r2_daily_display=f"{r2_daily*100:.2f}%" if r2_daily else "N/A"
        ),
        daily_chart_data=chart_data,
        daily_table_data=chart_data,
        total_records=len(merged)
    )


# ============================================
# 3. run_validation 함수 (API 엔드포인트에서 호출)
# ============================================
def run_validation(query: ValidationQueryRequest, store_config: Dict[str, Any]) -> ValidationResponse:
    
    print("=" * 40)
    print(f"API 요청 정보 확인:")
    print(f"   선택 지점: {query.store_name}")
    print(f"   시작일:    {query.start_date.strftime('%Y-%m-%d')}")
    print(f"   종료일:    {query.end_date.strftime('%Y-%m-%d')}")
    print(f"   선택 상품: {query.product_name}")
    print("=" * 40)
    
    # 지점 설정 가져오기 (테스트를 위해 필요)
    if query.store_name not in store_config:
        raise HTTPException(status_code=404, detail="지점을 찾을 수 없습니다")
    
    config = store_config[query.store_name]

    # calculate_r2 (핵심 로직) 호출
    try:
        return calculate_r2(query, config)
    except Exception as e:
        print(f"--- [FATAL ERROR] 검증 로직 처리 중 예외 발생: {type(e).__name__} - {e} ---")
        raise HTTPException(status_code=500, detail=f"데이터 처리 오류: {type(e).__name__} - {str(e)}")
