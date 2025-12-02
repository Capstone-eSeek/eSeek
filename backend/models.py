from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Dict, Optional, Any

# ====================================================================
# 1. Store/Health Check (API 정보 및 지점 목록)
# ====================================================================

class HealthCheckResponse(BaseModel):
    """GET /health 응답 모델"""
    status: str
    timestamp: datetime
    available_stores: List[str]

class StoreInfo(BaseModel):
    """단일 지점 정보"""
    name: str = Field(..., description="지점 이름 (예: 제주애월점)")
    model: str = Field(..., description="사용 중인 머신러닝 모델 타입 (예: CatBoost)")

class StoreListResponse(BaseModel):
    """GET /stores 응답 모델"""
    stores: List[StoreInfo]

# ====================================================================
# 2. Validation (검증) 관련 모델
# ====================================================================

class ValidationQueryRequest(BaseModel):
    store_name: str = Field(..., description="검증할 지점 이름")
    start_date: date = Field(..., description="조회 시작일 (YYYY-MM-DD 형식)")
    end_date: date = Field(..., description="조회 종료일 (YYYY-MM-DD 형식)")
    product_name: str = Field(..., description="조회할 상품명, (전체)인 경우 모든 상품")
    

class MetricsData(BaseModel):
    """R² 등 검증 지표 데이터"""
    r2_weighted: Optional[float] = Field(None, description="가중 R-squared 값 (0.0~1.0)")
    r2_daily: Optional[float] = Field(None, description="일별 R-squared 평균 값 (0.0~1.0)")
    r2_weighted_display: str = Field(..., description="화면 표시용 가중 R² (예: 95.00%)")
    r2_daily_display: str = Field(..., description="화면 표시용 일별 R² (예: 92.50%)")

class DailyAggregate(BaseModel):
    """일별 집계 차트/테이블 데이터 레코드"""
    date: str = Field(..., description="날짜 (YYYY-MM-DD)")
    actual_sales: int = Field(..., description="실제 판매량 합계")
    model_prediction: float = Field(..., description="e시크 모델 예측량 합계")
    avg_prediction: Optional[float] = Field(None, description="기존(Legacy) 모델 예측량 합계 (사용하지 않으면 None)")
    error: float = Field(..., description="오차 (예측량 - 실제 판매량)")

class QueryPeriod(BaseModel):
    """조회 기간 정보"""
    start: str
    end: str

class ValidationResponse(BaseModel):
    """POST /validate 응답 모델"""
    store_name: str
    model_type: str
    query_period: QueryPeriod
    metrics: MetricsData
    daily_chart_data: List[DailyAggregate] = Field(..., description="차트용 일별 데이터")
    daily_table_data: List[DailyAggregate] = Field(..., description="테이블용 일별 데이터")
    total_records: int = Field(..., description="검증에 사용된 총 레코드 수 (POS/예측 병합 기준)")


# ====================================================================
# 3. Forecast (예측) 관련 모델 
# ====================================================================

class ForecastRequest(BaseModel):
    """POST /forecast 요청 모델"""
    store_name: str
    base_date: date
    horizon: int = Field(7, description="예측 기간 (일수)")

class PredictionRecord(BaseModel):
    """단일 상품에 대한 예측 결과 레코드"""
    date: str
    product_name: str
    predicted_qty: float
    order_qty_ceil: int = Field(..., description="올림 처리된 최종 주문 추천 수량")

class ForecastResponse(BaseModel):
    """POST /forecast 응답 모델"""
    store_name: str
    base_date: str
    horizon: int
    predictions: List[PredictionRecord]
    total_predictions: int
    csv_filename: str
    status: str