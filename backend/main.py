# api 엔드포인트 (최소 버전)
import sys
from pathlib import Path
from typing import List, Dict, Any

# python 폴더를 import 경로에 추가
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
PYTHON_DIR = PROJECT_ROOT/"python"
sys.path.insert(0, str(PYTHON_DIR))

# 경로 설정
EXPORT_DIR = PROJECT_ROOT / "data" / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_DIR = PROJECT_ROOT / "data" / "auto_forecast_results"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import importlib

# 파이썬 스케줄러
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
from datetime import date, timedelta
import pytz # 시간대 처리를 위해 추가

# 서울 시간대 설정
SEOUL_TZ = pytz.timezone('Asia/Seoul')

# Pydantic 모델 import
from models import (
    ValidationQueryRequest,
    ValidationResponse,
    StoreListResponse,
    ForecastResponse,
    ForecastRequest,
    PredictionRecord
)

# 외부 모듈: python/validate.py 에서 run_validation 함수 import
try:
    from validate import run_validation
except ImportError:
    # validate.py 파일이 없거나 함수가 정의되지 않은 경우를 대비한 임시 예외 처리
    def run_validation(query: ValidationQueryRequest, store_config: Dict[str, Any]):
        raise NotImplementedError("validate.py 모듈의 run_validation 함수가 정의되지 않았습니다.")
    
# 스케줄러
scheduler = AsyncIOScheduler(timezone=str(SEOUL_TZ))

# 지점별 설정 (필수)
STORE_CONFIG = {
    "제주애월점": {
        "pos_file": "aewol_POS.csv",
        "pred_file": "aewol_PRED.csv",
        "model_type": "HGBR",
        "predict_module": "aewol_predict"
    },
    "부산광안리점": {
        "pos_file": "gwangan_POS.csv",
        "pred_file": "gwangan_PRED.csv",
        "model_type": "CatBoost",
        "predict_module": "gwangan_predict"
    },
    "수원타임빌라스지점": {
        "pos_file": "suwon_POS.csv",
        "pred_file": "suwon_PRED.csv",
        "model_type": "CatBoost",
        "predict_module": "suwon_predict"
    },
    "연남점": {
        "pos_file": "yeonnam_POS.csv",
        "pred_file": "yeonnam_PRED.csv",
        "model_type": "CatBoost",
        "predict_module": "yeonnam_predict"
    }
}

def get_product_list_from_pos(store_name: str) -> List[str]:
    """지점 이름으로 POS 파일을 찾아 상품명 리스트를 반환합니다."""
    
    if store_name not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail=f"지점 '{store_name}' 설정을 찾을 수 없습니다.")

    pos_filename = STORE_CONFIG[store_name]["pos_file"]
    file_path = DATA_DIR / pos_filename
    
    if not file_path.exists():
        print(f"--- [ERROR] POS 파일 없음: {file_path} ---")
        return []

    try:
        # 데이터 로드 (validate.py의 load_data 로직을 간소화)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='euc-kr')

        # '상품명' 컬럼의 고유값 추출
        if '상품명' in df.columns:
            # 중복 제거 후 리스트로 반환 (NaN 값은 제외)
            products = df['상품명'].dropna().unique().tolist()
            # 정렬하여 반환
            return sorted(products)
        else:
            print(f"--- [ERROR] POS 파일 '{pos_filename}'에 '상품명' 컬럼이 없습니다. ---")
            return []
            
    except Exception as e:
        print(f"--- [FATAL ERROR] 상품 목록 로드 중 오류 발생: {type(e).__name__} - {e} ---")
        return []

# ============================================
# API 엔드포인트
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 시작(startup) 및 종료(shutdown) 이벤트를 처리합니다.
    """
    
    # 1. Startup 로직: 스케줄러 등록 및 시작
    scheduler.add_job(
        scheduled_forecast_job, 
        CronTrigger(day_of_week='tue', hour=12, minute=29),
        id='weekly_forecast', 
        name='주간 예측 실행'
    )
    scheduler.start()
    print("--- [INFO] APScheduler 시작 및 주간 예측 작업 등록 완료 (월요일 03:00) ---")

    # 애플리케이션이 이 지점에서 실행됩니다 (yield 이후는 종료 시 실행).
    yield
    
    # 2. Shutdown 로직: 스케줄러 종료
    if scheduler.running:
        scheduler.shutdown()
        print("--- [INFO] APScheduler 종료 완료 ---")

# fastAPI 앱 생성
app = FastAPI(
    title="Randy's Donuts Forecast API (Validation Focus)",
    description="머신러닝 기반 수요예측 시스템 API (검증 기능만 남김)",
    version="1.0.0",
    lifespan=lifespan
)



@app.get("/stores", response_model=StoreListResponse, tags=["Stores"])
async def get_stores():
    """지점 목록 조회"""
    stores = [
        {"name": name, "model": config["model_type"]}
        for name, config in STORE_CONFIG.items()
    ]
    return {"stores": stores}

@app.get("/products/{store_name}", response_model=List[str], tags=["Stores"])
async def get_products_by_store(store_name: str):
    """
    특정 지점의 POS 데이터에서 판매된 상품 목록을 조회합니다.
    """
    try:
        product_list = get_product_list_from_pos(store_name)
        
        if not product_list:
            # 파일이 없거나 컬럼이 없는 경우
            raise HTTPException(status_code=404, detail=f"지점 '{store_name}'의 상품 목록을 찾을 수 없습니다.")
            
        # React 코드에서 '(전체)'를 추가할 예정이므로, 여기서는 실제 목록만 반환합니다.
        return product_list
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상품 목록 조회 오류: {str(e)}")

@app.post("/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_predictions(query: ValidationQueryRequest):
    """
    검증 조회 엔드포인트
    - POS 데이터와 예측 데이터를 병합
    - R² 계산
    - 차트/테이블 데이터 반환
    """
    try:
        # 외부 함수 호출: validate.py의 run_validation 함수를 호출하여 검증 로직을 실행
        response = run_validation(query, STORE_CONFIG)
        return response
        
    except NotImplementedError as e:
         raise HTTPException(status_code=501, detail=f"검증 로직 미구현: {str(e)}")
    except HTTPException as e:
        # run_validation 내부에서 발생한 HTTPException을 그대로 반환
        raise e
    except Exception as e:
        # 그 외 예외 처리
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
    
@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def generate_forecast(request: ForecastRequest):
    """
    다음주 예측 생성 엔드포인트
    - 지점별 모델 로드 및 예측 실행
    - CSV 저장
    """
    print(f"--- [INFO] Forecast 요청: 지점={request.store_name}, 기준일={request.base_date} ---")
    
    try:
        # 1. 지점 설정 및 모듈 이름 가져오기
        if request.store_name not in STORE_CONFIG:
            raise HTTPException(status_code=404, detail="지점을 찾을 수 없습니다")
        
        config = STORE_CONFIG[request.store_name]
        module_name = config["predict_module"] 
        
        # 2. 지점별 예측 모듈 동적 로드 및 함수 참조
        try:
            # importlib을 사용하여 python/aewol_predict.py 모듈을 로드
            predict_module = importlib.import_module(module_name)
            predict_func = getattr(predict_module, "predict_next_week")
        except ImportError:
            raise HTTPException(status_code=501, detail=f"예측 모듈 '{module_name}.py'를 찾을 수 없습니다. python 폴더에 파일이 있는지 확인하세요.")
        except AttributeError:
            raise HTTPException(status_code=501, detail=f"예측 모듈 '{module_name}.py'에 'predict_next_week' 함수가 정의되지 않았습니다.")

        # 3. 예측 생성 실행
        # predict_next_week(store_name, base_date, horizon) 호출
        predictions_df = predict_func(
            store_name=request.store_name,
            base_date=request.base_date,
            horizon=request.horizon
        )
        
        # 4. CSV 저장
        base_date_str = request.base_date.strftime('%Y%m%d')
        filename = f"{request.store_name}_forecast_{base_date_str}.csv"
        csv_path = EXPORT_DIR / filename
        
        predictions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        print(f"--- [INFO] 예측 CSV 저장 완료: {csv_path} ---")

        # 5. 응답 데이터 구성 (PredictionRecord 모델에 맞춰)
        predictions = [
            PredictionRecord(
                date=row["날짜"], # predict_next_week의 결과 DF의 '날짜'가 str이라고 가정
                product_name=row["상품명"],
                predicted_qty=float(row["예측수량"]),
                order_qty_ceil=int(row["주문량_ceil"])
            )
            for _, row in predictions_df.iterrows()
        ]
        
        return ForecastResponse(
            store_name=request.store_name,
            base_date=request.base_date.strftime("%Y-%m-%d"),
            horizon=request.horizon,
            predictions=predictions,
            total_predictions=len(predictions),
            csv_filename=filename,
            status="success"
        )
        
    except HTTPException as e:
        raise e
    except requests.exceptions.RequestException as e:
        # 날씨 API 실패 등 외부 API 오류 처리
        raise HTTPException(status_code=503, detail=f"외부 API 오류 (날씨/데이터): {str(e)}")
    except Exception as e:
        print(f"--- [FATAL ERROR] 예측 생성 중 심각한 오류 발생: {type(e).__name__} - {e} ---")
        raise HTTPException(status_code=500, detail=f"예측 로직 내부 오류: {type(e).__name__} - {str(e)}")

@app.get("/download/{filename}", tags=["Download"])
async def download_csv(filename: str):
    """CSV 파일 다운로드"""
    file_path = EXPORT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv"
    )

# ============================================
# 스케줄러 함수 
# ============================================
def get_today() -> date:
    """오늘 날짜 반환"""
    # 오늘 날짜
    today = date.today()
    return today

async def scheduled_forecast_job():
    """매주 월요일 새벽 3시에 실행되어 모든 지점의 7일 예측을 수행합니다."""
    
    # 예측 기준일자 (실행되는 월요일 날짜)
    base_date = get_today()
    horizon = 7 # 7일 예측
    
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏰ 주간 수요 예측 스케줄 작업 시작")
    print(f"기준일자: {base_date.strftime('%Y-%m-%d')}, 예측 기간: {horizon}일")
    print("=" * 60)

    for store_name, config in STORE_CONFIG.items():
        try:
            print(f"--- [INFO] {store_name} 예측 시작...")
            
            module_name = config["predict_module"] 
            
            # 1. 예측 모듈 동적 로드 (main.py 상단에 importlib 필요)
            predict_module = importlib.import_module(module_name)
            predict_func = getattr(predict_module, "predict_next_week")

            # 2. 예측 실행
            predictions_df = predict_func(
                store_name=store_name,
                base_date=base_date,
                horizon=horizon
            )
            
            # 3. CSV 저장
            base_date_str = base_date.strftime('%Y%m%d')
            filename = f"{store_name}_weekly_forecast_{base_date_str}.csv"
            csv_path = RESULT_DIR / filename
            
            predictions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            
            print(f"✅ {store_name} 예측 완료 및 저장: {filename}")
            
        except Exception as e:
            print(f"❌ {store_name} 예측 실패: {type(e).__name__} - {e}")
            
    print("=" * 60)
    print("스케줄 작업 완료.")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)