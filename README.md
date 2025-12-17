
## 🥐머신러닝 기반 베이커리 수요예측 시스템: Randy’s Donuts과의 공동 프로젝트

🔗 Demo URL: http://3.39.242.117/

본 프로젝트는 프랜차이즈 베이커리(협업사: Randy's Donuts)의 POS데이터를 기반으로  
일자별/상품별/점포별 수요를 예측하는 시스템을 개발하는 것을 목표로 합니다.  

점주의 '감'에 의존하여 과잉 발주와 과소 발주를 야기하던 발주량 결정에서 벗어나,  
데이터 기반 예측을 통해 과잉 발주·결품을 최소화하고 재고 운영의 안정성을 향상시키고자 합니다.

---

## 📂프로젝트 구조

```
eSeek/
├── backend/
│   ├── main.py     # FastAPI 앱 및 Lifespan(모델 로드) 정의
│   └── models.py   # Pydantic 데이터 모델 
├── data/
│   ├── auto_forecast_results/  # 스케줄러에 의해 주 단위 예측된 지점별 결과 자동 저장 
│   ├── processed/              # 전처리된 POS 데이터, 예측 결과 데이터 통합본 저장 
│   └── results/                # 사용자가 예측 실행 시 결과 저장 
├── frontend/
│   └── src/App.js              # 웹 대시보드 구조 및 API 호출 로직 
├── python/
│   ├── *_predict.py             # 각 지점별 예측 로직 
│   ├── prediction_utils.py      # 공통 피처링 유틸리티 
│   ├── *_model.pkl              # 학습된 모델 객체 (4개 지점)
│   ├── *_ct.pkl                 # 학습된 transformer 객체 (4개 지점)
│   └── validate.py              # 검증(validation) 로직 
├── nginx/
├── Dockerfile                   # [배포] 백엔드 엔진 및 Docker 이미지 빌드 
├── Dockerfile.web               # [배포] 프론트엔드/Nginx 이미지 빌드 
├── docker-compose.yml           # [배포] 서비스 통합 및 관리 정의(Nginx/FastAPI) 
├── requirements.txt            # python 가상환경 패키지 목록 
│
├── 스타트/                     # 초기(Proof-of-Concept) 단계
│   ├── data/                   # 원본/전처리 데이터
│   │   └── data_pos_raw_2023_2024.csv
│   │
│   ├── predictions/            # 스타트 단계 예측 및 분석 결과
│   │   ├── pred_lightgbm.csv
│   │   ├── pred_xgboost.csv
│   │   ├── pred_prophet.csv
│   │   └── analysis_sku_factors.csv
│   │
│   └── app/                    # 대시보드(UI) 코드
│       └── app_dashboard.py
│
├── 그로쓰/                     
│   ├── DEMO/
│   ├── 포스터.pdf
│   ├── 1차보고서
│   ├── 2차보고서
│   └── 최종보고서 
│
├── GroundRule.MD               # 협업 규칙
└── README.md                   # 전체 프로젝트 개요
```

---
## 🧠 소스코드 설명
- Backend (FastAPI)
    - FastAPI 기반 REST API
    - 서버 시작 시 지점별 모델을 메모리에 로드하여 저지연 예측 제공
    - 예측 요청 → Feature Engineering → 모델 추론 → 결과 저장/반환
- Prediction Engine (Python)
    - 지점별 특성을 반영한 독립 모델 구조
    - CatBoost / HGBR 모델 사용
    - 날씨·요일·휴일·시간대·과거 판매량 등 다차원 Feature Engineering
- Frontend (Web Dashboard)
    - React 기반 단일 페이지 대시보드
    - 날짜·점포·상품 선액 후 예측 결과 시각화
    - CSV 다운로드 기능 제공공
---

## 🔎주요 기능

- 데이터 전처리 : POS데이터와 외부변수(날씨, 요일, 공휴일, 프로모션)을 결합하여 전처리하는 기술
- 수요예측 엔진 : 머신러닝 모델(CatBoost, HGBR)을 활용하여 일자별/상품별/점포별 판매량을 예측하는 기능
- 결과 제공 : 일자별/상품별/점포별 수요예측치를 포함한 CSV파일을 주1회 협업사(Randy's Donuts)에 배포
- 웹 대시보드 : React 기반 대시보드를 통해 예측정확도 검증 결과, 수요예측 결과를 협업사에 제공

---
## 🛠 How to Build
코드를 처음 clone하면 프론트엔드 빌드 결과물이 없습니다. 아래 방법을 통해 프론트엔드를 먼저 빌드해야합니다.
<br />
Docker를 통해 자동 빌드하는 것도 가능하지만, 메모리 효율을 위해 프론트엔드는 미리 따로 빌드하는 것을 권장합니다.

```bash
cd frontend
npm install
npm run build
```

이후 Docker를 이용해 서비스를 통합 빌드합니다.

```bash
docker-compose build
```

---
## 📦 How to Install
빌드를 완료하면, Docker를 통해 시스템을 실행할 수 있습니다.

```bash
docker-compose up -d 
```

---
## 🧪 How to Test
시스템의 정상 작동 여부를 다음 세 가지 방법으로 확인할 수 있습니다.

1️⃣ 라이브 웹 접속 테스트 

접속 주소: http://3.39.242.117/

확인 사항: 지점 및 상품 선택 시 차트와 예측 결과 테이블이 정상적으로 렌더링되는지 확인합니다.  
<br />


2️⃣ 시나리오 기반 기능 테스트

시스템의 주요 기능을 아래 순서에 따라 단계별로 검증할 수 있습니다.

1) 지점 및 상품 선택: 좌측/상단 메뉴에서 [애월점] 지점을 선택한 후, 분석하고자 하는 [상품명]을 지정합니다.

2) 과거 데이터 및 모델 검증: [조회 시작일]과 [조회 종료일]을 선택하고 [조회] 버튼을 클릭합니다. 선택한 기간의 실제 판매 추이와 모델의 검증 결과를 확인합니다.

- 참고: 해당 기간 내 판매 이력이 없는 상품은 데이터 부족으로 인해 결과가 반환되지 않을 수 있습니다.

3) 수요 예측 실행: 예측의 기준이 되는 [기준일자]를 선택하고 [다음주 예측 생성하기] 버튼을 누릅니다.

4) 결과 확인 및 안정성 검증: 서버로부터 계산된 차기 주간 예측 데이터($\hat{y}$)를 성공적으로 수신하여 대시보드에 정상적으로 렌더링되는지 확인합니다.

- 이 과정은 대량의 추론 연산을 포함하므로, Nginx의 타임아웃(600s) 설정이 정상적으로 작동하여 연결이 유지되는지 함께 검증하는 핵심 단계입니다.


---
## 📊 Description of Sample Data
- POS 데이터
    - 날짜
    - 상품명
    - 판매 수량
- 외부 변수
    - 날씨 (기온, 강수량)
    - 요일 / 주말 여부
    - 공휴일 여부
    - 이벤트 정보
---
## 🗄 Database / Data Used
- CSV 기반 데이터 저장 구조
- 예측 결과는 서버 로컬 디렉토리에 자동 저장
---
## 🔓 Used Open Source
- FastAPI: Backend API
- CatBoost: 수요예측 모델
- scikit-learn: 전처리 및 평가
- pandas / numpy: 데이터 처리
- React: Frontend
- Docker: 배포 환경
- Nginx: Reverse Proxy
---
## 🚀 DEMO 
### 1️⃣ 중간 단계 [Prototype]: Streamlit 기반 모델 검증

본 Streamlit 앱은  최종 서비스 개발 이전 단계에서 **모델 성능 검증 및 데이터 흐름 확인을 위한 프로토타입**입니다.

1. Python 3.10 이상 환경에서 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. Streamlit 앱 실행:

```bash
streamlit run app/main.py
```

### 2️⃣ 최종 단계 [Production]: React + FastAPI 웹 서비스

본 단계는 **실제 점포 운영 환경에서 사용 가능한 형태의 웹 서비스**로, 모델 예측 결과를 **사용자 친화적인 UI를 통해 제공**하는 것을 목표로 합니다.
- Frontend: React 기반 대시보드
- Backend: FastAPI 기반 예측 API 서버
- 배포 환경: Docker + Nginx

실행방법 (Docker 권장)
```bash
docker-compose up --build -d
```
- 웹 서비스 접속: 사용자가 직접 이용하는 React 기반 웹 화면
- 백엔드 API: 예측 모델이 동작하는 FastAPI 서버 (프론트엔드에서 호출)
---

## 👥 팀 정보

- 프로젝트명: 머신러닝 기반 베이커리 수요예측 시스템 : Randy's Donuts과의 공동 프로젝트
- 팀번호/팀명: 13팀/e시크 (이화여자대학교 컴퓨터공학과 2025 캡스톤디자인과 창업프로젝트)

---

## 💌 문의

프로젝트와 관련된 문의는 아래 이메일로 부탁드립니다.

> jellyjw01@ewha.ac.kr
