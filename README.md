
## 🥐머신러닝 기반 베이커리 수요예측 시스템: Randy’s Donuts과의 공동 프로젝트

이 프로젝트는 프랜차이즈 베이커리 가맹점의 POS 데이터를 기반으로  
품목별 수요를 예측하는 시스템을 개발하는 것을 목표로 합니다.  

기존처럼 점주의 경험과 직관에만 의존하던 발주 방식에서 벗어나,  
데이터 기반의 정량적 예측 모델을 통해 보다 효율적이고 안정적인 재고 관리를 지원합니다.

---

## 📂프로젝트 구조

```
eSeek/
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
├── 그로쓰/                     # (추후) 고도화 단계
│   ├── data/
│   ├── predictions/
│   ├── models/
│   └── app/
│
├── GroundRule.MD               # 협업 규칙
├── README.md                   # 전체 프로젝트 개요
└── requirements.txt            # (필요 시) 공통 패키지 목록
```

---

## 🔎주요 기능

- POS 데이터 기반 품목별 일일 수요 예측 (CatBoost 등 ML 모델 활용)
- 공휴일 및 날씨 정보 자동 반영
- 웹 기반 UI에서 날짜/품목 선택 후 예측 결과 확인 및 다운로드

---

## 🚀실행 방법

1. Python 3.10 이상 환경에서 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. Streamlit 앱 실행:

```bash
streamlit run app/main.py
```

---

## 👥팀 정보

- 프로젝트명: eSeek
- 팀번호: 13팀 (이화여자대학교 컴퓨터공학과 캡스톤디자인)

---

## 💌문의

프로젝트와 관련된 문의는 아래 이메일로 부탁드립니다.

> jellyjw01@ewha.ac.kr
