
# 프랜차이즈 베이커리 수요 예측 및 발주 자동화 시스템

이 프로젝트는 프랜차이즈 베이커리 가맹점에서 발생하는 POS 데이터를 기반으로,
품목별 수요를 예측하고 발주량을 자동으로 계산해주는 시스템을 개발하는 것을 목표로 합니다.

점주의 경험에만 의존하던 기존 발주 방식에서 벗어나,
정량적인 예측 모델을 바탕으로 보다 효율적이고 안정적인 재고 관리를 지원합니다.

---

## 프로젝트 구조

```
├── preprocessing/         # POS 데이터 전처리 및 피처 생성
├── models/                # LightGBM / Prophet 모델 학습 및 예측 코드
├── app/                   # Streamlit 기반 대시보드 UI 코드
├── data/                  # 예시 POS 판매 데이터 (익명 처리)
├── utils/                 # 공휴일 및 날씨 API 연동 관련 유틸 함수
├── requirements.txt       # 필요 라이브러리 목록
├── README.md              # 현재 문서
```

---

## 주요 기능

- POS 데이터 기반 품목별 일일 수요 예측 (LightGBM / Prophet)
- 공휴일 및 날씨 정보 자동 반영
- 최소 발주 단위(MOQ), 버퍼율 적용 발주 수량 계산
- 웹 기반 UI에서 날짜/품목 선택 → 예측 결과 확인 및 다운로드

---

## 실행 방법

1. Python 3.10 이상 환경에서 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. Streamlit 앱 실행:

```bash
streamlit run app/main.py
```

---

## 팀 정보

- 프로젝트명: eSeek
- 팀명: 13조 (이화여자대학교 컴퓨터공학과 캡스톤디자인)
- 데이터 출처: Kaggle - French Bakery Daily Sales (시뮬레이션용)

---

## 문의

프로젝트와 관련된 문의는 아래 이메일로 부탁드립니다.

> jellyjw01@ewha.ac.kr
