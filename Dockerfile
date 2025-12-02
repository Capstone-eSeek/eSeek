# Dockerfile: 단일 requirements.txt를 사용하여 패키지 설치

# 1. 베이스 이미지 설정: 경량화된 Python 3.11 버전 사용
FROM python:3.11-slim

# AWS 서버 내부에서 작업할 디렉토리 설정
WORKDIR /app

# 2. requirements.txt 파일만 먼저 복사합니다. (가장 캐시 효율이 높은 방식)
COPY requirements.txt ./

# 3. 모든 패키지 설치 (gunicorn, uvicorn 포함)
# 디스크 공간 확보 후이므로 이 단계는 성공해야 합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 모든 프로젝트 파일과 데이터를 복사합니다.
# 이 단계는 패키지 설치가 완료된 후에 진행되어야 합니다.
COPY . /app

# 환경 변수 설정
ENV PYTHONUNBUFFERED 1
ENV TZ Asia/Seoul

# 5. Gunicorn (프로덕션 웹 서버)으로 FastAPI 앱 구동
CMD ["gunicorn", "backend.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]