# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# requirements.txt 파일 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사 (backend, data, python 등)
COPY . /app

# 환경 변수 설정
ENV PYTHONUNBUFFERED 1
ENV TZ Asia/Seoul

# Gunicorn (프로덕션 웹 서버)으로 FastAPI 앱 구동
# backend/main.py 파일의 app 객체를 실행
CMD ["gunicorn", "backend.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]