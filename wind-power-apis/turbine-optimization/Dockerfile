# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for xgboost/lightgbm wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL source code (train.py, app.py, models/, etc.)
COPY . /app

EXPOSE 8002

# Default = production API (Gunicorn + Uvicorn workers)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", \
     "--bind", "0.0.0.0:8002", "--workers", "2", "--timeout", "90"]
