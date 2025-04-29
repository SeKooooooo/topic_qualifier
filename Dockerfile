FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .


# Создаем необходимые папки
RUN mkdir -p /app/uploads 

ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

EXPOSE 5000

CMD ["python", "app.py"]