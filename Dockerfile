# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Создание директорий для моделей и конфигов
RUN mkdir -p models/backups config

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:///./clinical_data.db

# Запуск приложения
CMD ["python", "main.py"]
