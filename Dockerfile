FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Создаем директорию для моделей
RUN mkdir -p /app/server/storage

EXPOSE 8000

# Используем полный путь к uvicorn из виртуального окружения
CMD ["/opt/venv/bin/uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]