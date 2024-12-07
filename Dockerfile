# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN pip install poetry

# Настраиваем Poetry
RUN poetry config virtualenvs.create false

# Копируем файлы Poetry для установки зависимостей
COPY pyproject.toml poetry.lock /app/

# Устанавливаем Python-зависимости через Poetry
RUN poetry install --no-interaction --no-ansi

# Устанавливаем MLflow
RUN pip install mlflow boto3

# Копируем весь исходный код проекта
COPY . /app

# Указываем порт, который будет слушать приложение
EXPOSE 8000 5000

# Позволяем выбирать, что запускать: приложение или MLflow
ENTRYPOINT ["sh", "-c"]
CMD ["if [ \"$1\" = 'mlflow' ]; then mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://mlflow/ --host 0.0.0.0 --port 5000; else uvicorn mlops_service.main:app --host 0.0.0.0 --port 8000; fi", "$@"]


