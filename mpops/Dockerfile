FROM python:3.12-slim

# Обновим pip и системные зависимости
RUN apt-get update && apt-get install -y curl && pip install --upgrade pip

# Установка poetry из официального скрипта
RUN curl -sSL https://install.python-poetry.org | python3 -

# Добавляем poetry в PATH
ENV PATH="/root/.local/bin:$PATH"

# Установка зависимостей проекта
WORKDIR /app
COPY . .


RUN poetry config virtualenvs.create false && poetry install --only main


CMD ["tail", "-f", "/dev/null"]
