#!/bin/bash

# Чтобы пути не были преобразованы в Windows стиле
export MSYS_NO_PATHCONV=1

# Загрузка переменных окружения
set -a
source .env
set +a

# Параметры скрипта
DATA_PATH="./data/processed/processed.csv"  # Путь к данным внутри контейнера
PARAMS_SOURCE="params.yml"  # Исходный файл параметров
BASE_EXP_NAME="svc-exp"     # Базовое имя эксперимента
PARAMS_DIR="./temp_params"  # Директория для временных конфигов

# Создаем директорию для временных конфигов
mkdir -p "$PARAMS_DIR"

if [ ! -f "$PARAMS_SOURCE" ]; then
    echo "Error: params.yml not found!"
    exit 1
fi

if [ ! -f "data/processed/processed.csv" ]; then
    echo "Error: processed.csv not found!"
    exit 1
fi

# Парсим степени через OmegaConf и генерируем конфиги
echo "Starting Python script to generate configs..."
poetry run python bash_scripts/generate_config.py "$PARAMS_SOURCE" "$PARAMS_DIR"

# Проверка созданных конфигураций
if [ ! -d "$PARAMS_DIR" ] || [ -z "$(ls -A "$PARAMS_DIR")" ]; then
    echo "Error: No config files generated!"
    exit 1
fi

# Основной цикл по параметрам
for config in "${PARAMS_DIR}"/params_degree*.yml; do
    degree=$(basename "$config" | grep -oP '(?<=degree)\d+')
    echo "Running experiment with degree=$degree"

    # Запускаем Docker-контейнер с текущим конфигом
    docker run --rm \
               --env-file .env \
               -v "$(pwd)/data:/app/data" \
               -v "$(pwd)/reports:/app/reports" \
               -v "$(pwd)/models:/app/models" \
               -v "$(pwd)/temp_params:/app/temp_params" \
               --network host \
               training-image:latest \
               python ml_ops/modeling/train.py \
               --params "/app/$PARAMS_DIR/params_degree${degree}.yml" \
               --data_path "/app/$DATA_PATH" \
               --experiment_name "${BASE_EXP_NAME}-degree${degree}"
done

# Очистка конфигов
rm -rf "$PARAMS_DIR"
