#!/bin/bash

poetry install
docker-compose up --build -d

# 1. Отправка исходного файла в S3
bash bash_scripts/upload.sh

# 2. Загрузка исходного файла из S3
bash bash_scripts/download.sh

# 3. Обработка исходного файла
bash bash_scripts/process.sh

# 4. Загрузка обработанного файла в S3
bash bash_scripts/upload_processed.sh

# 5. Обучение и сохранение модели SVC
bash bash_scripts/train.sh