#!/bin/bash

# 1. Отправка исходного файла в S3
bash lr3/bash_scripts/upload.sh

# 2. Загрузка исходного файла из S3
bash lr3/bash_scripts/download.sh

# 3. Обработка исходного файла
bash lr3/bash_scripts/process.sh

# 4. Загрузка обработанного файла в S3
bash lr3/bash_scripts/upload_processed.sh