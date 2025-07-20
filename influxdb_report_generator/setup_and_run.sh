#!/bin/bash

# Ollama InfluxDB Report Generator
# Автоматическая установка и запуск генератора отчетов

set -e

echo "🚀 Ollama InfluxDB Report Generator"
echo "=================================="

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION найден"

# Создаем виртуальное окружение
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
fi

# Активируем виртуальное окружение
echo "🔄 Активация виртуального окружения..."
source venv/bin/activate

# Обновляем pip
echo "📦 Обновление pip..."
pip install --upgrade pip > /dev/null 2>&1

# Устанавливаем зависимости
echo "📦 Установка зависимостей..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ Зависимости установлены"

# Проверяем конфигурацию
if [ ! -f "config.json" ]; then
    echo "⚠️  Файл config.json не найден"
    echo "📝 Создан пример конфигурации. Отредактируйте config.json"
    echo "   Укажите правильные параметры InfluxDB:"
    echo "   - url: адрес InfluxDB сервера"
    echo "   - token: токен доступа"
    echo "   - org: организация"
    echo "   - bucket: имя bucket"
    exit 1
fi

# Создаем папку для отчетов
mkdir -p reports

# Проверяем подключение к InfluxDB
echo "🔍 Проверка подключения к InfluxDB..."
if python3 -c "
import json
import requests
import sys

with open('config.json', 'r') as f:
    config = json.load(f)

url = config['influxdb']['url']
token = config['influxdb']['token']

try:
    health_url = f'{url}/health'
    response = requests.get(health_url, timeout=5)
    if response.status_code == 200:
        print('✅ InfluxDB доступен')
    else:
        print('❌ InfluxDB недоступен')
        sys.exit(1)
except Exception as e:
    print(f'❌ Ошибка подключения: {e}')
    sys.exit(1)
"; then
    echo "✅ Подключение к InfluxDB успешно"
else
    echo "❌ Не удалось подключиться к InfluxDB"
    echo "   Проверьте настройки в config.json"
    exit 1
fi

# Запускаем генератор отчетов
echo "📊 Генерация отчета..."
python3 report_generator.py

echo ""
echo "🎉 Готово! Отчет сгенерирован в папке reports/"
echo "📂 Список файлов:"
ls -la reports/ | tail -n +2

echo ""
echo "💡 Для повторного запуска:"
echo "   cd influxdb_report_generator"
echo "   source venv/bin/activate"
echo "   python3 report_generator.py"
