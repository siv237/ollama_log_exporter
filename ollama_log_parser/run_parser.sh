#!/bin/bash

# Ollama Log Parser
# Запуск эталонного парсера логов

set -e

echo "🔍 Ollama Log Parser"
echo "==================="

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION найден"

# Проверяем права sudo
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  Парсер требует права sudo для доступа к journalctl"
    echo "🔄 Перезапуск с sudo..."
    exec sudo "$0" "$@"
fi

# Проверяем зависимости
echo "📦 Проверка зависимостей..."
if python3 -c "from dateutil import parser" 2>/dev/null; then
    echo "✅ python-dateutil найден"
else
    echo "❌ python-dateutil не найден"
    echo "📦 Установка python-dateutil..."
    pip3 install python-dateutil
    echo "✅ python-dateutil установлен"
fi

# Проверяем сервис Ollama
echo "🔍 Проверка сервиса Ollama..."
if systemctl is-active --quiet ollama; then
    echo "✅ Сервис Ollama активен"
else
    echo "⚠️  Сервис Ollama неактивен"
    echo "   Парсер все равно может анализировать старые логи"
fi

# Проверяем доступ к логам
echo "🔍 Проверка доступа к логам..."
if journalctl -u ollama --since "1 hour ago" --lines 1 &>/dev/null; then
    echo "✅ Доступ к логам Ollama есть"
else
    echo "❌ Нет доступа к логам Ollama"
    echo "   Проверьте, что сервис ollama существует"
    exit 1
fi

# Создаем папку для отчетов
mkdir -p reports

# Запускаем парсер
echo "📊 Запуск парсера..."
python3 parser.py

echo ""
echo "🎉 Готово! Отчеты созданы в папке reports/"
echo "📂 Список файлов:"
ls -la reports/ | tail -n +2

echo ""
echo "📖 Основной отчет: reports/report.md"
echo "💡 Для повторного запуска: sudo ./run_parser.sh"
