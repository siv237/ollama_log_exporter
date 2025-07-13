#!/bin/bash
# Скрипт для корректного запуска экспортера в виртуальном окружении.

# Путь к директории с виртуальным окружением
VENV_DIR="venv"

# Проверка, существует ли venv, и создание, если нет
if [ ! -d "$VENV_DIR" ]; then
    echo "Создание виртуального окружения в '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Ошибка: не удалось создать виртуальное окружение."
        exit 1
    fi
fi

# Активация venv для установки зависимостей
source "$VENV_DIR/bin/activate"

# Установка/обновление зависимостей
echo "Установка зависимостей из requirements.txt..."
"$VENV_DIR/bin/pip" install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Ошибка: не удалось установить зависимости."
    exit 1
fi

deactivate

# Принудительно освобождаем порт, если он занят
echo "Проверка и освобождение порта 9877..."
sudo fuser -k 9877/tcp || true

# Запуск экспортера с правами sudo, но с использованием Python из venv
echo "Запуск Ollama Log Exporter с правами sudo..."
sudo "$VENV_DIR/bin/python3" exporter.py "$@"
