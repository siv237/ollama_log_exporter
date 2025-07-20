#!/bin/bash
"""
Скрипт для установки Ollama InfluxDB Integration как systemd сервис.
"""

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверяем что скрипт запущен от root
if [[ $EUID -ne 0 ]]; then
   log_error "Этот скрипт должен быть запущен от root"
   exit 1
fi

# Определяем пути
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/ollama-influxdb-integration"
SERVICE_NAME="ollama-influxdb-integration"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

log_info "Установка Ollama InfluxDB Integration как systemd сервис"
log_info "Директория скрипта: $SCRIPT_DIR"
log_info "Директория установки: $INSTALL_DIR"

# Создаем директорию установки
log_info "Создание директории установки..."
mkdir -p "$INSTALL_DIR"

# Копируем файлы
log_info "Копирование файлов..."
cp "$SCRIPT_DIR/main.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/influxdb_writer.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/config.json" "$INSTALL_DIR/"

# Копируем парсер из родительской директории
if [[ -f "$SCRIPT_DIR/../parser.py" ]]; then
    cp "$SCRIPT_DIR/../parser.py" "$INSTALL_DIR/"
    log_info "Парсер скопирован"
else
    log_warn "Файл parser.py не найден в родительской директории"
fi

# Устанавливаем права
chmod +x "$INSTALL_DIR/main.py"
chmod 644 "$INSTALL_DIR/config.json"

# Создаем пользователя для сервиса (если не существует)
if ! id "ollama-integration" &>/dev/null; then
    log_info "Создание пользователя ollama-integration..."
    useradd --system --no-create-home --shell /bin/false ollama-integration
else
    log_info "Пользователь ollama-integration уже существует"
fi

# Устанавливаем владельца
chown -R ollama-integration:ollama-integration "$INSTALL_DIR"

# Создаем директорию для логов
mkdir -p /var/log/ollama-influxdb-integration
chown ollama-integration:ollama-integration /var/log/ollama-influxdb-integration

# Создаем systemd service файл
log_info "Создание systemd service файла..."
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Ollama InfluxDB Integration Service
Documentation=https://github.com/your-repo/ollama-influxdb-integration
After=network.target influxdb.service
Wants=influxdb.service

[Service]
Type=simple
User=ollama-integration
Group=ollama-integration
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/main.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ollama-influxdb-integration

# Ограничения ресурсов
MemoryMax=512M
CPUQuota=50%

# Безопасность
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$INSTALL_DIR /var/log/ollama-influxdb-integration /tmp

[Install]
WantedBy=multi-user.target
EOF

# Перезагружаем systemd
log_info "Перезагрузка systemd daemon..."
systemctl daemon-reload

# Включаем сервис
log_info "Включение сервиса для автозапуска..."
systemctl enable "$SERVICE_NAME"

log_info "✅ Установка завершена!"
echo
log_info "Следующие шаги:"
echo "1. Отредактируйте конфигурацию: $INSTALL_DIR/config.json"
echo "2. Запустите сервис: systemctl start $SERVICE_NAME"
echo "3. Проверьте статус: systemctl status $SERVICE_NAME"
echo "4. Просмотр логов: journalctl -u $SERVICE_NAME -f"
echo
log_warn "ВАЖНО: Не забудьте настроить токен InfluxDB в config.json!"