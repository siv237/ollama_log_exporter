# Ollama InfluxDB Integration

Интеграция парсера логов Ollama с InfluxDB для визуализации в Grafana.

## Архитектура решения

```
Ollama Logs → Parser → InfluxDB Writer → InfluxDB → Grafana
```

## Сервер: 192.168.237.198

### Проверка сервера

Проверяем доступность и состояние сервера:

```bash
ping 192.168.237.198
ssh root@192.168.237.198
```

### Текущее состояние сервера

- **IP:** 192.168.237.198
- **Доступ:** SSH root ✅
- **ОС:** Ubuntu 24.04 LTS (Noble Numbat)
- **Ядро:** Linux 6.8.12-11-pve (Proxmox VE)
- **RAM:** 2.0 GB (доступно 1.5 GB)
- **Диск:** 7.8 GB (использовано 4.0 GB, свободно 3.4 GB)
- **Grafana:** ✅ Работает на порту 3000
- **Prometheus:** ✅ Работает на порту 9090  
- **InfluxDB:** ❌ НЕ установлен (порт 8086 свободен)
- **Docker:** НЕ используем (по требованию)

### План установки InfluxDB

#### 1. Проверка системы
```bash
# Проверяем ОС и ресурсы
uname -a
cat /etc/os-release
df -h
free -h
systemctl status grafana-server
```

#### 2. Установка InfluxDB (нативно, без Docker)

**Для Ubuntu/Debian:**
```bash
# Добавляем репозиторий InfluxDB
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133fddaf92e15b16e6ac9ce4c6 influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

# Устанавливаем InfluxDB
sudo apt-get update && sudo apt-get install influxdb2
```

**Для CentOS/RHEL:**
```bash
# Добавляем репозиторий
cat <<EOF | sudo tee /etc/yum.repos.d/influxdata.repo
[influxdata]
name = InfluxData Repository - Stable
baseurl = https://repos.influxdata.com/stable/\$basearch/main
enabled = 1
gpgcheck = 1
gpgkey = https://repos.influxdata.com/influxdata-archive_compat.key
EOF

# Устанавливаем
sudo yum install influxdb2
```

#### 3. Настройка InfluxDB
```bash
# Запускаем сервис
sudo systemctl enable influxdb
sudo systemctl start influxdb
sudo systemctl status influxdb

# Проверяем порт
sudo netstat -tlnp | grep 8086
```

#### 4. Первоначальная настройка
```bash
# Настройка через веб-интерфейс
# http://192.168.237.198:8086

# Или через CLI
influx setup \
  --username admin \
  --password your-password \
  --org ollama-monitoring \
  --bucket ollama-logs \
  --retention 30d \
  --force
```

### Структура данных InfluxDB

#### Measurements (таблицы):

1. **ollama_sessions** - сессии моделей
2. **ollama_requests** - API запросы  
3. **ollama_system_events** - системные события
4. **ollama_model_info** - информация о моделях

#### Схема данных:

```python
# ollama_sessions
{
    "measurement": "ollama_sessions",
    "tags": {
        "model": "gemma3:12b",
        "session_id": "session_1_3085",
        "pid": "3085",
        "state": "loading|active|idle",
        "gpu": "cuda",
        "client_ip": "172.17.0.3"
    },
    "fields": {
        "ctx_size": 2048,
        "batch_size": 512,
        "gpu_layers": 48,
        "threads": 6,
        "parallel": 1,
        "vram_available_bytes": 11726832640,
        "vram_required_bytes": 1932735283,
        "loading_duration_seconds": 5.52,
        "model_layers_total": 27,
        "model_layers_offloaded": 27
    },
    "time": "2025-07-18T11:59:04Z"
}

# ollama_requests
{
    "measurement": "ollama_requests", 
    "tags": {
        "model": "gemma3:12b",
        "session_id": "session_1_3085",
        "method": "POST",
        "endpoint": "/api/chat",
        "status": "200",
        "client_ip": "172.17.0.3"
    },
    "fields": {
        "latency_seconds": 8.129942112,
        "processing_start_timestamp": 1705567144,
        "processing_end_timestamp": 1705567152
    },
    "time": "2025-07-18T11:59:12Z"
}
```

### Интеграция с парсером

Создадим новый компонент `influxdb_writer.py` который будет:

1. Читать данные из парсера
2. Преобразовывать в формат InfluxDB
3. Записывать в базу данных
4. Поддерживать batch операции

### Grafana Dashboard

После установки InfluxDB настроим в Grafana:

1. **Data Source:** InfluxDB v2
2. **Панели:**
   - State Timeline для состояний моделей
   - Gantt Chart для сессий
   - Request Rate графики
   - Latency Heatmap
   - Resource Usage (VRAM, GPU)

## Установка и настройка

### Шаг 1: Установка InfluxDB

На сервере 192.168.237.198 выполните:

```bash
# Для Ubuntu/Debian
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133fddaf92e15b16e6ac9ce4c6 influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update && sudo apt-get install influxdb2

# Запуск сервиса
sudo systemctl enable influxdb
sudo systemctl start influxdb
sudo systemctl status influxdb
```

### Шаг 2: Первоначальная настройка InfluxDB

Откройте веб-интерфейс: http://192.168.237.198:8086

Или через CLI:
```bash
influx setup \
  --username admin \
  --password your-secure-password \
  --org ollama-monitoring \
  --bucket ollama-logs \
  --retention 30d \
  --force
```

Создайте API токен:
```bash
influx auth create \
  --org ollama-monitoring \
  --all-access \
  --description "Ollama Integration Token"
```

### Шаг 3: Настройка интеграции

1. Отредактируйте `config.json`:
```json
{
  "influxdb": {
    "url": "http://192.168.237.198:8086",
    "token": "ваш-токен-influxdb",
    "org": "ollama-monitoring", 
    "bucket": "ollama-logs"
  },
  "ollama": {
    "manifests_path": "/root/.ollama/models/manifests",
    "log_hours": 24
  },
  "integration": {
    "interval_seconds": 300,
    "batch_size": 1000
  }
}
```

2. Установите зависимости Python:
```bash
pip install requests python-dateutil
```

### Шаг 4: Запуск интеграции

```bash
# Проверка подключений
python3 main.py --test

# Один цикл сбора данных
python3 main.py --once

# Непрерывный мониторинг
python3 main.py
```

## Использование

### Структура данных в InfluxDB

#### Measurements:

1. **ollama_sessions** - сессии моделей
   - Tags: model, session_id, pid, gpu_library, model_sha256
   - Fields: ctx_size, batch_size, gpu_layers, threads, parallel, vram_available_bytes, vram_required_bytes, loading_duration_seconds, model_layers_total, model_layers_offloaded

2. **ollama_requests** - API запросы
   - Tags: model, session_id, method, endpoint, status, client_ip
   - Fields: latency_seconds

3. **ollama_system_events** - системные события
   - Tags: event_type, event_name
   - Fields: details, value, дополнительные поля в зависимости от типа события

### Примеры запросов InfluxDB

```flux
// Средняя задержка запросов по моделям
from(bucket: "ollama-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "ollama_requests")
  |> filter(fn: (r) => r._field == "latency_seconds")
  |> group(columns: ["model"])
  |> mean()

// Использование VRAM по времени
from(bucket: "ollama-logs")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "ollama_sessions")
  |> filter(fn: (r) => r._field == "vram_required_bytes")
  |> aggregateWindow(every: 5m, fn: last)

// Количество запросов по эндпоинтам
from(bucket: "ollama-logs")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "ollama_requests")
  |> group(columns: ["endpoint"])
  |> count()
```

## Настройка Grafana

1. Добавьте InfluxDB как источник данных:
   - URL: http://192.168.237.198:8086
   - Organization: ollama-monitoring
   - Token: ваш-токен-influxdb
   - Default Bucket: ollama-logs

2. Импортируйте готовые дашборды из папки `../dashboards/`

## Мониторинг и логи

Логи интеграции можно просматривать в реальном времени:
```bash
# Если запущено как сервис
journalctl -u ollama-influxdb-integration -f

# Если запущено вручную
python3 main.py 2>&1 | tee integration.log
```

## Устранение неполадок

### InfluxDB недоступен
```bash
# Проверка статуса
sudo systemctl status influxdb
sudo netstat -tlnp | grep 8086

# Перезапуск
sudo systemctl restart influxdb
```

### Ошибки авторизации
- Проверьте токен в config.json
- Убедитесь что токен имеет права на запись в bucket

### Нет данных в Grafana
- Проверьте что интеграция запущена и работает
- Убедитесь что данные записываются в InfluxDB
- Проверьте настройки источника данных в Grafana

## TODO

- [x] Создать InfluxDB Writer
- [x] Интегрировать с парсером
- [x] Создать основной скрипт интеграции
- [ ] Установить InfluxDB на сервер
- [ ] Настроить базу данных
- [ ] Протестировать интеграцию
- [ ] Создать Grafana Dashboard
- [ ] Настроить автозапуск как сервис