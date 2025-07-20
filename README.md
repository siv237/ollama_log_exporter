# Ollama Log Exporter

Три независимых инструмента для работы с логами Ollama:
1. **Ollama Log Parser** - эталонный анализатор логов с детальными отчетами
2. **InfluxDB Integration** - система мониторинга для Grafana
3. **InfluxDB Report Generator** - автономный генератор отчетов из InfluxDB

## 🎯 Назначение

### 🔍 Parser (parser.py)
**Независимый инструмент** для получения максимально подробного анализа работы Ollama.

Создает детальные отчеты в формате Markdown для:
- 📊 Анализа производительности моделей
- 🔧 Отработки алгоритмов обработки данных
- 🐛 Диагностики проблем в работе Ollama
- 📈 Исследования паттернов использования

### 🔗 InfluxDB Integration
**Отдельная система мониторинга** для записи данных в InfluxDB и визуализации в Grafana.

Использует алгоритмы парсинга для:
- ⚡ Реального времени мониторинга
- 📊 Создания дашбордов в Grafana
- 🔄 Непрерывного сбора метрик

## 🏗️ Архитектура

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ollama Logs   │───▶│    Parser    │───▶│   Reports    │
│  (journalctl)   │    │ (parser.py)  │    │ (Markdown)   │
└─────────────────┘    └──────────────┘    └──────────────┘
                              
┌─────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Ollama Logs   │───▶│  InfluxDB   │───▶│  InfluxDB   │───▶│   Grafana   │
│  (journalctl)   │    │ Integration │    │  Database   │    │ Dashboards  │
└─────────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 📁 Структура проекта

```
├── ollama_log_parser/                  # 🔍 Эталонный парсер логов
│   ├── parser.py                      # Основной парсер (937 строк)
│   ├── run_parser.sh                  # Скрипт запуска
│   ├── reports/                       # Отчеты парсера
│   │   ├── report.md                  # Детальный анализ сессий
│   │   ├── ollama_log_dump.txt       # Полный дамп логов
│   │   ├── analysis_filtered_log.txt # Очищенные логи
│   │   └── startup_info.txt          # Информация о запуске
│   └── README.md                      # Документация парсера
├── influxdb_report_generator/          # 🚀 Автономный генератор отчетов
│   ├── setup_and_run.sh               # Установка и запуск
│   ├── report_generator.py            # Генератор из InfluxDB
│   ├── config.json                    # Конфигурация InfluxDB
│   ├── requirements.txt               # Python зависимости
│   ├── venv/                          # Виртуальное окружение
│   └── reports/                       # Сгенерированные отчеты
├── ollama_influxdb_integration/        # 🔗 Интеграция с InfluxDB
│   ├── main.py                        # Основной скрипт интеграции
│   ├── influxdb_writer.py             # Запись данных в InfluxDB
│   ├── config.json                    # Конфигурация
│   ├── install_service.sh             # Установка как systemd сервис
│   └── dashboards/                    # 📈 Дашборды Grafana
│       ├── ollama_gantt_simple.json   # Диаграмма Ганта
│       └── README.md                  # Инструкции по дашбордам
└── README.md                          # Этот файл
```

## 🚀 Быстрый старт

### 🔍 Эталонный парсер логов

```bash
cd ollama_log_parser
./run_parser.sh

# Результат:
# ✅ reports/report.md - детальный анализ всех сессий
# ✅ reports/ollama_log_dump.txt - полный дамп логов
# ✅ reports/analysis_filtered_log.txt - очищенные логи
# ✅ reports/startup_info.txt - информация о запуске
```

### 🚀 Генератор отчетов из InfluxDB

```bash
cd influxdb_report_generator
./setup_and_run.sh

# Результат:
# ✅ Автоматическая установка зависимостей
# ✅ Проверка подключения к InfluxDB
# ✅ Генерация отчета в reports/
```

### 🔗 Использование InfluxDB Integration

```bash
cd ollama_influxdb_integration

# Проверка подключений к InfluxDB
python3 main.py --test

# Один цикл сбора и записи данных
python3 main.py --once

# Непрерывный мониторинг (каждые 10 секунд)
python3 main.py
```

### 🛠️ Установка интеграции как сервис

```bash
cd ollama_influxdb_integration
sudo ./install_service.sh

# Управление сервисом
sudo systemctl start ollama-influxdb-integration
sudo systemctl status ollama-influxdb-integration
sudo journalctl -u ollama-influxdb-integration -f
```

## 📊 Что анализирует парсер

### 🔍 Детальная информация о сессиях:

- **Модели**: название, SHA256, размер, архитектура
- **Ресурсы**: использование VRAM, GPU layers, threads
- **Производительность**: время загрузки, latency запросов
- **API активность**: все HTTP запросы с временными метками
- **Системные события**: запуск/остановка сервисов

### 📈 Пример отчета:

```markdown
## Сессия 1: gemma3:12b

*   **Время старта сессии:** 2025-07-20T14:40:09+10:00
*   **PID процесса Ollama:** 3061
*   **Модель:** gemma3:12b
*   **SHA256:** adca500fad9b54c565ae672184e0c9eb690eb6014ba63f8ec13849d4f73a32d3
*   **GPU layers:** 48
*   **CTX size:** 2048
*   **Размер модели:** 7 GB

### Свободная память при старте:
*   **VRAM свободно на GPU:** [10.6 GiB] (требуется 1.8 GiB)

### Обслуженные API-запросы:
| Время | Статус | Задержка | IP | Метод | Путь |
|---|---|---|---|---|---|
| 14:40:22 | 200 | 13.524s | 172.17.0.3 | POST | /api/chat |
| 14:40:28 | 200 | 5.535s | 172.17.0.3 | POST | /api/chat |
```

## 🔧 Конфигурация InfluxDB

### Сервер: 192.168.237.198

```json
{
  "influxdb": {
    "url": "http://192.168.237.198:8086",
    "token": "ваш-токен-influxdb",
    "org": "ollama-monitoring",
    "bucket": "ollama-logs"
  },
  "integration": {
    "interval_seconds": 10,
    "batch_size": 1000
  }
}
```

### Структура данных в InfluxDB:

#### Measurements:
- **`ollama_sessions`** - сессии моделей
- **`ollama_requests`** - API запросы  
- **`ollama_system_events`** - системные события

#### Теги:
- `model`, `session_id`, `pid`, `gpu_library`
- `method`, `endpoint`, `status`, `client_ip`

#### Поля:
- `latency_seconds`, `vram_available_bytes`, `loading_duration_seconds`
- `ctx_size`, `batch_size`, `gpu_layers`, `threads`

## 📈 Grafana Dashboards

### Диаграмма Ганта моделей

Импорт дашборда:
1. Grafana → Dashboards → Import
2. Upload `ollama_influxdb_integration/dashboards/ollama_gantt_simple.json`
3. Выбрать источник данных "Ollama InfluxDB"

**Показывает:**
- 🟠 Время загрузки моделей (оранжевые полосы)
- 🟢 Момент готовности (зеленые точки)
- ⏱️ Длительность операций
- 🔄 Переключения между моделями

## 🛠️ Проверка полноты данных в InfluxDB

### 🔍 Проверка что все данные попали в InfluxDB без потерь:

```bash
# Полная проверка данных в InfluxDB
python3 check_influxdb_data.py

# Результат:
# ✅ Проверка measurements, полей, тегов
# ✅ Сравнение количества сессий и запросов с отчетом парсера
# ✅ Генерация Flux запросов для восстановления данных
# 💾 Сохранение запросов в influxdb_reconstruction_queries.flux
```

### 📊 Генерация идентичного отчета из InfluxDB:

```bash
# Создание отчета прямо из базы данных
python3 generate_report_from_influxdb.py

# Результат:
# 📄 reports/influxdb_report_TIMESTAMP.md - полный отчет из InfluxDB
# 📊 Идентичная структура с отчетом парсера
# ✅ Все данные восстановлены без потерь
```

### 🔧 Отладка и проверка:

#### 1. Проверка парсера:
```bash
# Запуск с детальным выводом
sudo python3 parser.py

# Проверка отчета
cat reports/report.md | head -50
```

#### 2. Проверка InfluxDB:
```bash
# Тест подключения
python3 ollama_influxdb_integration/main.py --test

# Полная проверка данных
python3 check_influxdb_data.py

# Сравнение отчетов
diff reports/report.md reports/influxdb_report_*.md
```

#### 3. Проверка Grafana:
- Откройте http://192.168.237.198:3000
- Проверьте источник данных "Ollama InfluxDB"
- Убедитесь что дашборд показывает данные

## 🔍 Типичные проблемы и решения

### Парсер не находит логи:
```bash
# Проверка сервиса Ollama
sudo systemctl status ollama.service
sudo journalctl -u ollama.service --since "1 hour ago"
```

### InfluxDB недоступен:
```bash
# Проверка сервиса
curl http://192.168.237.198:8086/health
```

### Нет данных в Grafana:
1. Проверьте временной диапазон (увеличьте до 24 часов)
2. Убедитесь что интеграция работает: `systemctl status ollama-influxdb-integration`
3. Проверьте логи интеграции: `journalctl -u ollama-influxdb-integration -f`

## 📋 Требования

- **Python 3.8+**
- **Права sudo** (для доступа к journalctl)
- **Зависимости**: `requests`, `python-dateutil`
- **InfluxDB 2.x** на сервере 192.168.237.198
- **Grafana** для визуализации

## 🎯 Workflow использования

### 🔍 Для анализа и отработки алгоритмов:
1. **Запустить парсер** → получить детальный отчет
2. **Изучить отчет** → проанализировать данные и паттерны
3. **Отработать алгоритмы** → использовать данные для разработки

### 🔗 Для мониторинга в реальном времени:
1. **Настроить InfluxDB Integration** → подключить к базе данных
2. **Запустить интеграцию** → начать сбор метрик
3. **Настроить Grafana** → создать дашборды для визуализации

## 📝 Логирование

```bash
# Логи парсера (автономный запуск)
sudo python3 parser.py 2>&1 | tee parser.log

# Логи интеграции (мониторинг)
journalctl -u ollama-influxdb-integration -f

# Логи InfluxDB
sudo journalctl -u influxdb -f
```

---

**Parser**: Независимый инструмент для максимально детального анализа логов Ollama и отработки алгоритмов обработки данных.

**InfluxDB Integration**: Система мониторинга в реальном времени для визуализации в Grafana.