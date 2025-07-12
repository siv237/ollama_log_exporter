# Ollama Log Exporter

Пассивный экспортер для Prometheus, который парсит логи Ollama и экспонирует реальные метрики использования моделей, пользователей и задержек работы Ollama.

## Возможности
- Подсчет количества реальных запросов к Ollama (по пользователям, моделям, статусам)
- Измерение времени обработки (latency)
- Подсчет ошибок
- Экспорт в формате Prometheus

## Быстрый старт (установка на сервере Ollama)

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/siv237/ollama_log_exporter.git
   cd ollama_log_exporter
   ```
2. **Создайте и активируйте виртуальное окружение:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   # или вручную:
   pip install prometheus_client
   ```
4. **(Для ручного теста) Запустите экспортер:**
   ```bash
   python exporter.py --port 9877
   ```
   Метрики будут доступны на http://localhost:9877/metrics

## Автоматический запуск через systemd

1. **Скопируйте unit-файл службы:**
   ```bash
   cp /opt/ollama_log_exporter/ollama_log_exporter.service /etc/systemd/system/
   ```
2. **Перезагрузите systemd и включите службу:**
   ```bash
   systemctl daemon-reload
   systemctl enable --now ollama_log_exporter.service
   ```
3. **Проверьте статус службы:**
   ```bash
   systemctl status ollama_log_exporter.service
   ```

## Обновление кода экспортера

1. Локально внесите изменения, закоммитьте и запушьте на GitHub.
2. На сервере Ollama выполните:
   ```bash
   cd /opt/ollama_log_exporter
   git pull
   systemctl restart ollama_log_exporter.service
   ```
3. Проверьте метрики:
   ```bash
   curl http://localhost:9877/metrics
   # или с другой машины:
   curl http://ollama.localnet:9877/metrics
   ```

---

**ВАЖНО:** Экспортер не создает искусственную нагрузку на Ollama, а только анализирует реальные логи через systemd-journal.

## TODO
- [ ] Прототип парсера логов Ollama
- [ ] Экспортер метрик на HTTP-порт
- [ ] Конфигурирование путей к логам и портов
- [ ] Сборка Dockerfile (по желанию)

---

## Grafana Dashboard

В репозитории есть готовый дашборд Grafana: `grafana_ollama_dashboard.json`.

### Как импортировать:
1. Откройте Grafana (например, http://192.168.237.198:3000)
2. В меню выберите “Dashboards” → “Import”
3. Загрузите файл `grafana_ollama_dashboard.json` или скопируйте его содержимое в поле импорта
4. Выберите ваш Prometheus datasource и подтвердите импорт

### Что показывает дашборд:
- **График количества запросов** к Ollama (по endpoint, методу, статусу)
- **График ошибок** (4xx, 5xx)
- **График времени обработки** (latency, 95-й перцентиль)
- **Таблица активности** по IP, endpoint и методу

---

**ВАЖНО:** Экспортер не создает искусственную нагрузку на Ollama, а только анализирует реальные логи.

## TODO
