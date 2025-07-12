#!/usr/bin/env python3
"""
Ollama Log Exporter for Prometheus
Парсит логи Ollama и экспонирует реальные метрики использования моделей, пользователей и задержек работы Ollama.
"""
import re
import subprocess
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import argparse

# --- Глобальные переменные и метрики ---
# Эта переменная будет хранить ID последней загруженной модели между вызовами парсера
last_seen_model = 'unknown'

# Метрики объявляются здесь, чтобы быть доступными для всех функций
OLLAMA_INFO = Gauge(
    'ollama_info',
    'Information about the Ollama instance',
    ['version', 'num_parallel', 'max_loaded_models']
)
REQUESTS_TOTAL = Counter(
    'ollama_requests_total',
    'Total requests to Ollama',
    ['ip', 'endpoint', 'method', 'status', 'model']
)
REQUEST_ERRORS_TOTAL = Counter(
    'ollama_request_errors_total',
    'Total error requests to Ollama',
    ['ip', 'endpoint', 'method', 'status', 'model']
)
REQUEST_DURATION = Histogram(
    'ollama_request_duration_seconds',
    'Request duration to Ollama',
    ['ip', 'endpoint', 'method', 'model']
)

# --- Функции парсинга ---

def parse_log_line(line):
    """Анализирует одну строку лога и возвращает словарь с данными или None."""
    global last_seen_model

    # Сначала ищем GIN-логи, т.к. они имеют уникальный формат
    if '[GIN]' in line:
        # Извлекаем только саму GIN-строку
        gin_part_match = re.search(r'(\[GIN\].*)', line)
        if not gin_part_match:
            return None
        
        gin_part = gin_part_match.group(1)
        parts = [p.strip() for p in gin_part.split('|')]
        if len(parts) < 5:
            return None
        
        status = parts[1]
        duration_str = parts[2]
        ip = parts[3]
        method_endpoint = parts[4].split()
        method = method_endpoint[0]
        endpoint = method_endpoint[1].strip('"')
        
        return {
            'status': status,
            'duration': duration_str,
            'ip': ip,
            'method': method,
            'endpoint': endpoint,
            'model': last_seen_model
        }

    # Затем парсим логи конфигурации из строки 'inference compute'
    if 'msg="inference compute"' in line:
        num_parallel_match = re.search(r'parallel=(\d+)', line)
        # max_loaded_models в этой строке нет, оставляем unknown
        
        num_parallel = num_parallel_match.group(1) if num_parallel_match else 'unknown'
        
        OLLAMA_INFO.labels(
            version='unknown', # версия также отсутствует
            num_parallel=num_parallel,
            max_loaded_models='unknown'
        ).set(1)
        return None

    # И в конце парсим ID модели
    if 'msg="starting llama server"' in line:
        match = re.search(r'sha256-([a-f0-9]{64})', line)
        if match:
            last_seen_model = f"sha256:{match.group(1)[:12]}"
        return None
        
    return None

def duration_to_seconds(duration):
    """Преобразует строку вроде '1m22s', '59.6s' или '74.5µs' в float секунд."""
    total_seconds = 0.0
    try:
        if 'm' in duration:
            parts = duration.split('m')
            total_seconds += float(parts[0]) * 60
            duration = parts[1]

        if 's' in duration:
            total_seconds += float(duration.replace('s', ''))
        elif 'ms' in duration:
            total_seconds += float(duration.replace('ms', '')) / 1000
        elif 'µs' in duration or 'us' in duration:
            total_seconds += float(duration.replace('µs', '').replace('us', '')) / 1_000_000
    except (ValueError, IndexError):
        return 0.0 # Возвращаем 0 если не смогли распарсить
    return total_seconds

def journalctl_follow(unit):
    """Читает и очищает логи из journalctl."""
    process = subprocess.Popen(['journalctl', '-u', unit, '-f', '-n', '0', '--no-pager'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in iter(process.stdout.readline, ''):
        # Убираем метаданные journalctl
        clean_line = re.sub(r'^.*?ollama\[\d+\]: ', '', line.strip())
        yield clean_line

# --- Основная логика ---

def main():
    parser = argparse.ArgumentParser(description='Ollama Log Exporter for Prometheus.')
    parser.add_argument('--port', type=int, default=9877, help='Port to listen on')
    parser.add_argument('--unit', type=str, default='ollama', help='systemd unit name')
    args = parser.parse_args()

    start_http_server(args.port)
    print(f"Exporter started on :{args.port}, following journalctl -u {args.unit}")

    for line in journalctl_follow(args.unit):
        data = parse_log_line(line)
        if not data:
            continue

        duration_sec = duration_to_seconds(data['duration'])
        
        # Обновляем метрики
        labels = {
            'ip': data['ip'],
            'endpoint': data['endpoint'],
            'method': data['method'],
            'status': data['status'],
            'model': data['model']
        }
        duration_labels = {
            'ip': data['ip'],
            'endpoint': data['endpoint'],
            'method': data['method'],
            'model': data['model']
        }

        REQUESTS_TOTAL.labels(**labels).inc()
        REQUEST_DURATION.labels(**duration_labels).observe(duration_sec)

        if data['status'].startswith('4') or data['status'].startswith('5'):
            REQUEST_ERRORS_TOTAL.labels(**labels).inc()

if __name__ == '__main__':
    main()
