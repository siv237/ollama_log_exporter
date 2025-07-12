#!/usr/bin/env python3
"""
Ollama Log Exporter for Prometheus
Парсит логи Ollama и экспонирует реальные метрики использования моделей, пользователей и задержек работы Ollama.
"""
import re
import time
from collections import defaultdict
from prometheus_client import start_http_server, Counter, Histogram
import argparse
import os

def parse_log_line(line):
    # Пример: [GIN] 2025/07/12 - 09:27:25 | 500 | 59.654288737s | 127.0.0.1 | POST     "/api/generate"
    pattern = re.compile(r"\\[GIN\\] (\\d{4}/\\d{2}/\\d{2}) - (\\d{2}:\\d{2}:\\d{2}) \\| (\\d{3}) \\| ([\\d.]+[a-z]+) \\| ([\\d.]+\\.[\\d.]+\\.[\\d.]+|[\\d.]+) \\| (\\w+) +\\"([^"]+)\\"")
    m = pattern.match(line)
    if not m:
        return None
    date, t, status, duration, ip, method, endpoint = m.groups()
    return {
        'datetime': f"{date} {t}",
        'status': status,
        'duration': duration,
        'ip': ip,
        'method': method,
        'endpoint': endpoint
    }

def duration_to_seconds(duration):
    # Преобразует строку вроде '59.654288737s' или '74.582µs' в float секунд
    if duration.endswith('s'):
        return float(duration[:-1])
    elif duration.endswith('ms'):
        return float(duration[:-2]) / 1000
    elif duration.endswith('µs') or duration.endswith('us'):
        return float(duration[:-2]) / 1_000_000
    else:
        return 0.0

def tail_f(filename):
    with open(filename, 'r') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            yield line.rstrip('\n')

def main():
    parser = argparse.ArgumentParser(description='Ollama Log Exporter for Prometheus')
    parser.add_argument('--log', type=str, required=True, help='Path to ollama log file')
    parser.add_argument('--port', type=int, default=9877, help='Port to expose metrics')
    args = parser.parse_args()

    # Метрики
    REQ_COUNTER = Counter('ollama_requests_total', 'Total requests to Ollama', ['endpoint', 'method', 'status', 'ip'])
    ERROR_COUNTER = Counter('ollama_request_errors_total', 'Total error requests to Ollama', ['endpoint', 'method', 'status', 'ip'])
    DURATION_HIST = Histogram('ollama_request_duration_seconds', 'Request duration to Ollama', ['endpoint', 'method', 'ip'])

    start_http_server(args.port)
    print(f"Exporter started on :{args.port}, tailing {args.log}")

    for line in tail_f(args.log):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        endpoint = parsed['endpoint']
        method = parsed['method']
        status = parsed['status']
        ip = parsed['ip']
        duration = duration_to_seconds(parsed['duration'])
        REQ_COUNTER.labels(endpoint=endpoint, method=method, status=status, ip=ip).inc()
        if status.startswith('5') or status.startswith('4'):
            ERROR_COUNTER.labels(endpoint=endpoint, method=method, status=status, ip=ip).inc()
        DURATION_HIST.labels(endpoint=endpoint, method=method, ip=ip).observe(duration)

if __name__ == '__main__':
    main()
