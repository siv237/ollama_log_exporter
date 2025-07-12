#!/usr/bin/env python3
"""
Ollama Log Exporter for Prometheus
Парсит логи Ollama и экспонирует реальные метрики использования моделей, пользователей и задержек работы Ollama.
"""
import re
import subprocess
from prometheus_client import start_http_server, Counter, Histogram
import argparse

def parse_log_line(line):
    # Пример: [GIN] 2025/07/12 - 09:27:25 | 500 | 59.654288737s | 127.0.0.1 | POST     "/api/generate"
    pattern = re.compile(
        r'^\[GIN\]\s+(\d{4}/\d{2}/\d{2})\s+-\s+(\d{2}:\d{2}:\d{2})\s+\|\s+(\d{3})\s+\|\s+([\dm\.sµ]+)\s+\|\s+([\d\.]+)\s+\|\s+(\w+)\s+"([^"\s]+)"'
    )
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
    # Преобразует строку вроде '1m22s', '59.6s' или '74.5µs' в float секунд
    total_seconds = 0.0
    if 'm' in duration:
        parts = duration.split('m')
        total_seconds += float(parts[0]) * 60
        duration = parts[1]

    if duration.endswith('s'):
        total_seconds += float(duration[:-1])
    elif duration.endswith('ms'):
        total_seconds += float(duration[:-2]) / 1000
    elif duration.endswith('µs') or duration.endswith('us'):
        total_seconds += float(duration[:-2]) / 1_000_000
    
    return total_seconds

def journalctl_follow(unit):
    # Читает логи из journalctl -u ollama -f
    p = subprocess.Popen([
        "journalctl", "-u", unit, "-f", "-n", "0", "--output", "cat"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        yield line.rstrip('\n')

def main():
    parser = argparse.ArgumentParser(description='Ollama Log Exporter for Prometheus')
    parser.add_argument('--port', type=int, default=9877, help='Port to expose metrics')
    parser.add_argument('--unit', type=str, default='ollama', help='systemd unit name (default: ollama)')
    args = parser.parse_args()

    # Метрики
    REQ_COUNTER = Counter('ollama_requests_total', 'Total requests to Ollama', ['endpoint', 'method', 'status', 'ip'])
    ERROR_COUNTER = Counter('ollama_request_errors_total', 'Total error requests to Ollama', ['endpoint', 'method', 'status', 'ip'])
    DURATION_HIST = Histogram('ollama_request_duration_seconds', 'Request duration to Ollama', ['endpoint', 'method', 'ip'])

    start_http_server(args.port)
    print(f"Exporter started on :{args.port}, following journalctl -u {args.unit}")

    for line in journalctl_follow(args.unit):
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
