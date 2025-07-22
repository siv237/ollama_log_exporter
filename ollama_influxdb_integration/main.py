#!/usr/bin/env python3
"""
Основной скрипт интеграции Ollama с InfluxDB.
Использует существующий парсер логов и записывает данные в InfluxDB.
"""

import os
import sys
import json
import time
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта парсера
sys.path.append(str(Path(__file__).parent.parent))

# Убираем зависимость от parser.py - делаем свой парсинг
import re
import subprocess
import json
from dateutil import parser as dtparser

from influxdb_writer import OllamaInfluxDBWriter


class OllamaInfluxDBIntegration:
    """Класс для интеграции Ollama с InfluxDB."""
    
    def __init__(self, config_file: str = 'config.json'):
        """Инициализация интеграции."""
        self.config = self.load_config(config_file)
        
        # Путь к манифестам Ollama
        self.manifests_root = Path(self.config.get('ollama', {}).get('manifests_path', '/root/.ollama/models/manifests'))
        
        self.writer = OllamaInfluxDBWriter(
            influxdb_url=self.config['influxdb']['url'],
            token=self.config['influxdb']['token'],
            org=self.config['influxdb']['org'],
            bucket=self.config['influxdb']['bucket'],
            manifests_path=str(self.manifests_root)
        )
        
        # Пути к файлам
        self.reports_dir = Path('reports')
        self.reports_dir.mkdir(exist_ok=True)
        
        self.dump_file = self.reports_dir / 'ollama_log_dump.txt'
        
        # Отслеживание последнего времени парсинга
        self.last_parse_time = None
    
    def load_config(self, config_file: str) -> dict:
        """Загружает конфигурацию из файла."""
        config_path = Path(__file__).parent / config_file
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Создаем конфигурацию по умолчанию
            default_config = {
                "influxdb": {
                    "url": "http://192.168.237.198:8086",
                    "token": "your-influxdb-token-here",
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
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            print(f"Создан файл конфигурации: {config_path}")
            print("Пожалуйста, отредактируйте конфигурацию и запустите скрипт снова.")
            sys.exit(1)
    
    def test_connections(self) -> bool:
        """Проверяет подключения к внешним системам."""
        print("Проверка подключения к InfluxDB...")
        
        if not self.writer.test_connection():
            print("❌ Не удалось подключиться к InfluxDB")
            return False
        
        print("✅ Подключение к InfluxDB успешно")
        
        # Проверяем доступность манифестов Ollama
        if not self.manifests_root.exists():
            print(f"⚠️  Путь к манифестам Ollama не найден: {self.manifests_root}")
            print("Будет использоваться только информация из логов")
        else:
            print(f"✅ Манифесты Ollama найдены: {self.manifests_root}")
        
        return True
    
    def dump_logs(self, since_time=None):
        """Сохраняет логи ollama.service."""
        try:
            if since_time:
                # Инкрементальная выгрузка - только новые логи
                cmd = f"journalctl -u ollama.service --since '{since_time}' --no-pager -o short-iso"
                mode = 'a'  # Добавляем к существующему файлу
                print(f"    Выгрузка новых логов с {since_time}")
            else:
                # Полная выгрузка - все логи за 24 часа
                cmd = "journalctl -u ollama.service --since '24 hours ago' --no-pager -o short-iso"
                mode = 'w'  # Перезаписываем файл
                print(f"    Полная выгрузка логов за 24 часа")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            
            with open(self.dump_file, mode, encoding='utf-8') as f:
                f.write(result.stdout)
            
            lines_count = len(result.stdout.splitlines())
            print(f"    Сохранено {lines_count} строк логов")
            
            # Обновляем время последнего парсинга
            if lines_count > 0:
                self.last_parse_time = time.strftime('%Y-%m-%d %H:%M:%S')
                
        except Exception as e:
            print(f"    ❌ Ошибка выгрузки логов: {e}")
    
    def parse_ollama_sessions(self) -> list:
        """Парсит сессии моделей из логов."""
        sessions = []
        
        with open(self.dump_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_session = None
        
        for line in lines:
            # Ищем начало новой сессии
            if 'msg="starting llama server"' in line:
                # Сохраняем предыдущую сессию
                if current_session:
                    sessions.append(current_session)
                
                # Создаем новую сессию
                current_session = {
                    'start_time': line.split()[0],
                    'raw_lines': [line]
                }
                
                # Извлекаем PID
                pid_match = re.search(r'ollama\[(\d+)\]', line)
                if pid_match:
                    current_session['pid'] = pid_match.group(1)
                
                # Извлекаем параметры из команды
                if 'cmd=' in line:
                    cmd_str = line.split('cmd=')[1].strip().strip('"')
                    self._parse_command_params(current_session, cmd_str)
            
            elif current_session:
                current_session['raw_lines'].append(line)
                
                # Ищем время загрузки
                if 'msg="llama runner started' in line and ' in ' in line and ' seconds' in line:
                    try:
                        time_str = line.split(' in ')[-1].split(' seconds')[0]
                        current_session['runner_start_time'] = f"{time_str} s"
                    except:
                        pass
                
                # Ищем информацию о модели
                if 'general.name' in line:
                    match = re.search(r'general\.name\s+str\s*=\s*(.+)', line)
                    if match:
                        current_session['model_name'] = match.group(1).strip()
                
                # Ищем offload информацию
                if 'msg=offload' in line:
                    offload_str = line.split('msg=offload ')[1]
                    current_session['offload_info'] = self._parse_key_value_string(offload_str)
        
        # Сохраняем последнюю сессию
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _parse_command_params(self, session: dict, cmd_str: str):
        """Извлекает параметры из команды запуска."""
        cmd_parts = cmd_str.split()
        params = {}
        
        for i, part in enumerate(cmd_parts):
            if part.startswith('--') and i + 1 < len(cmd_parts):
                params[part] = cmd_parts[i + 1]
        
        # Извлекаем SHA256 из пути модели
        model_path = params.get('--model', '')
        if 'sha256-' in model_path:
            sha = model_path.split('sha256-')[-1]
            session['sha256'] = sha
        
        # Сохраняем параметры
        session['ctx_size'] = params.get('--ctx-size', 'N/A')
        session['batch_size'] = params.get('--batch-size', 'N/A')
        session['gpu_layers'] = params.get('--n-gpu-layers', 'N/A')
        session['threads'] = params.get('--threads', 'N/A')
        session['parallel'] = params.get('--parallel', 'N/A')
    
    def _parse_key_value_string(self, s: str) -> dict:
        """Парсит строку с парами ключ=значение."""
        pattern = r'(\w+(?:\.\w+)*)=("\[.*?\]"|\[.*?\]|"[^"]*"|[\w\./\:-]+)'
        matches = re.findall(pattern, s)
        return {key: value.strip('"') for key, value in matches}
    
    def parse_gin_requests(self) -> list:
        """Парсит GIN-запросы из логов."""
        requests = []
        gin_pattern = re.compile(r'^([\d\-:T\+]+)\s+[^:]+: \[GIN\]\s+(\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2}) \| (\d+) \| ([^|]+) \|\s*([^|]+) \|\s*(\w+)\s+"([^"]+)"')
        
        with open(self.dump_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = gin_pattern.search(line)
                if match:
                    requests.append({
                        'journal_time': match.group(1).strip(),
                        'status': match.group(3).strip(),
                        'latency': match.group(4).strip(),
                        'ip': match.group(5).strip(),
                        'method': match.group(6).strip(),
                        'path': match.group(7).strip()
                    })
        
        return requests
    
    def parse_unloading_events(self) -> list:
        """Парсит события выгрузки моделей из логов."""
        unloading_events = []
        
        # Паттерн для событий "gpu VRAM usage didn't recover within timeout"
        vram_pattern = re.compile(
            r'^([\d\-:T\+]+)\s+[^:]+:\s+.*msg="gpu VRAM usage didn\'t recover within timeout"\s+'
            r'seconds=([\d\.]+)\s+model=([^\s]+)'
        )
        
        with open(self.dump_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = vram_pattern.search(line)
                if match:
                    journal_time = match.group(1).strip()
                    timeout_seconds = float(match.group(2))
                    model_path = match.group(3).strip()
                    
                    # Извлекаем SHA256 из пути модели
                    sha256 = None
                    if 'sha256-' in model_path:
                        sha256 = model_path.split('sha256-')[-1]
                    
                    unloading_events.append({
                        'journal_time': journal_time,
                        'timeout_seconds': timeout_seconds,
                        'model_path': model_path,
                        'sha256': sha256,
                        'event_type': 'vram_timeout'
                    })
        
        return unloading_events
    
    def assign_requests_to_sessions(self, sessions: list, requests: list) -> list:
        """Привязывает API запросы к сессиям по времени."""
        # Преобразуем время сессий в datetime
        for session in sessions:
            if session.get('start_time'):
                try:
                    session['_dt'] = dtparser.parse(session['start_time'])
                except:
                    session['_dt'] = None
        
        # Сортируем сессии по времени
        sessions_sorted = sorted(
            [(i, s) for i, s in enumerate(sessions) if s.get('_dt')], 
            key=lambda x: x[1]['_dt']
        )
        
        # Привязываем запросы к сессиям
        for idx, (i, session) in enumerate(sessions_sorted):
            start = session['_dt']
            end = sessions_sorted[idx+1][1]['_dt'] if idx+1 < len(sessions_sorted) else None
            session['gin_requests'] = []
            
            for req in requests:
                try:
                    req_dt = dtparser.parse(req['journal_time'])
                    if start and (not end or req_dt < end) and req_dt >= start:
                        session['gin_requests'].append(req)
                except:
                    continue
        
        # Убираем временные поля
        for session in sessions:
            if '_dt' in session:
                del session['_dt']
        
        return sessions
    
    def collect_log_data(self, incremental=False) -> tuple:
        """Собирает и парсит данные из логов Ollama."""
        print("Сбор данных из логов Ollama...")
        
        # 1. Сохраняем логи
        if incremental and self.last_parse_time:
            print("  - Выгрузка новых логов...")
            self.dump_logs(since_time=self.last_parse_time)
        else:
            print("  - Полная выгрузка логов...")
            self.dump_logs()
        
        # 2. Парсим сессии
        print("  - Парсинг сессий моделей...")
        sessions = self.parse_ollama_sessions()
        
        # 3. Парсим API запросы
        print("  - Парсинг API запросов...")
        gin_requests = self.parse_gin_requests()
        
        # 4. Привязываем запросы к сессиям
        print("  - Привязка запросов к сессиям...")
        sessions = self.assign_requests_to_sessions(sessions, gin_requests)
        
        # 5. Парсим события выгрузки моделей
        print("  - Парсинг событий выгрузки...")
        unloading_events = self.parse_unloading_events()
        
        print(f"  ✅ Собрано: {len(sessions)} сессий, {len(gin_requests)} запросов, {len(unloading_events)} выгрузок")
        
        return sessions, [], unloading_events, []
    
    def write_to_influxdb(self, sessions: list, systemd_events: list, ollama_events: list, unloading_events: list = None) -> bool:
        """Записывает данные в InfluxDB."""
        print("Запись данных в InfluxDB...")
        
        success = True
        
        # Записываем сессии
        if sessions:
            print(f"  - Запись {len(sessions)} сессий...")
            if not self.writer.write_sessions(sessions):
                success = False
        
        # Записываем системные события
        if systemd_events or ollama_events:
            print(f"  - Запись {len(systemd_events + ollama_events)} событий...")
            if not self.writer.write_system_events(systemd_events, ollama_events):
                success = False
        
        # Записываем события выгрузки
        if unloading_events:
            print(f"  - Запись {len(unloading_events)} событий выгрузки...")
            if not self.writer.write_unloading_events(unloading_events, sessions):
                success = False
        
        return success
    
    def run_once(self) -> bool:
        """Выполняет один цикл сбора и записи данных."""
        try:
            # Собираем данные
            sessions, systemd_events, unloading_events, models = self.collect_log_data()
        
            # Записываем в InfluxDB
            success = self.write_to_influxdb(sessions, systemd_events, [], unloading_events)
            
            if success:
                print("✅ Цикл интеграции завершен успешно")
            else:
                print("❌ Ошибки при записи в InfluxDB")
            
            return success
            
        except Exception as e:
            print(f"❌ Ошибка в цикле интеграции: {e}")
            return False
    
    def run_continuous(self):
        """Запускает непрерывный мониторинг."""
        interval = 10  # Фиксированный интервал 10 секунд
        
        print(f"Запуск непрерывного мониторинга (интервал: {interval} секунд)")
        print("Для остановки нажмите Ctrl+C")
        
        first_run = True
        
        try:
            while True:
                print(f"\n--- Цикл интеграции {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
                
                if first_run:
                    print("🚀 Первый запуск - полный парсинг всех логов")
                    sessions, systemd_events, unloading_events, models = self.collect_log_data(incremental=False)
                    first_run = False
                else:
                    print("⚡ Инкрементальный парсинг новых логов")
                    sessions, systemd_events, unloading_events, models = self.collect_log_data(incremental=True)
                
                # Записываем в InfluxDB только если есть данные
                if sessions or systemd_events or unloading_events:
                    success = self.write_to_influxdb(sessions, systemd_events, [], unloading_events)
                    if success:
                        print("✅ Цикл интеграции завершен успешно")
                    else:
                        print("❌ Ошибки при записи в InfluxDB")
                else:
                    print("ℹ️  Новых данных нет")
                
                print(f"Ожидание {interval} секунд до следующего цикла...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Остановка мониторинга по запросу пользователя")
        except Exception as e:
            print(f"\n❌ Критическая ошибка: {e}")


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Интеграция Ollama с InfluxDB')
    parser.add_argument('--config', '-c', default='config.json', 
                       help='Путь к файлу конфигурации')
    parser.add_argument('--once', action='store_true',
                       help='Выполнить один цикл и завершить')
    parser.add_argument('--test', action='store_true',
                       help='Только проверить подключения')
    
    args = parser.parse_args()
    
    # Создаем интеграцию
    integration = OllamaInfluxDBIntegration(args.config)
    
    # Проверяем подключения
    if not integration.test_connections():
        sys.exit(1)
    
    if args.test:
        print("✅ Проверка подключений завершена успешно")
        return
    
    if args.once:
        # Один цикл
        success = integration.run_once()
        sys.exit(0 if success else 1)
    else:
        # Непрерывный мониторинг
        integration.run_continuous()


if __name__ == '__main__':
    main()