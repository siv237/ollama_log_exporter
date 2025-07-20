#!/usr/bin/env python3
"""
Генерация отчета о работе Ollama прямо из данных InfluxDB.
Создает идентичный отчет парсера, но используя только данные из базы.
"""

import json
import sys
import requests
from pathlib import Path
from datetime import datetime, timezone
import csv
from io import StringIO


class InfluxDBReportGenerator:
    """Класс для генерации отчета из данных InfluxDB."""
    
    def __init__(self, config_file: str = 'config.json'):
        """Инициализация генератора."""
        self.config = self.load_config(config_file)
        self.influxdb_url = self.config['influxdb']['url']
        self.token = self.config['influxdb']['token']
        self.org = self.config['influxdb']['org']
        self.bucket = self.config['influxdb']['bucket']
        
        # Headers для API запросов
        self.headers = {
            'Authorization': f'Token {self.token}',
            'Content-Type': 'application/vnd.flux',
            'Accept': 'application/csv'
        }
        
        self.query_url = f"{self.influxdb_url}/api/v2/query"
        
        # Данные для отчета
        self.models_data = []
        self.sessions_data = []
        self.requests_data = []
        self.system_events = []
    
    def load_config(self, config_file: str) -> dict:
        """Загружает конфигурацию."""
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"❌ Файл конфигурации не найден: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def execute_flux_query(self, query: str) -> list:
        """Выполняет Flux запрос и возвращает данные как список словарей."""
        try:
            params = {'org': self.org}
            response = requests.post(
                self.query_url,
                headers=self.headers,
                params=params,
                data=query,
                timeout=30
            )
            
            if response.status_code == 200:
                # Парсим CSV ответ
                csv_data = response.text
                if not csv_data.strip():
                    return []
                
                reader = csv.DictReader(StringIO(csv_data))
                return list(reader)
            else:
                print(f"❌ Ошибка запроса: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"❌ Ошибка выполнения запроса: {e}")
            return []
    
    def fetch_models_data(self):
        """Получает данные о моделях."""
        print("🔍 Получение данных о моделях...")
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "ollama_sessions")
          |> group(columns: ["model", "model_sha256"])
          |> distinct(column: "model")
          |> keep(columns: ["model", "model_sha256"])
        '''
        
        self.models_data = self.execute_flux_query(query)
        print(f"  📊 Найдено моделей: {len(self.models_data)}")
    
    def fetch_sessions_data(self):
        """Получает данные о сессиях."""
        print("🔍 Получение данных о сессиях...")
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "ollama_sessions")
          |> group(columns: ["session_id", "_time", "model", "model_sha256", "pid"])
          |> pivot(rowKey:["session_id", "_time", "model", "model_sha256", "pid"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        
        self.sessions_data = self.execute_flux_query(query)
        print(f"  📊 Найдено сессий: {len(self.sessions_data)}")
    
    def fetch_requests_data(self):
        """Получает данные о запросах."""
        print("🔍 Получение данных о запросах...")
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "ollama_requests")
          |> filter(fn: (r) => r._field == "latency_seconds")
          |> sort(columns: ["_time"])
        '''
        
        self.requests_data = self.execute_flux_query(query)
        print(f"  📊 Найдено запросов: {len(self.requests_data)}")
    
    def fetch_system_events(self):
        """Получает системные события."""
        print("🔍 Получение системных событий...")
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "ollama_system_events")
          |> sort(columns: ["_time"])
        '''
        
        self.system_events = self.execute_flux_query(query)
        print(f"  📊 Найдено событий: {len(self.system_events)}")
    
    def format_memory_size(self, bytes_value: str) -> str:
        """Форматирует размер памяти."""
        try:
            bytes_val = float(bytes_value)
            if bytes_val >= 1024**4:
                return f"{bytes_val / 1024**4:.1f} TB"
            elif bytes_val >= 1024**3:
                return f"{bytes_val / 1024**3:.1f} GB"
            elif bytes_val >= 1024**2:
                return f"{bytes_val / 1024**2:.1f} MB"
            elif bytes_val >= 1024:
                return f"{bytes_val / 1024:.1f} KB"
            else:
                return f"{int(bytes_val)} B"
        except:
            return str(bytes_value)
    
    def format_duration(self, seconds_value: str) -> str:
        """Форматирует длительность (как в эталонном отчете)."""
        try:
            seconds = float(seconds_value)
            if seconds >= 60:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes}m{secs:.0f}s"
            elif seconds >= 1:
                return f"{seconds:.9f}s"
            elif seconds >= 0.001:
                ms = seconds * 1000
                return f"{ms:.6f}ms"
            else:
                us = seconds * 1000000
                return f"{us:.3f}µs"
        except:
            return str(seconds_value)
    
    def format_timestamp(self, timestamp_str: str) -> str:
        """Форматирует временную метку."""
        try:
            from dateutil import tz
            # Парсим ISO timestamp и конвертируем в локальное время
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            # Конвертируем в локальную зону (как в эталонном отчете)
            local_dt = dt.astimezone(tz.tzlocal())
            # Форматируем с двоеточием в часовом поясе для соответствия эталону
            formatted = local_dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            # Добавляем двоеточие в часовой пояс: +1000 -> +10:00
            if len(formatted) >= 5 and formatted[-5] in '+-':
                formatted = formatted[:-2] + ':' + formatted[-2:]
            return formatted
        except:
            return timestamp_str
    
    def group_requests_by_session(self) -> dict:
        """Группирует запросы по сессиям по времени (как в парсере)."""
        from dateutil import parser as dtparser
        
        requests_by_session = {}
        
        # Преобразуем время сессий в datetime и сортируем
        sessions_with_time = []
        for i, session in enumerate(self.sessions_data):
            try:
                session_time = dtparser.parse(session.get('_time', ''))
                sessions_with_time.append((i, session, session_time))
            except Exception:
                continue
        
        # Сортируем сессии по времени
        sessions_with_time.sort(key=lambda x: x[2])
        
        # Преобразуем запросы в список с временными метками
        requests_with_time = []
        for request in self.requests_data:
            try:
                request_time = dtparser.parse(request.get('_time', ''))
                requests_with_time.append((request, request_time))
            except Exception:
                continue
        
        # Для каждой сессии определяем временной диапазон и находим соответствующие запросы
        for idx, (session_idx, session, session_start) in enumerate(sessions_with_time):
            session_key = f"session_{session_idx + 1}"
            requests_by_session[session_key] = []
            
            # Определяем конец сессии (начало следующей сессии)
            session_end = None
            if idx + 1 < len(sessions_with_time):
                session_end = sessions_with_time[idx + 1][2]
            
            # Находим запросы для этой сессии (как в парсере)
            seen_requests = set()  # Для дедупликации
            for request, request_time in requests_with_time:
                # Проверяем, что запрос попадает в диапазон сессии
                if session_start and (not session_end or request_time < session_end):
                    if request_time >= session_start:
                        # Фильтруем запросы - только те, у которых есть status и method
                        if request.get('status') is not None and request.get('method'):
                            # Создаем уникальный ключ для дедупликации
                            request_key = (
                                request.get('_time'),
                                request.get('status'),
                                request.get('method'),
                                request.get('endpoint'),
                                request.get('client_ip'),
                                request.get('_value')
                            )
                            if request_key not in seen_requests:
                                seen_requests.add(request_key)
                                requests_by_session[session_key].append(request)
        
        return requests_by_session
    
    def generate_models_table(self) -> str:
        """Генерирует таблицу моделей."""
        if not self.models_data:
            return "Данные о моделях не найдены в InfluxDB.\n"
        
        content = "## Список моделей Ollama (из InfluxDB)\n\n"
        content += "| Модель | SHA256 | Источник |\n"
        content += "|--------|--------|----------|\n"
        
        for model in self.models_data:
            model_name = model.get('model', 'unknown')
            sha256 = model.get('model_sha256', 'N/A')
            if sha256 and len(sha256) > 16:
                sha256 = sha256[:16] + "..."
            
            content += f"| {model_name} | {sha256} | InfluxDB |\n"
        
        content += "\n"
        return content
    
    def generate_sessions_report(self) -> str:
        """Генерирует отчет по сессиям."""
        if not self.sessions_data:
            return "Данные о сессиях не найдены в InfluxDB.\n"
        
        content = "## Сессии Ollama (из InfluxDB)\n\n"
        
        # Группируем запросы по сессиям
        requests_by_session = self.group_requests_by_session()
        
        # Сортируем сессии по времени для правильного порядка в отчете
        from dateutil import parser as dtparser
        sessions_with_time = []
        for i, session in enumerate(self.sessions_data):
            try:
                session_time = dtparser.parse(session.get('_time', ''))
                sessions_with_time.append((i, session, session_time))
            except Exception:
                continue
        
        sessions_with_time.sort(key=lambda x: x[2])
        
        for session_num, (session_idx, session, session_time) in enumerate(sessions_with_time, 1):
            session_key = f"session_{session_idx + 1}"
            
            content += f"## Сессия {session_num}: {session.get('model', 'unknown')}\n\n"
            
            # Основная информация
            content += f"*   **Время старта сессии:** {self.format_timestamp(session.get('_time', ''))}\n"
            content += f"*   **Session ID:** {session_key}\n"
            content += f"*   **PID процесса Ollama:** {session.get('pid', 'N/A')}\n"
            content += f"*   **Модель:** {session.get('model', 'N/A')}\n"
            content += f"*   **SHA256:** {session.get('model_sha256', 'N/A')}\n"
            content += f"*   **GPU:** {session.get('gpu_library', 'N/A')}\n"
            
            # Параметры модели
            if session.get('ctx_size'):
                content += f"*   **CTX size:** {session.get('ctx_size')}\n"
            if session.get('batch_size'):
                content += f"*   **Batch size:** {session.get('batch_size')}\n"
            if session.get('gpu_layers'):
                content += f"*   **GPU layers:** {session.get('gpu_layers')}\n"
            if session.get('threads'):
                content += f"*   **Threads:** {session.get('threads')}\n"
            if session.get('parallel'):
                content += f"*   **Parallel:** {session.get('parallel')}\n"
            
            # Память
            content += "\n### Использование памяти:\n\n"
            if session.get('vram_available_bytes'):
                vram_available = self.format_memory_size(session['vram_available_bytes'])
                content += f"*   **VRAM доступно:** {vram_available}\n"
            
            if session.get('vram_required_bytes'):
                vram_required = self.format_memory_size(session['vram_required_bytes'])
                content += f"*   **VRAM требуется:** {vram_required}\n"
            
            # Слои модели
            if session.get('model_layers_total') or session.get('model_layers_offloaded'):
                content += "\n### Размещение слоёв:\n\n"
                content += "| Параметр | Значение |\n"
                content += "|----------|----------|\n"
                
                if session.get('model_layers_total'):
                    content += f"| Всего слоев | {session['model_layers_total']} |\n"
                if session.get('model_layers_offloaded'):
                    content += f"| Offloaded на GPU | {session['model_layers_offloaded']} |\n"
                
                content += "\n"
            
            # Время загрузки
            if session.get('loading_duration_seconds'):
                duration = self.format_duration(session['loading_duration_seconds'])
                content += f"*   Время запуска runner: {duration}\n"
            
            # API запросы для этой сессии
            session_requests = requests_by_session.get(session_key, [])
            if session_requests:
                content += "\n### Обслуженные API-запросы (GIN)\n"
                content += "| Время | Статус | Задержка | IP | Метод | Путь |\n"
                content += "|---|---|---|---|---|---|\n"
                
                for req in session_requests:
                    time_str = self.format_timestamp(req.get('_time', ''))
                    status = req.get('status', 'N/A')
                    latency = self.format_duration(req.get('_value', '0'))
                    ip = req.get('client_ip', 'N/A')
                    method = req.get('method', 'N/A')
                    path = req.get('endpoint', 'N/A')
                    
                    content += f"| {time_str} | {status} | {latency} | {ip} | {method} | {path} |\n"
                
                content += "\n"
            
            content += "---\n\n"
        
        return content
    
    def generate_system_events_report(self) -> str:
        """Генерирует отчет по системным событиям."""
        if not self.system_events:
            return "Системные события не найдены в InfluxDB.\n"
        
        content = "## Системные события (из InfluxDB)\n\n"
        content += "| Время | Тип | Событие | Детали |\n"
        content += "|-------|-----|---------|--------|\n"
        
        for event in self.system_events:
            time_str = self.format_timestamp(event.get('_time', ''))
            event_type = event.get('event_type', 'N/A')
            event_name = event.get('event_name', 'N/A')
            details = event.get('details', '')
            
            content += f"| {time_str} | {event_type} | {event_name} | {details} |\n"
        
        content += "\n"
        return content
    
    def generate_summary_stats(self) -> str:
        """Генерирует сводную статистику."""
        content = "## Сводная статистика (из InfluxDB)\n\n"
        
        # Подсчитываем статистику
        total_models = len(set(session.get('model', '') for session in self.sessions_data))
        total_sessions = len(self.sessions_data)
        total_requests = len(self.requests_data)
        total_events = len(self.system_events)
        
        # Средняя задержка запросов
        if self.requests_data:
            latencies = []
            for req in self.requests_data:
                try:
                    latency = float(req.get('_value', 0))
                    latencies.append(latency)
                except:
                    pass
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            min_latency = min(latencies) if latencies else 0
        else:
            avg_latency = max_latency = min_latency = 0
        
        # Общее время загрузки
        total_loading_time = 0
        for session in self.sessions_data:
            try:
                loading_time = float(session.get('loading_duration_seconds', 0))
                total_loading_time += loading_time
            except:
                pass
        
        content += f"- **Уникальных моделей:** {total_models}\n"
        content += f"- **Всего сессий:** {total_sessions}\n"
        content += f"- **Всего API запросов:** {total_requests}\n"
        content += f"- **Системных событий:** {total_events}\n"
        content += f"- **Общее время загрузки:** {self.format_duration(str(total_loading_time))}\n"
        
        if self.requests_data:
            content += f"- **Средняя задержка запросов:** {self.format_duration(str(avg_latency))}\n"
            content += f"- **Максимальная задержка:** {self.format_duration(str(max_latency))}\n"
            content += f"- **Минимальная задержка:** {self.format_duration(str(min_latency))}\n"
        
        content += "\n"
        return content
    
    def generate_full_report(self) -> str:
        """Генерирует полный отчет."""
        print("📝 Генерация отчета из данных InfluxDB...")
        
        # Получаем все данные
        self.fetch_models_data()
        self.fetch_sessions_data()
        self.fetch_requests_data()
        self.fetch_system_events()
        
        # Генерируем отчет
        report = []
        
        # Заголовок
        report.append("# Анализ работы Ollama (из InfluxDB)")
        report.append("")
        report.append(f"**Отчет сгенерирован:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Источник данных:** InfluxDB ({self.influxdb_url})")
        report.append(f"**Организация:** {self.org}")
        report.append(f"**Bucket:** {self.bucket}")
        report.append("")
        
        # Сводная статистика
        report.append(self.generate_summary_stats())
        
        # Список моделей
        report.append(self.generate_models_table())
        
        # Системные события
        report.append(self.generate_system_events_report())
        
        # Детали сессий
        report.append(self.generate_sessions_report())
        
        # Подвал
        report.append("---")
        report.append("")
        report.append("*Отчет сгенерирован автоматически из данных InfluxDB*")
        report.append(f"*Скрипт: generate_report_from_influxdb.py*")
        
        return "\n".join(report)
    
    def save_report(self, report_content: str, filename: str = None):
        """Сохраняет отчет в файл."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/influxdb_report_{timestamp}.md"
        
        # Создаем директорию если не существует
        Path(filename).parent.mkdir(exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"💾 Отчет сохранен: {filename}")
        return filename


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Генерация отчета из InfluxDB')
    parser.add_argument('--config', '-c', default='config.json',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--output', '-o', 
                       help='Имя выходного файла (по умолчанию: reports/influxdb_report_TIMESTAMP.md)')
    parser.add_argument('--print', action='store_true',
                       help='Вывести отчет в консоль вместо сохранения в файл')
    
    args = parser.parse_args()
    
    # Создаем генератор
    generator = InfluxDBReportGenerator(args.config)
    
    try:
        # Генерируем отчет
        report_content = generator.generate_full_report()
        
        if args.print:
            # Выводим в консоль
            print("\n" + "="*80)
            print(report_content)
            print("="*80)
        else:
            # Сохраняем в файл
            filename = generator.save_report(report_content, args.output)
            print(f"\n✅ Отчет успешно сгенерирован из данных InfluxDB!")
            print(f"📄 Файл: {filename}")
            
            # Показываем краткую статистику
            lines_count = len(report_content.split('\n'))
            chars_count = len(report_content)
            print(f"📊 Размер отчета: {lines_count} строк, {chars_count} символов")
    
    except Exception as e:
        print(f"❌ Ошибка генерации отчета: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()