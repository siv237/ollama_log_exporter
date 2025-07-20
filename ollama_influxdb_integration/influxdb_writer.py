#!/usr/bin/env python3
"""
InfluxDB Writer для интеграции парсера логов Ollama с InfluxDB.
Преобразует данные из парсера в формат InfluxDB Line Protocol.
"""

import json
import re
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
from dateutil import parser as dtparser
import requests


def build_sha_to_name_map_from_manifests(manifests_root):
    """Строит карту sha256 → имя модели по всем манифестам Ollama (универсально, без хардкодов)."""
    sha_to_name = {}
    if not manifests_root or not os.path.exists(manifests_root):
        return sha_to_name
        
    for root, dirs, files in os.walk(manifests_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            # Имя модели из пути: .../library/phi4/latest → phi4:latest
            rel = os.path.relpath(fpath, manifests_root)
            parts = rel.split(os.sep)
            # ищем структуру .../repo/model/tag
            if len(parts) >= 3:
                model_name = f"{parts[-2]}:{parts[-1]}"
            else:
                model_name = parts[-1]
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # digest sha256 слоя модели
                for layer in data.get('layers', []):
                    if layer.get('mediaType', '').endswith('model') and 'sha256:' in layer.get('digest', ''):
                        sha = layer['digest'].split('sha256:')[-1]
                        sha_to_name[sha] = model_name
            except Exception:
                continue
    return sha_to_name


def get_model_name(sha, session=None, sha_to_name_manifests=None):
    """Возвращает имя модели по sha256: сначала ищет в manifests, потом в сессии."""
    if not sha:
        return "unknown"
    if sha_to_name_manifests and sha in sha_to_name_manifests:
        return sha_to_name_manifests[sha]
    if session and session.get('model_name'):
        return session['model_name']
    return "unknown"


class OllamaInfluxDBWriter:
    """Класс для записи данных Ollama в InfluxDB."""
    
    def __init__(self, influxdb_url: str, token: str, org: str, bucket: str, manifests_path: str = None):
        """
        Инициализация writer'а.
        
        Args:
            influxdb_url: URL InfluxDB (например, http://192.168.237.198:8086)
            token: API токен InfluxDB
            org: Организация InfluxDB
            bucket: Bucket для записи данных
            manifests_path: Путь к манифестам Ollama для определения названий моделей
        """
        self.influxdb_url = influxdb_url.rstrip('/')
        self.token = token
        self.org = org
        self.bucket = bucket
        self.write_url = f"{self.influxdb_url}/api/v2/write"
        
        # Строим карту SHA256 → название модели из манифестов
        self.sha_to_name_manifests = build_sha_to_name_map_from_manifests(manifests_path)
        
        # Headers для API запросов
        self.headers = {
            'Authorization': f'Token {token}',
            'Content-Type': 'text/plain; charset=utf-8',
            'Accept': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """Проверяет подключение к InfluxDB."""
        try:
            health_url = f"{self.influxdb_url}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ошибка подключения к InfluxDB: {e}")
            return False
    
    def _escape_tag_value(self, value: str) -> str:
        """Экранирует значения тегов для InfluxDB Line Protocol."""
        if not isinstance(value, str):
            value = str(value)
        
        # Заменяем русские символы на английские для тегов
        ru_to_en = {
            'Сервис запущен': 'Service_Started',
            'Сервис остановлен': 'Service_Stopped', 
            'Использовано CPU': 'CPU_Used',
            'Перезапуск запланирован': 'Restart_Scheduled',
            'API готов к работе': 'API_Ready',
            'Обнаружена GPU': 'GPU_Detected',
            'Неизвестная модель': 'Unknown_Model'
        }
        
        for ru, en in ru_to_en.items():
            if ru in value:
                value = value.replace(ru, en)
        
        # Экранируем запятые, пробелы и знаки равенства в значениях тегов
        return value.replace(',', '\\,').replace(' ', '\\ ').replace('=', '\\=')
    
    def _escape_field_value(self, value: Any) -> str:
        """Форматирует значения полей для InfluxDB Line Protocol."""
        if isinstance(value, str):
            # Строковые значения в кавычках, экранируем кавычки
            return f'"{value.replace(chr(34), chr(92) + chr(34))}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return f'"{str(value)}"'
    
    def _parse_timestamp(self, timestamp_str: str) -> int:
        """Преобразует строку времени в Unix timestamp (наносекунды)."""
        try:
            # Обработка русских дат из systemd журнала
            if 'июл' in timestamp_str or 'янв' in timestamp_str or 'фев' in timestamp_str:
                # Заменяем русские месяцы на английские
                month_map = {
                    'янв': 'Jan', 'фев': 'Feb', 'мар': 'Mar', 'апр': 'Apr',
                    'май': 'May', 'июн': 'Jun', 'июл': 'Jul', 'авг': 'Aug',
                    'сен': 'Sep', 'окт': 'Oct', 'ноя': 'Nov', 'дек': 'Dec'
                }
                for ru, en in month_map.items():
                    timestamp_str = timestamp_str.replace(ru, en)
                
                # Добавляем текущий год если его нет
                if len(timestamp_str.split()) == 3:
                    timestamp_str = f"{timestamp_str} {datetime.now().year}"
            
            dt = dtparser.parse(timestamp_str)
            # Преобразуем в UTC если есть timezone info
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc)
            else:
                # Если timezone не указан, считаем что это UTC
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Возвращаем timestamp в наносекундах
            return int(dt.timestamp() * 1_000_000_000)
        except Exception as e:
            print(f"Ошибка парсинга времени '{timestamp_str}': {e}")
            # Возвращаем текущее время
            return int(time.time() * 1_000_000_000)
    
    def _parse_memory_value(self, memory_str: str) -> Optional[int]:
        """Преобразует строку памяти (например, '10.9 GiB') в байты."""
        if not memory_str or memory_str == 'N/A':
            return None
        
        try:
            # Убираем скобки если есть
            memory_str = memory_str.strip('[]')
            
            # Парсим число и единицу измерения
            parts = memory_str.split()
            if len(parts) != 2:
                return None
            
            value = float(parts[0])
            unit = parts[1].upper()
            
            # Преобразуем в байты
            multipliers = {
                'B': 1,
                'KB': 1024,
                'MB': 1024**2,
                'GB': 1024**3,
                'TB': 1024**4,
                'KIB': 1024,
                'MIB': 1024**2,
                'GIB': 1024**3,
                'TIB': 1024**4
            }
            
            return int(value * multipliers.get(unit, 1))
        except Exception:
            return None
    
    def _parse_duration(self, duration_str: str) -> Optional[float]:
        """Преобразует строку длительности в секунды."""
        if not duration_str or duration_str == 'N/A':
            return None
        
        try:
            # Убираем 's' в конце если есть
            duration_str = duration_str.rstrip('s').strip()
            return float(duration_str)
        except Exception:
            return None
    
    def _parse_latency(self, latency_str: str) -> Optional[float]:
        """Преобразует строку задержки в секунды."""
        if not latency_str or latency_str == 'N/A':
            return None
        
        try:
            # Различные форматы: "8.129942112s", "972.567378ms", "76.853µs"
            latency_str = latency_str.strip()
            
            if latency_str.endswith('s') and not latency_str.endswith('ms') and not latency_str.endswith('µs'):
                return float(latency_str[:-1])
            elif latency_str.endswith('ms'):
                return float(latency_str[:-2]) / 1000
            elif latency_str.endswith('µs'):
                return float(latency_str[:-2]) / 1_000_000
            else:
                # Попробуем как число в секундах
                return float(latency_str)
        except Exception:
            return None
    
    def session_to_influx_points(self, session: Dict[str, Any]) -> List[str]:
        """Преобразует сессию в точки данных InfluxDB Line Protocol."""
        points = []
        
        # Базовые теги для всех метрик сессии
        base_tags = {}
        
        # Модель и SHA256
        sha256 = session.get('sha256', session.get('model_sha256', ''))
        model_name = get_model_name(sha256, session, self.sha_to_name_manifests)
        if sha256:
            base_tags['model_sha256'] = self._escape_tag_value(sha256[:16])  # Первые 16 символов
        if model_name and model_name != 'N/A':
            base_tags['model'] = self._escape_tag_value(model_name)
        
        # PID и session_id (уникальный на основе времени старта + PID)
        pid = session.get('pid')
        if pid:
            base_tags['pid'] = self._escape_tag_value(str(pid))
            # Создаем уникальный session_id для каждой реальной сессии (как в парсере)
            start_time = session.get('start_time', '')
            if start_time:
                # Используем полное время старта для уникальности каждой сессии
                start_time_clean = start_time.replace(':', '').replace('-', '').replace('+', '').replace('T', '_').replace('.', '')
                base_tags['session_id'] = self._escape_tag_value(f"session_{pid}_{start_time_clean}")
            else:
                base_tags['session_id'] = self._escape_tag_value(f"session_{pid}")
        
        # GPU информация
        offload_info = session.get('offload_info', {})
        if offload_info.get('library'):
            base_tags['gpu_library'] = self._escape_tag_value(offload_info['library'])
        
        # Время старта сессии
        start_time = session.get('start_time')
        if not start_time:
            return points
        
        timestamp_ns = self._parse_timestamp(start_time)
        
        # 1. Метрика сессии (ollama_sessions)
        session_tags = base_tags.copy()
        session_tags['state'] = 'loading'  # Начальное состояние
        
        session_fields = {}
        
        # Параметры модели
        if session.get('ctx_size') and session['ctx_size'] != 'N/A':
            try:
                session_fields['ctx_size'] = int(session['ctx_size'])
            except ValueError:
                pass
        
        if session.get('batch_size') and session['batch_size'] != 'N/A':
            try:
                session_fields['batch_size'] = int(session['batch_size'])
            except ValueError:
                pass
        
        if session.get('gpu_layers') and session['gpu_layers'] != 'N/A':
            try:
                session_fields['gpu_layers'] = int(session['gpu_layers'])
            except ValueError:
                pass
        
        if session.get('threads') and session['threads'] != 'N/A':
            try:
                session_fields['threads'] = int(session['threads'])
            except ValueError:
                pass
        
        if session.get('parallel') and session['parallel'] != 'N/A':
            try:
                session_fields['parallel'] = int(session['parallel'])
            except ValueError:
                pass
        
        # Память
        vram_available = self._parse_memory_value(offload_info.get('memory.available'))
        if vram_available:
            session_fields['vram_available_bytes'] = vram_available
        
        vram_required = self._parse_memory_value(offload_info.get('memory.required.full'))
        if vram_required:
            session_fields['vram_required_bytes'] = vram_required
        
        # Время загрузки
        runner_start_time = session.get('runner_start_time')
        if runner_start_time and runner_start_time != 'N/A':
            loading_duration = self._parse_duration(runner_start_time)
            if loading_duration:
                session_fields['loading_duration_seconds'] = loading_duration
        
        # Слои модели
        if offload_info.get('layers.model'):
            try:
                session_fields['model_layers_total'] = int(offload_info['layers.model'])
            except ValueError:
                pass
        
        if offload_info.get('layers.offload'):
            try:
                session_fields['model_layers_offloaded'] = int(offload_info['layers.offload'])
            except ValueError:
                pass
        
        # Создаем точку данных для сессии (записываем даже если нет loading_duration)
        if session_fields or base_tags:  # Записываем если есть хотя бы теги
            # Добавляем базовое поле если нет других полей
            if not session_fields:
                session_fields['session_started'] = 1
            
            tags_str = ','.join([f'{k}={v}' for k, v in session_tags.items()])
            fields_str = ','.join([f'{k}={self._escape_field_value(v)}' for k, v in session_fields.items()])
            points.append(f"ollama_sessions,{tags_str} {fields_str} {timestamp_ns}")
        
        # 2. Метрики запросов (ollama_requests)
        gin_requests = session.get('gin_requests', [])
        for req in gin_requests:
            req_tags = base_tags.copy()
            req_tags['method'] = self._escape_tag_value(req.get('method', 'unknown'))
            req_tags['endpoint'] = self._escape_tag_value(req.get('path', 'unknown'))
            req_tags['status'] = self._escape_tag_value(req.get('status', 'unknown'))
            req_tags['client_ip'] = self._escape_tag_value(req.get('ip', 'unknown'))
            
            req_fields = {}
            
            # Задержка запроса
            latency = self._parse_latency(req.get('latency'))
            if latency:
                req_fields['latency_seconds'] = latency
            
            # Время запроса
            journal_time = req.get('journal_time')
            if not journal_time:
                req_timestamp_ns = self._parse_timestamp(start_time)
            else:
                req_timestamp_ns = self._parse_timestamp(journal_time)
            
            if req_fields:
                req_tags_str = ','.join([f'{k}={v}' for k, v in req_tags.items()])
                req_fields_str = ','.join([f'{k}={self._escape_field_value(v)}' for k, v in req_fields.items()])
                points.append(f"ollama_requests,{req_tags_str} {req_fields_str} {req_timestamp_ns}")
        
        return points
    
    def write_sessions(self, sessions: List[Dict[str, Any]]) -> bool:
        """Записывает список сессий в InfluxDB."""
        all_points = []
        
        for session in sessions:
            points = self.session_to_influx_points(session)
            all_points.extend(points)
        
        if not all_points:
            print("Нет данных для записи в InfluxDB")
            return True
        
        # Объединяем все точки в один запрос
        line_protocol_data = '\n'.join(all_points)
        
        try:
            params = {
                'org': self.org,
                'bucket': self.bucket,
                'precision': 'ns'
            }
            
            response = requests.post(
                self.write_url,
                headers=self.headers,
                params=params,
                data=line_protocol_data,
                timeout=30
            )
            
            if response.status_code == 204:
                print(f"Успешно записано {len(all_points)} точек данных в InfluxDB")
                return True
            else:
                print(f"Ошибка записи в InfluxDB: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Ошибка при записи в InfluxDB: {e}")
            return False
    
    def write_system_events(self, systemd_events: List[Dict], ollama_events: List[Dict]) -> bool:
        """Записывает системные события в InfluxDB."""
        points = []
        
        # Системные события
        for event in systemd_events:
            timestamp_ns = self._parse_timestamp(event.get('timestamp', ''))
            
            tags = {
                'event_type': self._escape_tag_value('systemd'),
                'event_name': self._escape_tag_value(event.get('type', 'unknown'))
            }
            
            fields = {
                'details': event.get('details', ''),
                'value': 1  # Счетчик событий
            }
            
            tags_str = ','.join([f'{k}={v}' for k, v in tags.items()])
            fields_str = ','.join([f'{k}={self._escape_field_value(v)}' for k, v in fields.items()])
            points.append(f"ollama_system_events,{tags_str} {fields_str} {timestamp_ns}")
        
        # События Ollama
        for event in ollama_events:
            timestamp_ns = self._parse_timestamp(event.get('timestamp', ''))
            
            tags = {
                'event_type': self._escape_tag_value('ollama'),
                'event_name': self._escape_tag_value(event.get('type', 'unknown'))
            }
            
            fields = {
                'value': 1  # Счетчик событий
            }
            
            # Добавляем детали события как поля
            details = event.get('details', {})
            if isinstance(details, dict):
                for key, value in details.items():
                    if key in ['total', 'available']:
                        # Память в байтах
                        memory_bytes = self._parse_memory_value(str(value))
                        if memory_bytes:
                            fields[f'{key}_bytes'] = memory_bytes
                    else:
                        fields[key] = str(value)
            
            tags_str = ','.join([f'{k}={v}' for k, v in tags.items()])
            fields_str = ','.join([f'{k}={self._escape_field_value(v)}' for k, v in fields.items()])
            points.append(f"ollama_system_events,{tags_str} {fields_str} {timestamp_ns}")
        
        if not points:
            return True
        
        # Записываем в InfluxDB
        line_protocol_data = '\n'.join(points)
        
        try:
            params = {
                'org': self.org,
                'bucket': self.bucket,
                'precision': 'ns'
            }
            
            response = requests.post(
                self.write_url,
                headers=self.headers,
                params=params,
                data=line_protocol_data,
                timeout=30
            )
            
            if response.status_code == 204:
                print(f"Успешно записано {len(points)} системных событий в InfluxDB")
                return True
            else:
                print(f"Ошибка записи системных событий в InfluxDB: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Ошибка при записи системных событий в InfluxDB: {e}")
            return False


def main():
    """Пример использования InfluxDB Writer."""
    # Конфигурация InfluxDB
    config = {
        'influxdb_url': 'http://192.168.237.198:8086',
        'token': 'your-influxdb-token',  # Замените на реальный токен
        'org': 'ollama-monitoring',
        'bucket': 'ollama-logs'
    }
    
    # Создаем writer
    writer = OllamaInfluxDBWriter(**config)
    
    # Проверяем подключение
    if not writer.test_connection():
        print("Не удалось подключиться к InfluxDB")
        return
    
    print("Подключение к InfluxDB успешно!")
    
    # Пример данных (в реальном использовании данные будут из парсера)
    example_session = {
        'start_time': '2025-07-18T11:59:04+10:00',
        'pid': '3085',
        'model_name': 'gemma3:12b',
        'sha256': 'adca500fad9b54c565ae672184e0c9eb690eb6014ba63f8ec13849d4f73a32d3',
        'ctx_size': '2048',
        'batch_size': '512',
        'gpu_layers': '48',
        'threads': '6',
        'parallel': '1',
        'runner_start_time': '5.52',
        'offload_info': {
            'memory.available': '[10.9 GiB]',
            'memory.required.full': '1.8 GiB',
            'layers.model': '27',
            'layers.offload': '27',
            'library': 'cuda'
        },
        'gin_requests': [
            {
                'journal_time': '2025-07-18T11:59:12+10:00',
                'status': '200',
                'latency': '8.129942112s',
                'ip': '172.17.0.3',
                'method': 'POST',
                'path': '/api/chat'
            }
        ]
    }
    
    # Записываем тестовые данные
    success = writer.write_sessions([example_session])
    if success:
        print("Тестовые данные успешно записаны!")
    else:
        print("Ошибка записи тестовых данных")


if __name__ == '__main__':
    main()