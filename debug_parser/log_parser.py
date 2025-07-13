import subprocess
import logging
import re
import requests
import json

# --- Configuration ---
# The systemd service unit for Ollama
JOURNALCTL_UNIT = "ollama.service"
# The output file where logs will be saved
OUTPUT_FILE = "ollama_log_dump.txt"

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def dump_recent_logs_to_file():
    """Fetches logs for a systemd unit from the last 120 minutes and saves them to a file."""
    logging.info(f"Attempting to fetch logs for '{JOURNALCTL_UNIT}' from the last 120 minutes.")
    
    try:
        # Construct the command to get logs
        cmd = f"journalctl -u {JOURNALCTL_UNIT} --since '120 minutes ago' --no-pager -o short-iso"
        
        # Execute the command
        logging.info(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        log_content = result.stdout
        
        # Write the captured logs to the specified output file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(log_content)
            
        logging.info(f"Successfully saved {len(log_content.splitlines())} lines of logs to '{OUTPUT_FILE}'.")

    except subprocess.CalledProcessError as e:
        logging.error(f"The command failed with exit code {e.returncode}.")
        logging.error(f"Stderr: {e.stderr.strip()}")
    except FileNotFoundError:
        logging.error("Error: 'journalctl' command not found. Please ensure you are on a system with systemd.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

def dump_startup_log():
    """Finds the last startup event for Ollama and saves the context to a file."""
    startup_log_file = "startup_info.txt"
    logging.info(f"Attempting to find last service start for '{JOURNALCTL_UNIT}'.")
    try:
        cmd = f"journalctl -u {JOURNALCTL_UNIT} --no-pager"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        log_lines = result.stdout.splitlines()

        last_start_idx = -1
        for i, line in reversed(list(enumerate(log_lines))):
            if "Started ollama.service" in line or "Starting Ollama Service" in line:
                last_start_idx = i
                break

        if last_start_idx != -1:
            # Get 10 lines before and 10 lines after the start event
            start_slice = max(0, last_start_idx - 10)
            end_slice = last_start_idx + 10
            context_block = log_lines[start_slice:end_slice]
            with open(startup_log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(context_block))
            logging.info(f"Saved startup context to '{startup_log_file}'.")
        else:
            logging.warning("Could not find a startup event in the logs.")

    except Exception as e:
        logging.error(f"An unexpected error occurred while dumping startup log: {e}")


def get_model_sha_map():
    """Fetches Ollama's /api/tags and builds a map: sha256 -> model name."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        data = resp.json()
        sha_map = {}
        for m in data.get("models", []):
            digest = m.get("digest", "")
            name = m.get("name", m.get("model", ""))
            if digest and name:
                sha_map[digest] = name
        return sha_map
    except Exception as e:
        logging.error(f"Could not fetch model list from Ollama API: {e}")
        return {}


def parse_startup_info(startup_log_file="startup_info.txt"):
    """
    Парсит лог старта и возвращает две группы событий:
    - systemd_events: [{timestamp, type, details}]
    - ollama_events: [{timestamp, type, details_dict}]
    """
    import re
    systemd_events = []
    ollama_events = []
    try:
        with open(startup_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts or len(parts) < 4:
                continue
            timestamp_str = " ".join(parts[:3])
            message = " ".join(parts[3:])

            # SYSTEMD EVENTS
            if "systemd" in message:
                if "Started ollama.service" in message:
                    systemd_events.append({
                        "timestamp": timestamp_str,
                        "type": "Сервис запущен",
                        "details": "ollama.service - Ollama Service"
                    })
                elif "Deactivated successfully" in message:
                    systemd_events.append({
                        "timestamp": timestamp_str,
                        "type": "Сервис остановлен",
                        "details": "ollama.service: Deactivated successfully"
                    })
                elif "CPU time consumed" in message:
                    cpu_time = re.search(r'CPU time consumed: ([^,]+)', message)
                    systemd_events.append({
                        "timestamp": timestamp_str,
                        "type": "Использовано CPU",
                        "details": cpu_time.group(1) if cpu_time else message
                    })
                elif "Scheduled restart job" in message:
                    restart_cnt = re.search(r'restart counter is at (\d+)', message)
                    systemd_events.append({
                        "timestamp": timestamp_str,
                        "type": "Перезапуск запланирован",
                        "details": f"restart counter is at {restart_cnt.group(1)}" if restart_cnt else message
                    })

            # OLLAMA EVENTS
            elif "ollama[" in message:
                # API Ready
                if "Listening on" in message:
                    # Example: msg="Listening on [::]:11434 (version 0.6.5)"
                    port = None
                    version = None
                    # Ищем версию Ollama
                    m = re.search(r'Listening on [^ ]*:(\d+).*\(version ([\w.]+)\)', message)
                    if m:
                        port = m.group(1)
                        version = m.group(2)
                    else:
                        # fallback: ищем только порт
                        m2 = re.search(r'Listening on [^ ]*:(\d+)', message)
                        if m2:
                            port = m2.group(1)
                    ollama_events.append({
                        "timestamp": timestamp_str,
                        "type": "API готов к работе",
                        "details": {
                            "Порт": port,
                            "Версия": version
                        }
                    })
                # GPU Found
                elif 'msg="inference compute"' in message:
                    # Example: msg="inference compute" library=cuda variant=v12 compute=8.6 driver=12.4 name="NVIDIA GeForce RTX 3060" vram.total="11.8 GiB" vram.free="11.1 GiB"
                    details = {}
                    for key, val in re.findall(r'(\w+(?:\.\w+)*|projector\.\w+)=((?:"[^"]*")|(?:\[[^\]]*\])|(?:[^\s]+))', message):
                        details[key] = val.strip('"')
                    ollama_events.append({
                        "timestamp": timestamp_str,
                        "type": "Обнаружена GPU",
                        "details": details
                    })
    except FileNotFoundError:
        logging.warning(f"Startup log file not found: {startup_log_file}")
    except Exception as e:
        logging.error(f"Error parsing startup info: {e}")
    return systemd_events, ollama_events



def parse_latency_to_seconds(latency_str):
    """Converts a latency string like '1.2s', '500ms', '250µs' to seconds."""
    latency_str = latency_str.strip()
    try:
        if 'µs' in latency_str:
            return float(latency_str.replace('µs', '')) / 1_000_000.0
        if 'ms' in latency_str:
            return float(latency_str.replace('ms', '')) / 1000.0
        if 's' in latency_str:
            # Handle special case like '1m2.345s'
            if 'm' in latency_str:
                parts = latency_str.removesuffix('s').split('m')
                return float(parts[0]) * 60 + float(parts[1])
            return float(latency_str.replace('s', ''))
        return float(latency_str) # Assume seconds if no unit
    except (ValueError, IndexError):
        return -1.0 # Return a sentinel value for unparseable latency

def analyze_log_dump():
    """Reads the log dump file and creates a structured analysis file."""
    logging.info(f"Analyzing log file '{OUTPUT_FILE}'...")
    analysis_output_file = "log_analysis.md"

    model_sha_map = get_model_sha_map()

    # --- Модельные подробности ---
    model_events = []
    current_model_info = None
    loading_pattern = re.compile(r'loading" model=(\S*sha256-([a-f0-9]{64}))')
    arch_pattern = re.compile(r'architecture=([\w\d\-_.]+)')
    name_pattern = re.compile(r'general\.name\s*(?:str)?\s*=\s*([\w\d\-_.: ]+)')

    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

        analysis_content = ["# Log Analysis\n"]

        # Prepend startup info if it exists
        systemd_events, ollama_events = parse_startup_info()
        if systemd_events or ollama_events:
            analysis_content.append("## Service Startup Events\n")
            if systemd_events:
                analysis_content.append("#### Системные события\n\n")
                analysis_content.append("| Время | Событие | Детали |\n")
                analysis_content.append("|---|---|---|\n")
                for ev in systemd_events:
                    analysis_content.append(f"| {ev['timestamp']} | {ev['type']} | {ev['details']} |\n")
                analysis_content.append("\n")
            if ollama_events:
                analysis_content.append("#### Ollama события\n\n")
                for ev in ollama_events:
                    if ev['type'] == 'API готов к работе':
                        analysis_content.append("- **API готов к работе:**\n")
                        analysis_content.append(f"  - Время: {ev['timestamp']}\n")
                        version = ev['details'].get('Версия') or '?'
                        port = ev['details'].get('Порт') or '?'
                        analysis_content.append(f"  - Версия: {version}\n")
                        analysis_content.append(f"  - Порт: {port}\n\n")
                    elif ev['type'] == 'Обнаружена GPU':
                        analysis_content.append("- **Обнаружена GPU:**\n")
                        analysis_content.append(f"  - Время: {ev['timestamp']}\n\n")
                        details = ev['details']
                        if details:
                            analysis_content.append("  | Параметр | Значение |\n  |---|---|\n")
                            for k, v in details.items():
                                analysis_content.append(f"  | {k} | {v} |\n")
                        analysis_content.append("\n")


        # --- Data Collection Pass ---
        resource_events = []
        gpu_layers_pattern = re.compile(r'layers\.gpu=(\d+/\d+)')

        for idx, line in enumerate(log_lines):
            if 'msg="system memory"' in line:
                total_mem = re.search(r'total="([^"]+)"', line)
                free_mem = re.search(r'free="([^"]+)"', line)
                if total_mem and free_mem:
                    info_str = f"- System Memory: Total = {total_mem.group(1)}, Free = {free_mem.group(1)}"
                    resource_events.append({'line_num': idx, 'type': 'system', 'content': info_str})

            if 'msg=offload' in line:
                library_match = re.search(r'library=(\w+)', line)
                available_mem_match = re.search(r'memory\.available="([^"]+)"', line)
                if library_match and available_mem_match:
                    gpu_id_match = re.search(r'gpu=(\d+)', line)
                    gpu_id = gpu_id_match.group(1) if gpu_id_match else '0'
                    library = library_match.group(1)
                    vram = available_mem_match.group(1).strip('[]')
                    info_str = f"- GPU {gpu_id} ({library}): Available VRAM = {vram}"
                    resource_events.append({'line_num': idx, 'type': 'gpu', 'content': info_str})

            layers_match = gpu_layers_pattern.search(line)
            if layers_match:
                layers_info = layers_match.group(1)
                info_str = f"- Layer Distribution: {layers_info} on GPU"
                resource_events.append({'line_num': idx, 'type': 'layers', 'content': info_str})

        analysis_content.append("## Model Sessions Analysis\n")
        # Собираем подробности по каждому переключению модели (loading)
        def extract_first_timestamp(lines):
            ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2})')
            for l in lines:
                m = ts_pattern.match(l)
                if m:
                    return m.group(1)
            return None

        for idx, line in enumerate(log_lines):
            loading_match = loading_pattern.search(line)
            if loading_match:
                if current_model_info:
                    current_model_info['end_line'] = idx
                    model_events.append(current_model_info)
                
                blob_path = loading_match.group(1)
                digest = loading_match.group(2)
                name = model_sha_map.get(digest, None)
                # Попробуем извлечь время из первой строки сессии
                session_line = line
                ts_match = re.match(r'^(\S+)', session_line)
                session_start_time = ts_match.group(1) if ts_match else None
                current_model_info = {
                    'digest': digest,
                    'blob_path': blob_path,
                    'name': name,
                    'arch': None,
                    'lines': [line],
                    'start_line': idx,
                    'end_line': None,
                    'load_duration': None,
                    'required_vram': None,
                    'runner_start_duration': None,
                    'cuda_buffer_size': None,
                    'cpu_buffer_size': None,
                    'kv_buffer_size': None,
                    'n_ctx': None,
                    'n_batch': None,
                    'start_time': None
                }
            elif current_model_info:
                current_model_info['lines'].append(line)
                # Если это первая строка сессии, пробуем сохранить время старта
                if len(current_model_info['lines']) == 1 and not current_model_info.get('start_time'):
                    ts_match = re.match(r'^(\S+)', line)
                    if ts_match:
                        current_model_info['start_time'] = ts_match.group(1)
                if not current_model_info['arch']:
                    arch_match = arch_pattern.search(line)
                    if arch_match:
                        current_model_info['arch'] = arch_match.group(1)
                if not current_model_info['name']:
                    name_match = name_pattern.search(line)
                    if name_match:
                        current_model_info['name'] = name_match.group(1).strip()
        
        if current_model_info:
            current_model_info['end_line'] = len(log_lines)
            model_events.append(current_model_info)

        # После сбора model_events для каждой сессии ищем корректный timestamp
        for m in model_events:
            m['start_time'] = extract_first_timestamp(m['lines']) or extract_first_timestamp([log_lines[m['start_line']]])

        api_requests = []
        gin_pattern = re.compile(r'\[GIN\] (\S+ - \S+) \| (\d+) \|\s+([^|]+) \|\s+([^|]+) \|\s+(\w+)\s+(\S+)')
        gin_lines = [(idx, line) for idx, line in enumerate(log_lines) if gin_pattern.search(line)]

        model_ranges = [(m['start_line'], m['end_line'], m) for m in model_events]

        for idx, line in gin_lines:
            match = gin_pattern.search(line)
            if not match:
                continue
            timestamp, status, latency, ip, method, path = match.groups()
            # Find which model session this request belongs to
            active_event = None
            clean_path = path.strip().strip('"')

            # Requests for /api/tags are global and not tied to a model session
            if clean_path != '/api/tags':
                for me in model_events:
                    if me['start_line'] <= idx < me['end_line']:
                        active_event = me
                        break
            
            latency_sec = parse_latency_to_seconds(latency)
            api_requests.append({
                'time': timestamp.strip(), 'status': status.strip(), 'latency': f"{latency_sec:.4f}s",
                'ip': ip.strip(), 'method': method.strip(), 'path': path.strip(),
                'model_event': active_event
            })

        if not model_events:
            analysis_content.append("No model load events found.\n")
        else:
            for m_event in model_events:
                for line in m_event['lines']:
                    if not m_event.get('load_duration'):
                        llm_load_match = re.search(r'llm_load.duration=(\S+)', line)
                        if llm_load_match:
                            m_event['load_duration'] = llm_load_match.group(1)
                    
                    if not m_event.get('required_vram'):
                        offload_match = re.search(r'msg=offload.*?memory\.required\.full="([^"]+)"', line)
                        if offload_match:
                            m_event['required_vram'] = offload_match.group(1)

                    if not m_event.get('runner_start_duration'):
                        runner_match = re.search(r'msg="llama runner started in (\S+) seconds"', line)
                        if runner_match:
                            m_event['runner_start_duration'] = runner_match.group(1) + "s"

                    # --- New detailed info ---
                    if not m_event.get('cuda_buffer_size'):
                        cuda_buf_match = re.search(r'load_tensors:\s+CUDA0 model buffer size\s+=\s+([\d.]+\s+\w+B)', line)
                        if cuda_buf_match:
                            m_event['cuda_buffer_size'] = cuda_buf_match.group(1)

                    if not m_event.get('cpu_buffer_size'):
                        cpu_buf_match = re.search(r'load_tensors:\s+CPU_Mapped model buffer size\s+=\s+([\d.]+\s+\w+B)', line)
                        if cpu_buf_match:
                            m_event['cpu_buffer_size'] = cpu_buf_match.group(1)

                    if not m_event.get('n_ctx'):
                        ctx_match = re.search(r'llama_init_from_model: n_ctx\s+=\s+(\d+)', line)
                        if ctx_match:
                            m_event['n_ctx'] = ctx_match.group(1)

                    if not m_event.get('n_batch'):
                        batch_match = re.search(r'llama_init_from_model: n_batch\s+=\s+(\d+)', line)
                        if batch_match:
                            m_event['n_batch'] = batch_match.group(1)

            for i, m_event in enumerate(model_events):
                model_name = m_event.get('name') or m_event.get('digest', 'Unknown')[:12]
                analysis_content.append(f"\n### Сессия {i+1}: {model_name}\n")
                if m_event.get('start_time'):
                    analysis_content.append(f"- **Время старта сессии**: {m_event['start_time']}\n")
                analysis_content.append(f"- **Модель**: {m_event.get('name', '(не найдено)')}\n")

                # --- Размещение модели по устройствам ---
                # Собираем все строки msg=offload
                offload_rows = []
                offload_pattern = re.compile(r'msg=offload.*?library=(\w+).*?memory\.available="([^"]+)".*?memory\.required\.full="([^"]+)".*?memory\.required\.partial="([^"]+)".*?memory\.required\.kv="([^"]+)".*?layers\.model=([\-\d]+).*?layers\.offload=([\-\d]+).*?layers\.split="([^"]*)"')
                for line in m_event['lines']:
                    m = offload_pattern.search(line)
                    if m:
                        lib, avail, req_full, req_part, req_kv, l_model, l_offload, l_split = m.groups()
                        dev = 'CPU' if lib == 'cpu' else 'GPU'
                        offload_rows.append({
                            'dev': dev,
                            'lib': lib,
                            'avail': avail,
                            'req_full': req_full,
                            'req_part': req_part,
                            'req_kv': req_kv,
                            'l_model': l_model,
                            'l_offload': l_offload,
                            'l_split': l_split
                        })
                # --- Новый блок: анализируем load_tensors: offloading ... to CPU/GPU ---
                lt_offload_pattern = re.compile(r'load_tensors: offloading (\d+) (\w+) layers to (CPU|GPU)', re.IGNORECASE)
                lt_offloaded_pattern = re.compile(r'load_tensors: offloaded (\d+)/(\d+) layers to (CPU|GPU)', re.IGNORECASE)
                cpu_buf_size = None
                cpu_kv_size = None
                cpu_output_buf = None
                cpu_compute_graph = False
                for line in m_event['lines']:
                    # CPU buffer size
                    cpu_buf_match = re.search(r'CPU_Mapped model buffer size\s*=\s*([\d.]+\s+\w+B)', line)
                    if cpu_buf_match:
                        cpu_buf_size = cpu_buf_match.group(1)
                    # CPU KV buffer size
                    cpu_kv_match = re.search(r'CPU KV buffer size\s*=\s*([\d.]+\s+\w+B)', line)
                    if cpu_kv_match:
                        cpu_kv_size = cpu_kv_match.group(1)
                    # CPU output buffer
                    cpu_out_match = re.search(r'CPU\s+output buffer size\s*=\s*([\d.]+\s+\w+B)', line)
                    if cpu_out_match:
                        cpu_output_buf = cpu_out_match.group(1)
                    # CPU compute graph
                    if 'compute graph' in line and 'backend=CPU' in line:
                        cpu_compute_graph = True
                    m1 = lt_offload_pattern.search(line)
                    if m1:
                        n_layers, layer_type, dev = m1.groups()
                        offload_rows.append({
                            'dev': dev.upper(),
                            'lib': dev.lower(),
                            'avail': '-',
                            'req_full': '-',
                            'req_part': '-',
                            'req_kv': '-',
                            'l_model': n_layers,
                            'l_offload': '-',
                            'l_split': f'{layer_type} (offloading)'
                        })
                    m2 = lt_offloaded_pattern.search(line)
                    if m2:
                        n_layers, total, dev = m2.groups()
                        offload_rows.append({
                            'dev': dev.upper(),
                            'lib': dev.lower(),
                            'avail': '-',
                            'req_full': '-',
                            'req_part': '-',
                            'req_kv': '-',
                            'l_model': n_layers,
                            'l_offload': '-',
                            'l_split': f'{n_layers}/{total} layers (offloaded)'
                        })
                # Выводим только raw msg=offload и таблицу параметров для каждой строки
                raw_offload_lines = [l.strip() for l in m_event['lines'] if 'msg=offload' in l]
                if raw_offload_lines:
                    analysis_content.append('\n**Offload parameters:**\n')
                    for raw_line in raw_offload_lines:
                        params = re.findall(r'(\w+(?:\.\w+)*|projector\.\w+)=((?:"[^"]*")|(?:\[[^\]]*\])|(?:[^\s]+))', raw_line)
                        if params:
                            analysis_content.append('\n| Parameter | Value |\n|---|---|\n')
                            for k, v in params:
                                analysis_content.append(f'| {k} | {v} |\n')
                analysis_content.append(f"- **SHA256**: {m_event.get('digest', 'N/A')}\n")
                analysis_content.append(f"- **Архитектура**: {m_event.get('arch', '(не найдено)')}\n")
                if m_event.get('load_duration'):
                    analysis_content.append(f"- **Время загрузки**: {m_event['load_duration']}\n")
                if m_event.get('runner_start_duration'):
                    analysis_content.append(f"- **Время запуска runner**: {m_event['runner_start_duration']}\n")
                if m_event.get('required_vram'):
                    analysis_content.append(f"- **Требуемая VRAM**: {m_event['required_vram']}\n")
                if m_event.get('cuda_buffer_size'):
                    analysis_content.append(f"- **CUDA Model Buffer**: {m_event['cuda_buffer_size']}\n")
                if m_event.get('cpu_buffer_size'):
                    analysis_content.append(f"- **CPU Model Buffer**: {m_event['cpu_buffer_size']}\n")
                if m_event.get('kv_buffer_size'):
                    analysis_content.append(f"- **CUDA KV Buffer**: {m_event['kv_buffer_size']}\n")
                if m_event.get('n_ctx'):
                    analysis_content.append(f"- **Context Size (n_ctx)**: {m_event['n_ctx']}\n")
                if m_event.get('n_batch'):
                    analysis_content.append(f"- **Batch Size (n_batch)**: {m_event['n_batch']}\n")
                analysis_content.append(f"- **Эпоха обслуживания**: строки {m_event['start_line']+1}..{m_event['end_line']} лога\n")

                # --- Расширенный анализ ресурсов ---
                session_resources = [res for res in resource_events if m_event['start_line'] <= res['line_num'] < m_event['end_line']]
                
                system_mems = [res['content'] for res in session_resources if res['type'] == 'system']
                gpus = [res['content'] for res in session_resources if res['type'] == 'gpu']
                layers = [res['content'] for res in session_resources if res['type'] == 'layers']

                if not gpus and not layers:
                    analysis_content.append("- **Используемое устройство**: CPU\n")
                
                if system_mems or gpus or layers:
                    analysis_content.append("\n  **Состояние ресурсов в сессии:**\n")
                    if system_mems:
                        analysis_content.extend([f"    {mem}\n" for mem in set(system_mems)])
                    if gpus:
                        analysis_content.extend([f"    {gpu}\n" for gpu in set(gpus)])
                    if layers:
                        analysis_content.extend([f"    {layer}\n" for layer in set(layers)])
                
                analysis_content.append("\n  **Обслуженные API запросы:**\n\n")
                session_requests = [req for req in api_requests if req['model_event'] and req['model_event']['digest'] == m_event['digest'] and req['model_event']['start_line'] == m_event['start_line']]

                if session_requests:
                    analysis_content.append("| time | ip | method | path | Response Time |\n")
                    analysis_content.append("|---|---|---|---|---|\n")
                    for req in session_requests:
                        analysis_content.append(f"| {req['time']} | {req['ip']} | {req['method']} | {req['path']} | {req['latency']} |\n")
                else:
                    analysis_content.append("    (нет запросов в этой сессии)\n")

        unbound_requests = [req for req in api_requests if not req['model_event']]
        if unbound_requests:
            analysis_content.append("\n### Непривязанные запросы\n\n")
            analysis_content.append("| time | ip | method | path | Response Time |\n")
            analysis_content.append("|---|---|---|---|---|\n")
            for req in unbound_requests:
                analysis_content.append(f"| {req['time']} | {req['ip']} | {req['method']} | {req['path']} | {req['latency']} |\n")

        with open(analysis_output_file, 'w', encoding='utf-8') as f:
            f.writelines(analysis_content)
            
        logging.info(f"Analysis saved to '{analysis_output_file}'.")

    except FileNotFoundError:
        logging.error(f"Log dump file '{OUTPUT_FILE}' not found. Please run the dump function first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}")

if __name__ == '__main__':
    # Step 1: Find and dump the last startup log
    dump_startup_log()

    # Step 2: Get the latest logs and save them to a file
    dump_recent_logs_to_file()
    
    # Step 3: Analyze the created log file
    analyze_log_dump()
