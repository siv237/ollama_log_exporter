import subprocess
import logging
import re


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


def get_model_id_map_from_cli():
    """Fetches model list from `ollama list` and builds a map: short_id -> model name."""
    id_map = {}
    try:
        cmd = "ollama list"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]: # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    # ID in `ollama list` is a short prefix of the full digest
                    name = parts[0]
                    short_id = parts[1]
                    id_map[short_id] = name
        logging.info(f"Built model ID map from 'ollama list' with {len(id_map)} entries.")
    except Exception as e:
        logging.error(f"Could not get model list from 'ollama list' command: {e}")
    return id_map


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

    model_id_map = get_model_id_map_from_cli()

    # --- Модельные подробности ---
    model_events = []
    current_model_info = None
    # Pattern to detect any occurrence of a model blob digest in the logs. This generally
    # appears both in the early scheduler lines ("new model will fit … model=/root/.ollama…")
    # and the later runner start command, so it is a more reliable boundary than just the
    # "starting llama server" message.
    digest_pattern = re.compile(r'sha256-([a-f0-9]{64})')
    server_start_pattern = re.compile(r'starting llama server', re.I)
    arch_pattern = re.compile(r'architecture=([\w\d\-_.]+)')
    file_type_pattern = re.compile(r'file_type=([\w\d\-_.]+)')
    num_tensors_pattern = re.compile(r'num_tensors=(\d+)')
    num_kv_pattern = re.compile(r'num_key_values=(\d+)')
    weights_pattern = re.compile(r'msg="model weights"\s+buffer=(\w+)\s+size="([^"]+)"')
    runner_started_pattern = re.compile(r'runner started in ([\d\.]+) seconds', re.I)
    # pattern to capture server cmd params
    server_cmd_re = re.compile(r'--ctx-size (\d+).*?--batch-size (\d+).*?--n-gpu-layers (\d+).*?--threads (\d+).*?--parallel (\d+).*?--port (\d+)', re.S)
    system_mem_pattern = re.compile(r'msg="system memory"\s+total="([^"]+)"\s+free="([^"]+)"')
    # compute graph lines could be parsed similarly in future
    name_pattern = re.compile(r'general\.name\s*(?:str)?\s*=\s*([\w\d\-_.: ]+)')

    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()

        analysis_content = ["# Log Analysis\n"]

        # Helper to convert human-readable sizes to bytes
        def _to_bytes(size_str: str) -> int:
            size_str = size_str.strip().replace('"', '')
            m = re.match(r'([\d\.]+)\s*(KiB|MiB|GiB|KB|MB|GB|B)', size_str, re.I)
            if not m:
                return 0
            num = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ('gib', 'gb'):
                return int(num * 1024 ** 3)
            if unit in ('mib', 'mb'):
                return int(num * 1024 ** 2)
            if unit in ('kib', 'kb'):
                return int(num * 1024)
            return int(num)

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

        pending_sys_total = None
        pending_sys_free = None
        # buffer lines that appear before we detect a new digest; will be attached to the upcoming session
        pre_digest_offload_lines = []

        for idx, line in enumerate(log_lines):
            # 0) Detect explicit server start command – treat as boundary even if digest repeats
            if server_start_pattern.search(line):
                if current_model_info:
                    current_model_info['end_line'] = idx
                    model_events.append(current_model_info)
                    current_model_info = None  # will create anew once we see digest or proceed

            # 1) Detect the digest in any line. If it differs from the current session's digest –
            #    treat this as the start of a NEW model session. This ensures that early metadata
            #    lines (which often precede the «starting llama server» message) are correctly
            #    attributed.
            # Capture system memory lines even before session starts
            if not current_model_info:
                sm_prefetch = system_mem_pattern.search(line)
                if sm_prefetch:
                    pending_sys_total, pending_sys_free = sm_prefetch.groups()

                # Accumulate potential offload lines for the NEXT session (they often precede the digest)
                if 'msg=offload' in line:
                    pre_digest_offload_lines.append(line)

            digest_match = digest_pattern.search(line)
            if digest_match:
                digest = digest_match.group(1)
                if (not current_model_info) or (current_model_info['digest'] != digest):
                    # Close the previous session, if any
                    if current_model_info:
                        current_model_info['end_line'] = idx
                        model_events.append(current_model_info)

                    # Attempt to resolve model name via the short-ID map from `ollama list`
                    name = None
                    for short_id, model_name in model_id_map.items():
                        if digest.startswith(short_id):
                            name = model_name
                            break

                    current_model_info = {
                        'digest': digest,
                        'blob_path': None,
                        'name': name,
                        'arch': None,
                        'file_type': None,
                        'num_tensors': None,
                        'num_key_values': None,
                        'lines': [],
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
                        'start_time': None,
                        'weight_buffers': {}, # Initialize new field
                        'sys_mem_total': None,
                        'sys_mem_free': None,
                    }

                    # If we recorded system memory shortly before session, attach it
                    if pending_sys_free:
                        current_model_info['sys_mem_total'] = pending_sys_total
                        current_model_info['sys_mem_free'] = pending_sys_free
                        pending_sys_total = pending_sys_free = None

                    # attach offload lines that were encountered BEFORE this digest
                    if pre_digest_offload_lines:
                        current_model_info['lines'].extend(pre_digest_offload_lines)
                        pre_digest_offload_lines = []

            # 2) Accumulate lines for the current model session, if any
            #    Also: if we encounter offload lines while inside a session, add them directly.
            if current_model_info:
                current_model_info['lines'].append(line)

                if 'msg=offload' in line:
                    pre_digest_offload_lines.append(line)  # keep for completeness in case of split or mis-detection

                # Save first timestamp as session start_time
                if (not current_model_info.get('start_time')):
                    ts_match = re.match(r'^(\S+)', line)
                    if ts_match:
                        current_model_info['start_time'] = ts_match.group(1)

                # Extract auxiliary fields on-the-fly
                if (not current_model_info['arch']):
                    arch_match = arch_pattern.search(line)
                    if arch_match:
                        current_model_info['arch'] = arch_match.group(1)
                if (not current_model_info['file_type']):
                    ft_match = file_type_pattern.search(line)
                    if ft_match:
                        current_model_info['file_type'] = ft_match.group(1)
                if (not current_model_info['num_tensors']):
                    nt_match = num_tensors_pattern.search(line)
                    if nt_match:
                        current_model_info['num_tensors'] = nt_match.group(1)
                if (not current_model_info['num_key_values']):
                    nkv_match = num_kv_pattern.search(line)
                    if nkv_match:
                        current_model_info['num_key_values'] = nkv_match.group(1)
                if (not current_model_info['name']):
                    name_match = name_pattern.search(line)
                    if name_match:
                        current_model_info['name'] = name_match.group(1).strip()

                # Collect weight buffers
                w_match = weights_pattern.search(line)
                if w_match:
                    buf, sz = w_match.groups()
                    weight_dict = current_model_info.setdefault('weight_buffers', {})
                    weight_dict[buf] = sz

                # Capture runner startup duration
                if (not current_model_info.get('runner_start_duration')):
                    rs = runner_started_pattern.search(line)
                    if rs:
                        current_model_info['runner_start_duration'] = rs.group(1) + ' s'

                # Parse starting llama server cmd-line params
                if 'starting llama server' in line:
                    mcmd = server_cmd_re.search(line)
                    if mcmd:
                        (ctx_sz, batch_sz, gpu_layers, threads, parallel, port) = mcmd.groups()
                        current_model_info.update({
                            'ctx_size': ctx_sz,
                            'batch_size': batch_sz,
                            'n_gpu_layers': gpu_layers,
                            'threads': threads,
                            'parallel': parallel,
                            'port': port
                        })

                # Capture system memory at session start
                if (not current_model_info.get('sys_mem_free')):
                    sm = system_mem_pattern.search(line)
                    if sm:
                        total, free = sm.groups()
                        current_model_info['sys_mem_total'] = total
                        current_model_info['sys_mem_free'] = free
        
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

            # Skip health-check or internal endpoints that are not real model requests
            skip_paths = {'/api/tags', '/api/ps', '/api/version', '/'}
            if method.strip() == 'HEAD' or clean_path in skip_paths:
                continue  # ignore

            # Remaining requests: try to map to a model session

            # Requests for /api/tags are global and not tied to a model session
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
            for i, m_event in enumerate(model_events):
                # Определяем человекочитаемое имя модели: сначала general.name, затем architecture, затем префикс digest
                display_name = m_event.get('name') or m_event.get('arch') or m_event.get('digest', 'Unknown')[:12]

                analysis_content.append(f"\n### Сессия {i+1}: {display_name}\n")

                if m_event.get('start_time'):
                    analysis_content.append(f"- **Время старта сессии**: {m_event['start_time']}\n")

                analysis_content.append(f"- **Модель**: {m_event.get('name') or m_event.get('arch') or display_name}\n")
                # --- Technical metadata immediately after model line ---
                if m_event.get('digest'):
                    analysis_content.append(f"- **SHA256**: {m_event['digest']}\n")

                arch_val = m_event.get('arch')
                if arch_val and arch_val.lower() not in ('none', '(не найдено)'):
                    analysis_content.append(f"- **Архитектура**: {arch_val}\n")

                if m_event.get('file_type'):
                    analysis_content.append(f"- **Тип файла**: {m_event['file_type']}\n")

                if m_event.get('num_tensors'):
                    analysis_content.append(f"- **Количество тензоров**: {m_event['num_tensors']}\n")

                if m_event.get('num_key_values'):
                    analysis_content.append(f"- **Количество KV**: {m_event['num_key_values']}\n")

                # --- Command-line start parameters ---
                if m_event.get('ctx_size'):
                    analysis_content.append(f"- **CTX size**: {m_event['ctx_size']}\n")
                if m_event.get('batch_size'):
                    analysis_content.append(f"- **Batch size**: {m_event['batch_size']}\n")
                if m_event.get('n_gpu_layers'):
                    analysis_content.append(f"- **GPU layers**: {m_event['n_gpu_layers']}\n")
                if m_event.get('threads'):
                    analysis_content.append(f"- **Threads**: {m_event['threads']}\n")
                if m_event.get('parallel'):
                    analysis_content.append(f"- **Parallel**: {m_event['parallel']}\n")

                # --- Размещение модели по устройствам ---
                offload_rows = []
                offload_param_re = re.compile(r'(\w+(?:\.\w+)*)=\"?([\[\]\w\.\s\-\/]+)\"?')
                for line in m_event['lines']:
                    if 'msg=offload' not in line:
                        continue
                    params = dict(offload_param_re.findall(line))
                    lib = params.get('library', 'unknown')
                    gpu_idx_match = re.search(r'gpu=(\d+)', line)
                    if lib == 'cpu':
                        dev = 'CPU'
                    else:
                        dev = f"GPU{gpu_idx_match.group(1)}" if gpu_idx_match else 'GPU'

                    # Skip duplicate unknown rows
                    if lib == 'unknown' and any(r for r in offload_rows if r['dev'] == dev and r['lib'] != 'unknown'):
                        continue

                    offload_rows.append({
                        'dev': dev,
                        'lib': lib,
                        'avail': params.get('memory.available', '-') ,
                        'req_full': params.get('memory.required.full', '-') ,
                        'req_part': params.get('memory.required.partial', '-') ,
                        'req_kv': params.get('memory.required.kv', '-') ,
                        'l_model': params.get('layers.model', '-') ,
                        'l_offload': params.get('layers.offload', '-') ,
                        'l_split': params.get('layers.split', '-') ,
                        'graph_full': params.get('memory.graph.full', '-')
                    })

                # ----- Summary of memory availability at session start -----
                if m_event.get('sys_mem_free') or offload_rows:
                    analysis_content.append("\n**Свободная память при старте:**\n\n")
                    if m_event.get('sys_mem_free'):
                        analysis_content.append(f"- **RAM свободно**: {m_event['sys_mem_free']} из {m_event.get('sys_mem_total','?')}\n")

                    # Use first offload row per device to show VRAM free/required
                    seen_dev = {}
                    for row in offload_rows:
                        dev_key = (row['dev'], row['lib'])
                        if dev_key in seen_dev:
                            prev = seen_dev[dev_key]
                            if (prev['avail'] == '-' or prev['req_full'] == '-') and (row['avail'] != '-' or row['req_full'] != '-'):
                                seen_dev[dev_key] = row
                        else:
                            seen_dev[dev_key] = row

                    for row in seen_dev.values():
                        analysis_content.append(f"- **VRAM свободно на {row['dev']}**: {row['avail']} (требуется {row['req_full']})\n")

                # After collecting offload_rows, output summarized placement table
                if offload_rows:
                    analysis_content.append("\n**Сводка размещения слоёв и памяти:**\n\n")
                    analysis_content.append("| Устройство | Avail | Req.Full | %Model | Req.KV | Model Layers | Offload | Split |\n")
                    analysis_content.append("|---|---|---|---|---|---|---|---|\n")

                    # === Aggregated CPU / GPU usage rows ===
                    cpu_total = 0
                    gpu_total = 0
                    for buf, sz in (m_event.get('weight_buffers') or {}).items():
                        if 'cuda' in buf.lower():
                            gpu_total += _to_bytes(sz)
                        else:
                            cpu_total += _to_bytes(sz)

                    primary_gpu_row = next((r for r in offload_rows if r['dev'].startswith('GPU') and r['req_kv']!='-'), None)
                    if primary_gpu_row:
                        gpu_total += _to_bytes(primary_gpu_row['req_kv'])
                        gpu_total += _to_bytes(primary_gpu_row.get('graph_full','0'))

                    cpu_total_mb = cpu_total / (1024**2) if cpu_total else 0
                    gpu_total_mb = gpu_total / (1024**2) if gpu_total else 0

                    total_model_bytes = cpu_total + gpu_total if (cpu_total + gpu_total)>0 else 1
                    if cpu_total_mb:
                        pct = cpu_total / total_model_bytes * 100
                        analysis_content.append(f"| CPU | {m_event.get('sys_mem_free','-')} | {cpu_total_mb:.1f} MiB | {pct:.0f}% | - | - | - | - |\n")

                    first_gpu = next((r for r in offload_rows if r['dev'].startswith('GPU') and r['lib'] != 'unknown'), None)
                    if not first_gpu:
                        first_gpu = next((r for r in offload_rows if r['dev'].startswith('GPU')), None)
                    if gpu_total_mb and first_gpu:
                        pctg = gpu_total / total_model_bytes * 100
                        # Try to populate Model Layers – take from first_gpu row, otherwise from any row having value
                        model_layers_val = first_gpu['l_model'] if first_gpu['l_model'] != '-' else next((r['l_model'] for r in offload_rows if r['l_model'] != '-'), '-')
                        analysis_content.append(f"| {first_gpu['dev']} | {first_gpu['avail']} | {gpu_total_mb:.1f} MiB | {pctg:.0f}% | {first_gpu['req_kv']} | {model_layers_val} | {first_gpu['l_offload']} | {first_gpu['l_split']} |\n")

                    # Skip rows already covered or unreliable
                    for row in offload_rows:
                        if row['lib'] == 'cpu':
                            continue
                        if first_gpu and row is first_gpu:
                            continue
                        if row['lib'] == 'unknown':
                            continue
                        analysis_content.append(f"| {row['dev']} ({row['lib']}) | {row['avail']} | {row['req_full']} | - | {row['req_kv']} | {row['l_model']} | {row['l_offload']} | {row['l_split']} |\n")

                # --- Service epoch and timing fields ---
                analysis_content.append(f"- **Эпоха обслуживания**: строки {m_event['start_line']+1}..{m_event['end_line']} лога\n")

                if m_event.get('load_duration'):
                    analysis_content.append(f"- **Время загрузки**: {m_event['load_duration']}\n")
                if m_event.get('runner_start_duration'):
                    analysis_content.append(f"- **Время запуска runner**: {m_event['runner_start_duration']}\n")
                if m_event.get('required_vram'):
                    analysis_content.append(f"- **Требуемая VRAM**: {m_event['required_vram']}\n")

                # (Отдельный раздел "Итоговое использование памяти" больше не выводится, так как все данные теперь входят в сводную таблицу)

                # --- Расширенный анализ ресурсов ---
                session_resources = [res for res in resource_events if m_event['start_line'] <= res['line_num'] < m_event['end_line']]
                
                system_mems = [res['content'] for res in session_resources if res['type'] == 'system']
                gpus = [res['content'] for res in session_resources if res['type'] == 'gpu']
                layers = [res['content'] for res in session_resources if res['type'] == 'layers']

                # Determine device usage heuristically
                has_gpu = bool(gpus or layers)
                if not has_gpu:
                    # Fallback: scan session lines for CUDA indications
                    for l in m_event['lines']:
                        if ('backend=CUDA' in l) or ('CUDA0' in l) or ('loaded CUDA backend' in l):
                            has_gpu = True
                            break

                # The detailed per-session resource lines are verbose and often duplicated;
                # omit them to keep the report concise. Device usage is already clear from the table.
                
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
