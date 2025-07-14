import re
import os
import subprocess
import logging
import json

def parse_key_value_string(s):
    """Парсит строку с парами ключ=значение."""
    # Используем регулярное выражение для поиска пар ключ="значение в кавычках" или ключ=значение_без_пробелов
    pattern = r'(\w+(?:\.\w+)*)=("\[.*?\]"|\[.*?\]|"[^"]*"|[\w\./\:-]+)'
    matches = re.findall(pattern, s)
    # Убираем кавычки из значений, если они есть
    return {key: value.strip('"') for key, value in matches}

def get_model_name(sha, session=None, sha_to_name_manifests=None, sha_to_name_log=None):
    """Возвращает имя модели по sha256: сначала ищет в manifests, потом в runner-блоке (сессии), потом в логе."""
    if not sha:
        return "N/A"
    if sha_to_name_manifests and sha in sha_to_name_manifests:
        return sha_to_name_manifests[sha]
    if session and session.get('model_name'):
        return session['model_name']
    if sha_to_name_log and sha in sha_to_name_log:
        return sha_to_name_log[sha]
    return f"Неизвестная модель (SHA: {sha[:12]}...)"

def get_model_architecture(sha, sha_to_arch_manifests=None, session=None):
    """Возвращает архитектуру по sha256: сначала ищет в manifests, потом в runner-блоке (сессии)."""
    if sha_to_arch_manifests and sha in sha_to_arch_manifests:
        return sha_to_arch_manifests[sha]
    if session and session.get('architecture'):
        return session['architecture']
    return "N/A"

def build_sha_to_name_map_from_manifests(manifests_root):
    """Строит карту sha256 → имя модели по всем манифестам Ollama (универсально, без хардкодов)."""
    import os, json
    sha_to_name = {}
    for root, dirs, files in os.walk(manifests_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            # Имя модели из пути: .../library/phi4/latest → phi4:latest
            rel = os.path.relpath(fpath, manifests_root)
            parts = rel.split(os.sep)
            # ищем структуру .../<repo>/<model>/<tag>
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

def build_sha_to_name_map(filtered_log_file):
    """Строит карту sha256 → имя модели (с размером) по analysis_filtered_log.txt."""
    import re
    sha_to_name = {}
    lines = []
    with open(filtered_log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'general.name' in line:
            m = re.search(r'general\.name\s+str\s*=\s*(.+)$', line)
            if m:
                model_name = m.group(1).strip()
                # Ищем размер рядом (в пределах 10 строк после general.name)
                size_label = None
                for k in range(i, min(i+10, len(lines))):
                    if 'general.size_label' in lines[k]:
                        m2 = re.search(r'general\.size_label\s+str\s*=\s*(.+)$', lines[k])
                        if m2:
                            size_label = m2.group(1).strip()
                            break
                if size_label:
                    model_name = f"{model_name} ({size_label})"
                # Ищем sha256 в ближайших 20 строках после general.name
                for j in range(i, min(i+20, len(lines))):
                    if 'msg="starting llama server"' in lines[j]:
                        cmd_str = lines[j].split('cmd=')[1].strip().strip('"')
                        cmd_parts = cmd_str.split()
                        params = {}
                        for l, part in enumerate(cmd_parts):
                            if part.startswith('--') and l + 1 < len(cmd_parts):
                                params[part] = cmd_parts[l+1]
                        model_path = params.get('--model', '')
                        sha = model_path.split('sha256-')[-1] if 'sha256-' in model_path else None
                        if sha:
                            sha_to_name[sha] = model_name
                        break
    return sha_to_name

def build_sha_to_arch_map_from_manifests(manifests_root):
    """Строит карту sha256 → архитектура по manifests Ollama."""
    import os, json
    sha_to_arch = {}
    for root, dirs, files in os.walk(manifests_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # ищем архитектуру в config/labels или аналогичных полях
                arch = None
                if 'config' in data and 'labels' in data['config']:
                    arch = data['config']['labels'].get('architecture')
                if not arch:
                    # иногда архитектура может быть в аннотациях или других полях
                    arch = data['config'].get('architecture') if 'config' in data else None
                for layer in data.get('layers', []):
                    if layer.get('mediaType', '').endswith('model') and 'sha256:' in layer.get('digest', ''):
                        sha = layer['digest'].split('sha256:')[-1]
                        if arch:
                            sha_to_arch[sha] = arch
            except Exception:
                continue
    return sha_to_arch

def parse_log(log_file, sha_to_name=None):
    """Парсит лог-файл и возвращает список сессий (stateful: сессия начинается только по runner, VRAM loading или смене PID). Параметры (архитектура, имя и т.д.) извлекаются только из блока runner'а (20 строк после запуска)."""
    import re
    sessions = []
    current_session = None
    current_pid = None
    runner_block_lines = 0
    metadata_block_active = False
    param_buffer = []
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        # --- Буферизация параметров print_info: model params ---
        if 'print_info: model params' in line:
            param_buffer.append(line)
        # --- Извлекаем PID из строки ollama[PID] ---
        m_pid = re.search(r'ollama\[(\d+)\]', line)
        pid = m_pid.group(1) if m_pid else None
        is_new_session = False
        # Новая сессия только по runner, VRAM loading или смене PID одновременно с этими событиями
        if (
            'msg="starting llama server"' in line or
            'msg="new model will fit in available VRAM in single GPU, loading"' in line
        ):
            is_new_session = True
        if is_new_session:
            if current_session and (
                current_session.get('sha256') or current_session.get('offload_info') or current_session.get('model_name') or current_session.get('model_sha256')
            ):
                    sessions.append(current_session)
            # --- ДОБАВЛЕНО: расширяем начало сессии назад на 30 строк ---
            start_idx = max(0, i-30)
            current_session = {'raw_lines': lines[start_idx:i+1], 'start_time': line.split()[0]}
            # --- ДОБАВЛЕНО: если в буфере есть параметры, добавляем их в начало raw_lines новой сессии ---
            if param_buffer:
                current_session['raw_lines'] = param_buffer + current_session['raw_lines']
                param_buffer = []
            current_pid = pid
            if pid:
                current_session['pid'] = pid
            runner_block_lines = 20 if 'msg="starting llama server"' in line else 0
            metadata_block_active = True
        if not current_session:
            continue  # Игнорируем строки до первой сессии
        current_session['raw_lines'].append(line)
        if pid and 'pid' not in current_session:
            current_session['pid'] = pid
        # --- Извлекаем параметры только из блока runner'а (20 строк после запуска) ---
        if runner_block_lines > 0:
            line_sha = None
            m_sha_in_line = re.search(r'sha256-([a-f0-9]{64})', line)
            if m_sha_in_line:
                line_sha = m_sha_in_line.group(1)
            sha_match = (not line_sha) or (current_session.get('sha256') and line_sha == current_session['sha256'])
            # --- Новый блок: разрешаем сохранять параметры только из подряд идущих general.* ---
            if metadata_block_active:
                if 'general.name' in line:
                    m = re.search(r'general\.name\s+str\s*=\s*(.+)$', line)
                    if m and 'model_name' not in current_session:
                        current_session['model_name'] = m.group(1).strip()
                if 'general.size_label' in line:
                    m = re.search(r'general\.size_label\s+str\s*=\s*(.+)$', line)
                    if m and 'size_label' not in current_session:
                        current_session['size_label'] = m.group(1).strip()
                if 'general.architecture' in line:
                    m = re.search(r'general\.architecture\s+str\s*=\s*(.+)$', line)
                    if m and 'architecture' not in current_session:
                        current_session['architecture'] = m.group(1).strip()
                # Новый: парсим print_info: model params = ...
                if 'print_info: model params' in line:
                    m = re.search(r'print_info: model params\s*=\s*([\d\.]+\s*[ ]*[BMKbmkg])', line)
                    if m and 'params_label' not in current_session:
                        current_session['params_label'] = m.group(1).strip()
                # Новый: парсим print_info: file size = ...
                if 'print_info: file size' in line:
                    m = re.search(r'print_info: file size\s*=\s*([\d\.]+\s*[GMK]i?B)', line)
                    if not m:
                        m = re.search(r'print_info: file size\s*=\s*([\d\.]+\s*[GMK]i?B)', line.replace('   ', ' '))
                    if not m:
                        m = re.search(r'print_info: file size\s*=\s*([^\(]+)', line)
                    if m and 'file_size_label' not in current_session:
                        current_session['file_size_label'] = m.group(1).strip().split('(')[0].strip()
                # Если строка не содержит general.*, завершаем блок сбора метаданных
                if not any(x in line for x in ['general.name', 'general.size_label', 'general.architecture', 'print_info: model params']):
                    metadata_block_active = False
            if 'msg="new model will fit in available VRAM in single GPU, loading"' in line:
                m_sha = re.search(r'sha256-([a-f0-9]{64})', line)
                if m_sha:
                    current_session['model_sha256'] = m_sha.group(1)
                    current_session['sha256'] = m_sha.group(1)
                    current_session['sha256_extracted_from_vram_loading'] = True
                m_gpu = re.search(r'gpu=([\w\-]+)', line)
                if m_gpu:
                    current_session['gpu'] = m_gpu.group(1)
                m_parallel = re.search(r'parallel=(\d+)', line)
                if m_parallel:
                    current_session['parallel'] = m_parallel.group(1)
                m_avail = re.search(r'available=([0-9]+)', line)
                if m_avail:
                    current_session['vram_available'] = m_avail.group(1)
                m_req = re.search(r'required="([^"]+)"', line)
                if m_req:
                    current_session['vram_required'] = m_req.group(1)
                m_model_path = re.search(r'model=([^\s]+)', line)
                if m_model_path:
                    current_session['model_path'] = m_model_path.group(1)
            if 'msg=offload' in line:
                offload_str = line.split('msg=offload ')[1]
                current_session['offload_info'] = parse_key_value_string(offload_str)
            if 'msg="starting llama server"' in line:
                cmd_str = line.split('cmd=')[1].strip().strip('"')
                cmd_parts = cmd_str.split()
                params = {}
                for j, part in enumerate(cmd_parts):
                    if part.startswith('--') and j + 1 < len(cmd_parts):
                        params[part] = cmd_parts[j+1]
                model_path = params.get('--model', '')
                sha = model_path.split('sha256-')[-1] if 'sha256-' in model_path else None
                current_session['sha256'] = sha
                current_session['ctx_size'] = params.get('--ctx-size', 'N/A')
                current_session['batch_size'] = params.get('--batch-size', 'N/A')
                current_session['gpu_layers'] = params.get('--n-gpu-layers', 'N/A')
                current_session['threads'] = params.get('--threads', 'N/A')
                current_session['parallel'] = params.get('--parallel', 'N/A')
                current_session['port'] = params.get('--port', 'N/A')
                if sha_to_name and sha in sha_to_name:
                    current_session['model_name'] = sha_to_name[sha]
            if 'msg="llama runner started' in line:
                time_str = line.split(' in ')[-1].split(' seconds')[0]
                current_session['runner_start_time'] = f"{time_str} s"
            runner_block_lines -= 1
        # --- вне блока runner'а не извлекаем параметры ---
    if current_session and (
        current_session.get('sha256') or current_session.get('offload_info') or current_session.get('model_name') or current_session.get('model_sha256')
    ):
        # --- ДОБАВЛЕНО: ищем print_info: model params по всей сессии, если не найдено в блоке runner'а ---
        if 'params_label' not in current_session:
            import re
            for l in current_session.get('raw_lines', []):
                m = re.search(r'print_info: model params.*=\s*([\d\.]+\s*[BMK]?)', l)
                if m:
                    current_session['params_label'] = m.group(1).strip()
                    break
        sessions.append(current_session)
    return sessions

def generate_md_report(sessions, output_file):
    """Генерирует MD-отчет только для сессий с SHA256."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Анализ работы Ollama\n\n")
        for i, session in enumerate(sessions):
            sha = session.get('sha256') or session.get('model_sha256')
            if not sha:
                continue  # Пропускаем сессии без SHA256
            model_name = get_model_name(sha, session)
            f.write(f"## Сессия {i+1}: {model_name}\n\n")
            f.write(f"*   **Время старта сессии:** {session.get('start_time', 'N/A')}\n")
            f.write(f"*   **PID процесса Ollama:** {session.get('pid', 'N/A')}\n")
            f.write(f"*   **Модель:** {model_name}\n")
            f.write(f"*   **SHA256:** {sha}\n")
            if session.get('sha256_extracted_from_vram_loading'):
                f.write(f"*   _SHA256 извлечён из строки VRAM loading, runner не запускался_\n")
            f.write(f"*   **Архитектура:** {session.get('architecture', 'N/A')}\n")
            f.write(f"*   **Путь к модели:** {session.get('model_path', 'N/A')}\n")
            f.write(f"*   **GPU:** {session.get('gpu', 'N/A')}\n")
            f.write(f"*   **CTX size:** {session.get('ctx_size', 'N/A')}\n")
            f.write(f"*   **Batch size:** {session.get('batch_size', 'N/A')}\n")
            f.write(f"*   **GPU layers:** {session.get('gpu_layers', 'N/A')}\n")
            f.write(f"*   **Threads:** {session.get('threads', 'N/A')}\n")
            f.write(f"*   **Parallel:** {session.get('parallel', 'N/A')}\n")
            # Удаляем строки с N/A по VRAM
            if session.get('vram_available', 'N/A') != 'N/A':
                f.write(f"*   **VRAM свободно:** {session.get('vram_available')}\n")
            if session.get('vram_required', 'N/A') != 'N/A':
                f.write(f"*   **VRAM требуется:** {session.get('vram_required')}\n")
            f.write(f"*   **Размер модели:** {session.get('size_label', 'N/A')}\n\n")

            f.write("### Свободная память при старте:\n\n")
            ram_free = session.get('ram_free', 'N/A')
            ram_total = session.get('ram_total', 'N/A')
            f.write(f"*   **RAM свободно:** {ram_free} из {ram_total}\n")
            offload_info = session.get('offload_info', {})
            vram_avail = offload_info.get('memory.available', 'N/A')
            vram_req = offload_info.get('memory.required.full', 'N/A')
            f.write(f"*   **VRAM свободно на GPU:** {vram_avail} (требуется {vram_req})\n\n")

            f.write("### Сводка размещения слоёв и памяти:\n\n")
            f.write("| Устройство | Avail      | Req.Full  | %Model | Req.KV    | Model Layers | Offload | Split |\n")
            f.write("|------------|------------|-----------|--------|-----------|--------------|---------|-------|\n")
            layers_model = int(offload_info.get('layers.model', 0))
            layers_offload = int(offload_info.get('layers.offload', 0))
            percent_model = f"{int((layers_offload / layers_model) * 100)}%" if layers_model > 0 else "0%"
            f.write(f"| GPU        | {vram_avail:<10} | {offload_info.get('memory.required.partial', 'N/A'):<9} | {percent_model:<6} | {offload_info.get('memory.required.kv', 'N/A'):<9} | {layers_model:<12} | {layers_offload:<7} | {offload_info.get('layers.split', '-'):<5} |\n\n")

            f.write(f"*   **Время запуска runner:** {session.get('runner_start_time', 'N/A')}\n\n")
            f.write("---\n\n")

def dump_recent_logs_to_file():
    """Сохраняет логи ollama.service за последние 24 часа в reports/ollama_log_dump.txt."""
    os.makedirs('reports', exist_ok=True)
    output_file = os.path.join('reports', 'ollama_log_dump.txt')
    try:
        cmd = "journalctl -u ollama.service --since '24 hours ago' --no-pager -o short-iso"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        log_content = result.stdout
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
        print(f"Сохранено {len(log_content.splitlines())} строк в {output_file}")
    except Exception as e:
        print(f"Ошибка при выгрузке логов: {e}")


def dump_startup_log():
    """Сохраняет контекст последнего запуска ollama.service в reports/startup_info.txt."""
    os.makedirs('reports', exist_ok=True)
    startup_log_file = os.path.join('reports', 'startup_info.txt')
    try:
        cmd = "journalctl -u ollama.service --no-pager"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        log_lines = result.stdout.splitlines()
        last_start_idx = -1
        for i, line in reversed(list(enumerate(log_lines))):
            if "Started ollama.service" in line or "Starting Ollama Service" in line:
                last_start_idx = i
                break
        if last_start_idx != -1:
            start_slice = max(0, last_start_idx - 10)
            end_slice = last_start_idx + 10
            context_block = log_lines[start_slice:end_slice]
            with open(startup_log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(context_block))
            print(f"Контекст старта сохранён в {startup_log_file}")
        else:
            print("Событие запуска ollama.service не найдено в журнале.")
    except Exception as e:
        print(f"Ошибка при выгрузке старта: {e}")


def filter_log(input_file, output_file):
    """Создаёт очищенный лог: убирает строки без полезной информации."""
    import re
    os.makedirs('reports', exist_ok=True)
    # GIN POST только для /api/chat и /api/generate
    gin_post_pattern = re.compile(r'\[GIN\].*\|\s*POST\s+"(/api/chat|/api/generate)"')
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            # Оставляем строки с ключевыми словами, включая general.name и general.size_label, а также только GIN POST /api/chat или /api/generate
            if any(x in line for x in [
                "starting llama server",
                "system memory",
                "msg=offload",
                "llama runner started",
                "general.name",
                "general.size_label",
                "architecture=",
                "general.architecture",
                "print_info: model params",
                "llama_init_from_model"  # <--- ДОБАВЛЕНО!
            ]):
                fout.write(line)
            elif "[GIN]" in line:
                if gin_post_pattern.search(line):
                    fout.write(line)
    print(f"Очищенный лог сохранён в {output_file}")


def parse_startup_info(startup_log_file="reports/startup_info.txt"):
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
                    port = None
                    version = None
                    m = re.search(r'Listening on [^ ]*:(\d+).*(version ([\w.]+))', message)
                    if m:
                        port = m.group(1)
                        version = m.group(3)
                    else:
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
                    details = {}
                    for key, val in re.findall(r'(\w+(?:\.\w+)*|projector\.\w+)=((?:"[^"]*")|(?:\[[^\]]*\])|(?:[^\s]+))', message):
                        details[key] = val.strip('"')
                    ollama_events.append({
                        "timestamp": timestamp_str,
                        "type": "Обнаружена GPU",
                        "details": details
                    })
    except FileNotFoundError:
        print(f"Startup log file not found: {startup_log_file}")
    except Exception as e:
        print(f"Error parsing startup info: {e}")
    return systemd_events, ollama_events


def generate_startup_md_report(systemd_events, ollama_events, output_file):
    """Генерирует markdown-отчёт по событиям старта Ollama."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# События старта Ollama\n\n")
        if systemd_events:
            f.write("## Системные события\n\n")
            f.write("| Время | Событие | Детали |\n")
            f.write("|---|---|---|\n")
            for ev in systemd_events:
                f.write(f"| {ev['timestamp']} | {ev['type']} | {ev['details']} |\n")
            f.write("\n")
        if ollama_events:
            f.write("## Ollama события\n\n")
            for ev in ollama_events:
                if ev['type'] == 'API готов к работе':
                    f.write("- **API готов к работе:**\n")
                    f.write(f"  - Время: {ev['timestamp']}\n")
                    version = ev['details'].get('Версия') or '?' 
                    port = ev['details'].get('Порт') or '?' 
                    f.write(f"  - Версия: {version}\n")
                    f.write(f"  - Порт: {port}\n\n")
                elif ev['type'] == 'Обнаружена GPU':
                    f.write("- **Обнаружена GPU:**\n")
                    f.write(f"  - Время: {ev['timestamp']}\n\n")
                    details = ev['details']
                    if details:
                        f.write("| Параметр | Значение |\n|---|---|\n")
                        for k, v in details.items():
                            f.write(f"| {k} | {v} |\n")
                    f.write("\n")

def parse_gin_requests(log_file):
    """Парсит GIN-строки из лога и возвращает список запросов с деталями, используя время из журнала (первое поле)."""
    import re
    gin_pattern = re.compile(r'^([\d\-:T\+]+)\s+[^:]+: \[GIN\]\s+(\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2}) \| (\d+) \| ([^|]+) \|\s*([^|]+) \|\s*(\w+)\s+"([^"]+)"')
    requests = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = gin_pattern.search(line)
            if m:
                journal_time = m.group(1).strip()  # время из systemd-журнала
                # остальные поля как раньше
                # timestamp = m.group(2).strip()  # внутренний timestamp GIN (не используем)
                status = m.group(3).strip()
                latency = m.group(4).strip()
                ip = m.group(5).strip()
                method = m.group(6).strip()
                path = m.group(7).strip()
                requests.append({
                    'journal_time': journal_time,
                    'status': status,
                    'latency': latency,
                    'ip': ip,
                    'method': method,
                    'path': path,
                    'raw_line': line.strip()
                })
    return requests

def assign_requests_to_sessions(sessions, requests):
    """Привязывает GIN-запросы к сессиям по времени журнала (journal_time)."""
    from dateutil import parser as dtparser
    # Преобразуем start_time сессий в datetime
    for s in sessions:
        s['_dt'] = None
        if s.get('start_time'):
            try:
                s['_dt'] = dtparser.parse(s['start_time'])
            except Exception:
                pass
    # Сортируем сессии по времени
    sessions_sorted = sorted([(i, s) for i, s in enumerate(sessions) if s.get('_dt')], key=lambda x: x[1]['_dt'])
    # Для каждой сессии определяем временной диапазон
    for idx, (i, s) in enumerate(sessions_sorted):
        start = s['_dt']
        end = sessions_sorted[idx+1][1]['_dt'] if idx+1 < len(sessions_sorted) else None
        s['gin_requests'] = []
        for req in requests:
            try:
                req_dt = dtparser.parse(req['journal_time'])
            except Exception:
                continue
            if start and (not end or req_dt < end):
                if req_dt >= start:
                    s['gin_requests'].append(req)
    # Удаляем временные поля
    for s in sessions:
        if '_dt' in s:
            del s['_dt']
    return sessions

def build_sha_to_size_map_from_manifests(manifests_root):
    """Строит карту sha256 → размер модели по всем манифестам Ollama (берёт максимальный size среди всех слоёв с данным sha256)."""
    import os, json
    sha_to_size = {}
    for root, dirs, files in os.walk(manifests_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for layer in data.get('layers', []):
                    if layer.get('mediaType', '').endswith('model') and 'sha256:' in layer.get('digest', ''):
                        sha = layer['digest'].split('sha256:')[-1]
                        size = layer.get('size', None)
                        if size is not None:
                            if sha not in sha_to_size or size > sha_to_size[sha]:
                                sha_to_size[sha] = size
            except Exception:
                continue
    return sha_to_size

def collect_ollama_models(manifests_root):
    """Собирает список всех моделей Ollama из manifests: имя, sha256, размер."""
    import os, json
    models = []
    for root, dirs, files in os.walk(manifests_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, manifests_root)
            parts = rel.split(os.sep)
            if len(parts) >= 3:
                model_name = f"{parts[-2]}:{parts[-1]}"
            else:
                model_name = parts[-1]
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for layer in data.get('layers', []):
                    if layer.get('mediaType', '').endswith('model') and 'sha256:' in layer.get('digest', ''):
                        sha = layer['digest'].split('sha256:')[-1]
                        size = layer.get('size', None)
                        models.append({
                            'name': model_name,
                            'sha256': sha,
                            'size': size
                        })
            except Exception:
                continue
    return models


if __name__ == "__main__":
    # 1. Сохраняем логи за 120 минут
    dump_recent_logs_to_file()
    # 2. Сохраняем контекст последнего старта
    dump_startup_log()
    # 3. Создаём очищенный лог
    filtered_log = os.path.join('reports', 'analysis_filtered_log.txt')
    dump_file = os.path.join('reports', 'ollama_log_dump.txt')
    filter_log(dump_file, filtered_log)
    # 4. Строим карту sha256 → имя модели по analysis_filtered_log.txt
    sha_to_name_log = build_sha_to_name_map(filtered_log)
    manifests_root = os.path.join('/root/.ollama/models/manifests')
    sha_to_name_manifests = build_sha_to_name_map_from_manifests(manifests_root)
    sha_to_arch_manifests = build_sha_to_arch_map_from_manifests(manifests_root)
    sha_to_size_manifests = build_sha_to_size_map_from_manifests(manifests_root)
    # Итоговая карта: приоритет у manifests
    sha_to_name = dict(sha_to_name_manifests)
    for k, v in sha_to_name_log.items():
        if k not in sha_to_name:
            sha_to_name[k] = v
    # 5. Анализируем очищенный лог и генерируем отчёт
    parsed_sessions = parse_log(filtered_log, sha_to_name)

    # --- УБРАНА ТЕСТОВАЯ ЗАГЛУШКА params_label ---
    # (оставлен только реальный механизм поиска params_label)

    # --- ДОПОЛНИТЕЛЬНЫЙ ПРОХОД: сопоставление params_label по SHA256 в окне ±100 строк ---
    with open(filtered_log, 'r', encoding='utf-8') as f:
        all_log_lines = f.readlines()
    params_pairs = []  # список словарей: {'sha256': ..., 'params_label': ...}
    import re
    for idx, line in enumerate(all_log_lines):
        m = re.search(r'print_info: model params\s*=\s*([\d\.]+\s*[BMKbmkg])', line)
        if m:
            params_label = m.group(1).strip()
            sha = None
            # Сначала ищем sha256 в 100 строках вниз
            for j in range(idx+1, min(idx+101, len(all_log_lines))):
                msha = re.search(r'sha256-([a-f0-9]{64})', all_log_lines[j])
                if msha:
                    sha = msha.group(1)
                    break
            # Если не нашли — ищем вверх
            if not sha:
                for j in range(max(0, idx-100), idx):
                    msha = re.search(r'sha256-([a-f0-9]{64})', all_log_lines[j])
                    if msha:
                        sha = msha.group(1)
                        break
            if sha:
                params_pairs.append({'sha256': sha, 'params_label': params_label})
    # Для каждой сессии, если sha256 совпадает и params_label ещё не установлен — добавить
    for session in parsed_sessions:
        sess_sha = session.get('sha256') or session.get('model_sha256')
        if not sess_sha or 'params_label' in session:
            continue
        for pair in params_pairs:
            if pair['sha256'] == sess_sha:
                session['params_label'] = pair['params_label']
                break
    # --- КОНЕЦ ДОПОЛНИТЕЛЬНОГО ПРОХОДА ---

    # --- ДОПОЛНИТЕЛЬНЫЙ ПРОХОД: сопоставление offload_info по PID и времени ---
    # Собираем все offload-блоки с PID и индексом
    offload_blocks = []
    for idx, line in enumerate(all_log_lines):
        if 'msg=offload' in line:
            m_pid = re.search(r'ollama\[(\d+)\]', line)
            pid = m_pid.group(1) if m_pid else None
            offload_blocks.append({'idx': idx, 'pid': pid, 'line': line})
    # Для каждой сессии, если offload_info отсутствует, ищем ближайший offload-блок ниже по индексу с тем же PID
    for session in parsed_sessions:
        if session.get('offload_info'):
            continue
        sess_pid = session.get('pid')
        # Находим индекс первой строки raw_lines в all_log_lines
        raw_lines = session.get('raw_lines', [])
        if not raw_lines:
            continue
        try:
            first_line = raw_lines[0]
            sess_start_idx = next(i for i, l in enumerate(all_log_lines) if l == first_line)
        except Exception:
            continue
        # Ищем offload-блок с тем же PID, ближайший после старта сессии
        for ob in offload_blocks:
            if ob['pid'] == sess_pid and ob['idx'] >= sess_start_idx:
                # Парсим offload_info
                offload_str = ob['line'].split('msg=offload ')[1]
                session['offload_info'] = parse_key_value_string(offload_str)
                break
    # --- КОНЕЦ ДОПОЛНИТЕЛЬНОГО ПРОХОДА ---

    # --- ДОПОЛНИТЕЛЬНЫЙ ПРОХОД: собираем все строки llama_init_from_model для каждой сессии ---
    for session in parsed_sessions:
        raw_lines = session.get('raw_lines', [])
        llama_init_info = [l.strip() for l in raw_lines if 'llama_init_from_model' in l]
        if llama_init_info:
            session['llama_init_info'] = llama_init_info
    # --- КОНЕЦ ДОПОЛНИТЕЛЬНОГО ПРОХОДА ---

    # --- ДОПОЛНИТЕЛЬНЫЙ ПРОХОД: строгий парсинг параметров и уведомлений из llama_init_from_model ---
    import re
    for session in parsed_sessions:
        raw_lines = session.get('raw_lines', [])
        llama_init_info = [l.strip() for l in raw_lines if 'llama_init_from_model' in l]
        if llama_init_info:
            params_rows = []
            notifications = []
            for l in llama_init_info:
                if 'llama_init_from_model:' in l:
                    parts = l.split('llama_init_from_model:', 1)
                    # timestamp — только дата/время (до первого пробела)
                    full_prefix = parts[0].strip()
                    timestamp = full_prefix.split(' ')[0] if full_prefix else ''
                    after_colon = parts[1].strip()
                else:
                    timestamp = ''
                    after_colon = l.strip()
                param_match = re.search(r'(.+?)\s*=\s*([^,]+)', after_colon)
                if param_match:
                    for part in re.split(r',\s*', after_colon):
                        m = re.match(r'(.+?)\s*=\s*(.+)', part)
                        if m:
                            key = m.group(1).strip()
                            val = m.group(2).strip()
                            params_rows.append((key, val))
                else:
                    notif = f'{timestamp} — {after_colon}' if timestamp else after_colon
                    notifications.append(notif)
            if params_rows:
                session['llama_init_params_rows'] = params_rows
            if notifications:
                session['llama_init_notifications'] = notifications
    # --- КОНЕЦ ДОПОЛНИТЕЛЬНОГО ПРОХОДА ---

    # Новый шаг: парсим GIN-запросы и привязываем к сессиям
    gin_requests = parse_gin_requests(dump_file)
    parsed_sessions = assign_requests_to_sessions(parsed_sessions, gin_requests)

    # 6. Анализируем события старта Ollama
    systemd_events, ollama_events = parse_startup_info()

    # 7. Генерируем итоговый отчёт с событиями старта в заголовке
    report_file = os.path.join('reports', 'report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Анализ работы Ollama\n\n")
        # --- Список моделей Ollama ---
        manifests_root = os.path.join('/root/.ollama/models/manifests')
        ollama_models = collect_ollama_models(manifests_root)
        if ollama_models:
            f.write("## Список моделей Ollama (по manifests)\n\n")
            f.write("| Модель | SHA256 | Размер |\n")
            f.write("|--------|------------------------------------------|--------|\n")
            for m in ollama_models:
                sz = m['size']
                if sz is None:
                    sz_str = ''
                elif sz > 1024**3:
                    sz_str = f"{sz//1024//1024//1024} GB"
                elif sz > 1024**2:
                    sz_str = f"{sz//1024//1024} MB"
                else:
                    sz_str = f"{sz} B"
                f.write(f"| {m['name']} | {m['sha256']} | {sz_str} |\n")
            f.write("\n")
        # --- События старта ---
        if systemd_events:
            f.write("## Системные события\n\n")
            f.write("| Время | Событие | Детали |\n")
            f.write("|---|---|---|\n")
            for ev in systemd_events:
                f.write(f"| {ev['timestamp']} | {ev['type']} | {ev['details']} |\n")
            f.write("\n")
        if ollama_events:
            f.write("## Ollama события\n\n")
            for ev in ollama_events:
                if ev['type'] == 'API готов к работе':
                    f.write("- **API готов к работе:**\n")
                    f.write(f"  - Время: {ev['timestamp']}\n")
                    version = ev['details'].get('Версия') or '?' 
                    port = ev['details'].get('Порт') or '?' 
                    f.write(f"  - Версия: {version}\n")
                    f.write(f"  - Порт: {port}\n\n")
                elif ev['type'] == 'Обнаружена GPU':
                    f.write("- **Обнаружена GPU:**\n")
                    f.write(f"  - Время: {ev['timestamp']}\n\n")
                    details = ev['details']
                    if details:
                        f.write("| Параметр | Значение |\n|---|---|\n")
                        for k, v in details.items():
                            f.write(f"| {k} | {v} |\n")
                    f.write("\n")
        # --- Сессии моделей ---
        for i, session in enumerate(parsed_sessions):
            model_name = get_model_name(session.get('sha256'), session, sha_to_name_manifests, sha_to_name_log)
            architecture = get_model_architecture(session.get('sha256'), sha_to_arch_manifests, session)
            size_label = session.get('size_label', 'N/A')
            sha = session.get('sha256')
            size_bytes = sha_to_size_manifests.get(sha) if sha else None
            if size_bytes is not None:
                if size_bytes > 1024**3:
                    size_str = f"{size_bytes//1024//1024//1024} GB"
                elif size_bytes > 1024**2:
                    size_str = f"{size_bytes//1024//1024} MB"
                elif size_bytes > 1024:
                    size_str = f"{size_bytes//1024} KB"
                else:
                    size_str = f"{size_bytes} B"
            else:
                size_str = None
            f.write(f"## Сессия {i+1}: {model_name}\n\n")
            f.write(f"*   **Время старта сессии:** {session.get('start_time', 'N/A')}\n")
            f.write(f"*   **PID процесса Ollama:** {session.get('pid', 'N/A')}\n")
            f.write(f"*   **Модель:** {model_name}\n")
            f.write(f"*   **SHA256:** {session.get('sha256', 'N/A')}")
            if model_name and not model_name.startswith("Неизвестная модель") and model_name != "N/A":
                f.write(f" ({model_name})")
            f.write("\n")
            f.write(f"*   **Архитектура:** {architecture}\n")
            # GPU: собираем все параметры из offload_info
            gpu_info = []
            offload = session.get('offload_info', {})
            if offload:
                gpu_part = f"GPU{offload['gpu']}" if 'gpu' in offload else None
                lib_part = offload.get('library')
                name_part = offload.get('name')
                gpu_str = ''
                if gpu_part:
                    gpu_str += gpu_part
                if lib_part:
                    gpu_str += f" ({lib_part}"
                    if name_part:
                        gpu_str += f", {name_part}"
                    gpu_str += ")"
                elif name_part:
                    if gpu_str:
                        gpu_str += f" ({name_part})"
                    else:
                        gpu_str = name_part
                if gpu_str:
                    f.write(f"*   **GPU:** {gpu_str}\n")
            # Путь к модели
            model_path = session.get('model_path')
            if model_path and model_path != 'N/A':
                f.write(f"*   **Путь к модели:** {model_path}\n")
            f.write(f"*   **CTX size:** {session.get('ctx_size', 'N/A')}\n")
            f.write(f"*   **Batch size:** {session.get('batch_size', 'N/A')}\n")
            f.write(f"*   **GPU layers:** {session.get('gpu_layers', 'N/A')}\n")
            f.write(f"*   **Threads:** {session.get('threads', 'N/A')}\n")
            f.write(f"*   **Parallel:** {session.get('parallel', 'N/A')}\n")
            # Удаляем строки с N/A по VRAM
            if session.get('vram_available', 'N/A') != 'N/A':
                f.write(f"*   **VRAM свободно:** {session.get('vram_available')}\n")
            if session.get('vram_required', 'N/A') != 'N/A':
                f.write(f"*   **VRAM требуется:** {session.get('vram_required')}\n")
            # Вместо строки с size_label/size_str
            params_label = session.get('params_label')
            if params_label:
                f.write(f"*   **Размер модели:** {params_label}\n")
            elif size_label != 'N/A' and size_str:
                f.write(f"*   **Размер модели:** {size_label} ({size_str})\n")
            elif size_label != 'N/A':
                f.write(f"*   **Размер модели:** {size_label}\n")
            elif size_str:
                f.write(f"*   **Размер модели:** {size_str}\n")
            else:
                f.write(f"*   **Размер модели:** N/A\n")

            f.write("### Свободная память при старте:\n\n")
            ram_free = session.get('ram_free', 'N/A')
            ram_total = session.get('ram_total', 'N/A')
            if not (ram_free == 'N/A' and ram_total == 'N/A'):
                f.write(f"*   **RAM свободно:** {ram_free} из {ram_total}\n")
            offload_info = session.get('offload_info', {})
            vram_avail = offload_info.get('memory.available', 'N/A')
            vram_req = offload_info.get('memory.required.full', 'N/A')
            f.write(f"*   **VRAM свободно на GPU:** {vram_avail} (требуется {vram_req})\n\n")

            f.write("### Сводка размещения слоёв и памяти:\n\n")
            f.write("| Устройство | Avail      | Req.Full  | %Model | Req.KV    | Model Layers | Offload | Split |\n")
            f.write("|------------|------------|-----------|--------|-----------|--------------|---------|-------|\n")
            layers_model = int(offload_info.get('layers.model', 0))
            layers_offload = int(offload_info.get('layers.offload', 0))
            percent_model = f"{int((layers_offload / layers_model) * 100)}%" if layers_model > 0 else "0%"
            f.write(f"| GPU        | {vram_avail:<10} | {offload_info.get('memory.required.partial', 'N/A'):<9} | {percent_model:<6} | {offload_info.get('memory.required.kv', 'N/A'):<9} | {layers_model:<12} | {layers_offload:<7} | {offload_info.get('layers.split', '-'):<5} |\n\n")
            
            f.write(f"*   **Время запуска runner:** {session.get('runner_start_time', 'N/A')}\n\n")
            
            # --- GIN-запросы ---
            f.write("### Обслуженные API-запросы (GIN)\n")
            gin_reqs = session.get('gin_requests', [])
            if gin_reqs:
                f.write("| Время | Статус | Задержка | IP | Метод | Путь |\n")
                f.write("|---|---|---|---|---|---|\n")
                for req in gin_reqs:
                    f.write(f"| {req['journal_time']} | {req['status']} | {req['latency']} | {req['ip']} | {req['method']} | {req['path']} |\n")
            else:
                f.write("(нет запросов в этой сессии)\n")
            f.write("---\n\n")

            # --- Параметры инициализации модели (llama_init_from_model) ---
            if session.get('llama_init_info'):
                f.write('<details><summary>Параметры инициализации модели (llama_init_from_model)</summary>\n\n')
                if session.get('llama_init_params_rows'):
                    f.write('| Параметр | Значение |\n|---|---|\n')
                    for k, v in session['llama_init_params_rows']:
                        f.write(f'| {k} | {v} |\n')
                f.write('\n</details>\n')
                if session.get('llama_init_notifications'):
                    f.write('\n### Уведомления\n\n')
                    for notif in session['llama_init_notifications']:
                        f.write(f'{notif}\n\n')
    print(f"Отчет '{report_file}' успешно сгенерирован.") 