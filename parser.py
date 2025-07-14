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

def get_model_name(sha, session=None):
    """Возвращает имя модели по SHA, либо найденное в логе, либо из словаря, либо заглушку."""
    if session and session.get('model_name_from_log'):
        return session['model_name_from_log']
    # Новое: если явно указано имя (пусть даже пустое)
    if session and 'model_name' in session:
        name = session['model_name']
        if name:
            return name
        # Если имя пустое, пробуем архитектуру
        arch = session.get('architecture')
        if arch:
            return f"неизвестно (архитектура: {arch})"
        return "неизвестно"
    if not sha:
        return "Неизвестная модель (SHA: N/A)"
    model_names = {
        "fd7b6731c33c57f61767612f56517460ec2d1e2e5a3f0163e0eb3d8d8cb5df20": "Llama-2-7B-Chat",
        "ad361f123f771269d7eb345075f36ae03c20d7d8ffc7acd8d2db88da3b1ed846": "SmallThinker 3B Preview",
        "ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373": "Codegemma-7B"
    }
    return model_names.get(sha, f"Неизвестная модель (SHA: {sha[:12]}...)")

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

def build_sha_to_name_map_from_manifests(manifests_root):
    """Строит карту sha256 → имя модели по манифестам Ollama."""
    sha_to_name = {}
    for root, dirs, files in os.walk(manifests_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            # Имя модели из пути: .../library/gemma3/1b → gemma3:1b
            rel = os.path.relpath(fpath, manifests_root)
            parts = rel.split(os.sep)
            if len(parts) >= 3:
                model_name = f"{parts[-2]}:{parts[-1]}"
            else:
                model_name = parts[-1]
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # digest sha256 модели
                for layer in data.get('layers', []):
                    if layer.get('mediaType', '').endswith('model') and 'sha256:' in layer.get('digest', ''):
                        sha = layer['digest'].split('sha256:')[-1]
                        sha_to_name[sha] = model_name
            except Exception as e:
                continue
    return sha_to_name

def parse_log(log_file, sha_to_name=None):
    """Парсит лог-файл и возвращает список сессий (best practices: сессия начинается с system memory, starting llama server, загрузки модели, digest)."""
    sessions = []
    current_session = None
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        # --- Новое: начало сессии по любому из событий ---
        is_new_session = False
        # 1. system memory
        if 'msg="system memory"' in line:
            is_new_session = True
        # 2. starting llama server
        if 'msg="starting llama server"' in line:
            is_new_session = True
        # 3. загрузка модели (VRAM)
        if 'msg="new model will fit in available VRAM in single GPU, loading"' in line:
            is_new_session = True
        # 4. digest (sha256) в пути к модели
        if re.search(r'sha256-[a-f0-9]{64}', line):
            is_new_session = True
        # --- Закрываем предыдущую сессию, если есть и не пуста ---
        if is_new_session:
            if current_session and (
                current_session.get('sha256') or current_session.get('offload_info') or current_session.get('model_name') or current_session.get('model_sha256')
            ):
                sessions.append(current_session)
            # Начинаем новую сессию
            current_session = {'raw_lines': [], 'start_time': line.split()[0]}
        if not current_session:
            continue  # Игнорируем строки до первой сессии
        current_session['raw_lines'].append(line)
        # --- Извлекаем параметры из строки загрузки модели ---
        if 'msg="new model will fit in available VRAM in single GPU, loading"' in line:
            m_sha = re.search(r'sha256-([a-f0-9]{64})', line)
            if m_sha:
                current_session['model_sha256'] = m_sha.group(1)
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
        # --- offload ---
        if 'msg=offload' in line:
            offload_str = line.split('msg=offload ')[1]
            current_session['offload_info'] = parse_key_value_string(offload_str)
        # --- general.name ---
        if 'general.name' in line:
            m = re.search(r'general\.name\s+str\s*=\s*(.+)$', line)
            if m:
                current_session['model_name'] = m.group(1).strip()
        # --- general.size_label ---
        if 'general.size_label' in line:
            m = re.search(r'general\.size_label\s+str\s*=\s*(.+)$', line)
            if m:
                current_session['size_label'] = m.group(1).strip()
        # --- general.architecture ---
        if 'general.architecture' in line:
            m = re.search(r'general\.architecture\s+str\s*=\s*(.+)$', line)
            if m:
                current_session['architecture'] = m.group(1).strip()
        # --- starting llama server ---
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
            # Имя из карты sha->имя
            if sha_to_name and sha in sha_to_name:
                current_session['model_name'] = sha_to_name[sha]
        # --- runner started ---
        if 'msg="llama runner started' in line:
            time_str = line.split(' in ')[-1].split(' seconds')[0]
            current_session['runner_start_time'] = f"{time_str} s"
    # Добавляем последнюю сессию, если она не пуста
    if current_session and (
        current_session.get('sha256') or current_session.get('offload_info') or current_session.get('model_name') or current_session.get('model_sha256')
    ):
        sessions.append(current_session)
    return sessions

def generate_md_report(sessions, output_file):
    """Генерирует MD-отчет на основе данных сессий (расширено для новых параметров)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Анализ работы Ollama\n\n")
        for i, session in enumerate(sessions):
            model_name = get_model_name(session.get('sha256') or session.get('model_sha256'), session)
            f.write(f"## Сессия {i+1}: {model_name}\n\n")
            f.write(f"*   **Время старта сессии:** {session.get('start_time', 'N/A')}\n")
            f.write(f"*   **Модель:** {model_name}\n")
            f.write(f"*   **SHA256:** {session.get('sha256', session.get('model_sha256', 'N/A'))}\n")
            f.write(f"*   **Архитектура:** {session.get('architecture', 'N/A')}\n")
            f.write(f"*   **Путь к модели:** {session.get('model_path', 'N/A')}\n")
            f.write(f"*   **GPU:** {session.get('gpu', 'N/A')}\n")
            f.write(f"*   **CTX size:** {session.get('ctx_size', 'N/A')}\n")
            f.write(f"*   **Batch size:** {session.get('batch_size', 'N/A')}\n")
            f.write(f"*   **GPU layers:** {session.get('gpu_layers', 'N/A')}\n")
            f.write(f"*   **Threads:** {session.get('threads', 'N/A')}\n")
            f.write(f"*   **Parallel:** {session.get('parallel', 'N/A')}\n")
            f.write(f"*   **VRAM свободно:** {session.get('vram_available', 'N/A')}\n")
            f.write(f"*   **VRAM требуется:** {session.get('vram_required', 'N/A')}\n")
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
    """Сохраняет логи ollama.service за последние 120 минут в reports/ollama_log_dump.txt."""
    os.makedirs('reports', exist_ok=True)
    output_file = os.path.join('reports', 'ollama_log_dump.txt')
    try:
        cmd = "journalctl -u ollama.service --since '120 minutes ago' --no-pager -o short-iso"
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
    os.makedirs('reports', exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            # Оставляем строки с ключевыми словами, включая general.name и general.size_label
            if any(x in line for x in [
                "starting llama server",
                "system memory",
                "msg=offload",
                "llama runner started",
                "general.name",
                "general.size_label",
                "architecture=",
                "general.architecture"
            ]):
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
    sha_to_name = build_sha_to_name_map(filtered_log)
    manifests_root = os.path.join('/root/.ollama/models/manifests')
    sha_to_name.update(build_sha_to_name_map_from_manifests(manifests_root))
    # 5. Анализируем очищенный лог и генерируем отчёт
    parsed_sessions = parse_log(filtered_log, sha_to_name)

    # 6. Анализируем события старта Ollama
    systemd_events, ollama_events = parse_startup_info()

    # 7. Генерируем итоговый отчёт с событиями старта в заголовке
    report_file = os.path.join('reports', 'report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Анализ работы Ollama\n\n")
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
            model_name = get_model_name(session.get('sha256'), session)
            f.write(f"## Сессия {i+1}: {model_name}\n\n")
            f.write(f"*   **Время старта сессии:** {session.get('start_time', 'N/A')}\n")
            f.write(f"*   **Модель:** {model_name}\n")
            f.write(f"*   **SHA256:** {session.get('sha256', 'N/A')}\n")
            f.write(f"*   **CTX size:** {session.get('ctx_size', 'N/A')}\n")
            f.write(f"*   **Batch size:** {session.get('batch_size', 'N/A')}\n")
            f.write(f"*   **GPU layers:** {session.get('gpu_layers', 'N/A')}\n")
            f.write(f"*   **Threads:** {session.get('threads', 'N/A')}\n")
            f.write(f"*   **Parallel:** {session.get('parallel', 'N/A')}\n\n")
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
    print(f"Отчет '{report_file}' успешно сгенерирован.") 