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
    """Fetches logs for a systemd unit from the last 30 minutes and saves them to a file."""
    logging.info(f"Attempting to fetch logs for '{JOURNALCTL_UNIT}' from the last 30 minutes.")
    
    try:
        # Construct the command to get logs
        cmd = f"journalctl -u {JOURNALCTL_UNIT} --since '30 minutes ago' --no-pager -o short-iso"
        
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
    """Parses the startup log file and extracts key events."""
    events = []
    try:
        with open(startup_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # Basic parsing, can be improved with regex for more robustness
            parts = line.strip().split()
            if not parts:
                continue
            
            timestamp_str = " ".join(parts[:3])
            message = " ".join(parts[3:])
            event = {"timestamp": timestamp_str, "source": "unknown", "type": "info", "message": message}

            if "systemd" in message:
                event["source"] = "systemd"
                if "Started ollama.service" in message:
                    event["type"] = "Service Start"
                elif "Deactivated successfully" in message:
                    event["type"] = "Service Stop"
                elif "Scheduled restart job" in message:
                    event["type"] = "Restart Scheduled"
                events.append(event)

            elif "ollama[" in message:
                event["source"] = "ollama"
                if "Listening on" in message:
                    event["type"] = "API Ready"
                elif 'msg="inference compute"' in message:
                    event["type"] = "GPU Found"
                # We only add key ollama events to avoid noise
                if event["type"] != "info":
                    events.append(event)

    except FileNotFoundError:
        logging.warning(f"Startup log file not found: {startup_log_file}")
    except Exception as e:
        logging.error(f"Error parsing startup info: {e}")
    
    return events


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
        startup_events = parse_startup_info()
        if startup_events:
            analysis_content.append("## Service Startup Events\n")
            analysis_content.append("| Timestamp | Source | Event Type | Message |\n")
            analysis_content.append("|---|---|---|---|\n")
            for event in startup_events:
                analysis_content.append(f"| {event['timestamp']} | {event['source']} | {event['type']} | `{event['message']}` |\n")
            analysis_content.append("\n") # It's ok if the file doesn't exist


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
        for idx, line in enumerate(log_lines):
            loading_match = loading_pattern.search(line)
            if loading_match:
                if current_model_info:
                    current_model_info['end_line'] = idx
                    model_events.append(current_model_info)
                
                blob_path = loading_match.group(1)
                digest = loading_match.group(2)
                name = model_sha_map.get(digest, None)
                current_model_info = {
                    'digest': digest,
                    'blob_path': blob_path,
                    'name': name,
                    'arch': None,
                    'lines': [line],
                    'start_line': idx,
                    'end_line': None,
                    'load_duration': None,
                    'required_vram': None
                }
            elif current_model_info:
                current_model_info['lines'].append(line)
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
            
            api_requests.append({
                'time': timestamp.strip(), 'status': status.strip(), 'latency': latency.strip(),
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
            
            for i, m_event in enumerate(model_events):
                model_name = m_event.get('name') or m_event.get('digest', 'Unknown')[:12]
                analysis_content.append(f"\n### Сессия {i+1}: {model_name}\n")
                analysis_content.append(f"- **Модель**: {m_event.get('name', '(не найдено)')}\n")
                analysis_content.append(f"- **SHA256**: {m_event.get('digest', 'N/A')}\n")
                analysis_content.append(f"- **Архитектура**: {m_event.get('arch', '(не найдено)')}\n")
                if m_event.get('load_duration'):
                    analysis_content.append(f"- **Время загрузки**: {m_event['load_duration']}\n")
                if m_event.get('required_vram'):
                    analysis_content.append(f"- **Требуемая VRAM**: {m_event['required_vram']}\n")
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
                    analysis_content.append("| time | ip | method | path |\n")
                    analysis_content.append("|---|---|---|---|\n")
                    for req in session_requests:
                        analysis_content.append(f"| {req['time']} | {req['ip']} | {req['method']} | {req['path']} |\n")
                else:
                    analysis_content.append("    (нет запросов в этой сессии)\n")

        unbound_requests = [req for req in api_requests if not req['model_event']]
        if unbound_requests:
            analysis_content.append("\n### Непривязанные запросы\n\n")
            analysis_content.append("| time | ip | method | path |\n")
            analysis_content.append("|---|---|---|---|\n")
            for req in unbound_requests:
                analysis_content.append(f"| {req['time']} | {req['ip']} | {req['method']} | {req['path']} |\n")

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
