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

def analyze_log_dump():
    """Reads the log dump file and creates a structured analysis file."""
    logging.info(f"Analyzing log file '{OUTPUT_FILE}'...")
    analysis_output_file = "log_analysis.txt"
    
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        analysis_content = []
        analysis_content.append("--- Log Analysis ---\n\n")

        # --- 1. Hardware and Resource Analysis ---
        analysis_content.append("## 1. Hardware and Resource Information\n")
        found_resources = False
        for line in log_lines:
            if 'msg="system memory"' in line:
                total_mem = re.search(r'total="([^"]+)"', line)
                free_mem = re.search(r'free="([^"]+)"', line)
                if total_mem and free_mem:
                    analysis_content.append(f"- System Memory: Total = {total_mem.group(1)}, Free = {free_mem.group(1)}\n")
                    found_resources = True

            if 'msg=offload' in line:
                library = re.search(r'library=(\w+)', line)
                available_mem = re.search(r'memory\.available="([^"]+)"', line)
                if library and available_mem:
                    analysis_content.append(f"- GPU Detected: Library = {library.group(1)}, Available VRAM = {available_mem.group(1)}\n")
                    found_resources = True
        
        if not found_resources:
            analysis_content.append("- No hardware and resource information found in this log slice.\n")
        
        analysis_content.append("\n")

        # --- 2. Model Lifecycle Analysis ---
        analysis_content.append("## 2. Model Lifecycle Analysis\n")
        models_by_pid = {}
        log_pattern = re.compile(r'^(\S+Z?)\s+.*?ollama\[(\d+)\]:\s+(.*)')

        for line in log_lines:
            match = log_pattern.match(line)
            if not match:
                continue
            
            timestamp, pid, message = match.groups()

            # A. Model Load Events
            if 'llama_model_loader: - kv' in message:
                if pid not in models_by_pid:
                    models_by_pid[pid] = {'start_time': timestamp, 'params': {}}
                
                kv_match = re.search(r'kv\s+\d+:\s+([^=]+?)\s+(?:str|u32|f32|arr\[str,\d+\])\s+=\s+(.*)', message)
                if kv_match:
                    key, value = kv_match.groups()
                    key = key.strip()
                    # We only care about a few key parameters for this analysis
                    if key in ['general.name', 'general.size_label', 'phi3.block_count', 'phi3.context_length']:
                         models_by_pid[pid]['params'][key] = value.strip()
        
        if models_by_pid:
            for pid, data in models_by_pid.items():
                analysis_content.append(f"- Model Load detected for PID: {pid} at {data['start_time']}\n")
                for key, value in data['params'].items():
                    analysis_content.append(f"    - {key}: {value}\n")
                analysis_content.append("\n")
        else:
            analysis_content.append("- No model load events found in this log slice.\n")

        analysis_content.append("\n")

        # --- 3. API Request Analysis ---
        analysis_content.append("## 3. API Request Analysis\n")
        api_requests = []
        gin_pattern = re.compile(r'\[GIN\] (\S+ - \S+) \| (\d+) \|\s+([^|]+) \|\s+([^|]+) \|\s+(\w+)\s+(\S+)')

        for line in log_lines:
            match = gin_pattern.search(line)
            if match:
                timestamp, status, latency, ip, method, path = match.groups()
                api_requests.append({
                    'time': timestamp.strip(),
                    'status': status.strip(),
                    'latency': latency.strip(),
                    'ip': ip.strip(),
                    'method': method.strip(),
                    'path': path.strip()
                })
        
        if api_requests:
            for req in api_requests:
                analysis_content.append(f"- {req['time']}: {req['method']} {req['path']} from {req['ip']} -> {req['status']} ({req['latency']})\n")
        else:
            analysis_content.append("- No API requests found in this log slice.\n")
        
        analysis_content.append("\n")

        with open(analysis_output_file, 'w', encoding='utf-8') as f:
            f.writelines(analysis_content)
            
        logging.info(f"Analysis saved to '{analysis_output_file}'.")

    except FileNotFoundError:
        logging.error(f"Log dump file '{OUTPUT_FILE}' not found. Please run the dump function first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}")

if __name__ == '__main__':
    # Step 1: Get the latest logs and save them to a file
    dump_recent_logs_to_file()
    
    # Step 2: Analyze the created log file
    analyze_log_dump()
