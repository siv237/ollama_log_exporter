#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Log Exporter for Prometheus

Scans Ollama manifest files and runner processes to provide accurate, real-time
Prometheus metrics for model usage, resource consumption, and API latency.
"""

import json
import locale
import logging
import os
import re
import locale
import time
import threading
import psutil
import xml.etree.ElementTree as ET
import subprocess
import shlex
import argparse
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# --- Configuration ---
JOURNALCTL_UNIT = "ollama.service"
MODEL_MAP_UPDATE_INTERVAL = 300  # 5 minutes
PROCESS_METRICS_UPDATE_INTERVAL = 5 # 5 seconds
LOG_COMMAND = f"journalctl -u {JOURNALCTL_UNIT} -f -n 0"

# --- Metrics ---
OLLAMA_REQUESTS_TOTAL = Counter(
    'ollama_requests_total',
    'Total number of requests to Ollama API',
    ['model', 'endpoint', 'session_id']
)
OLLAMA_REQUEST_LATENCY = Histogram(
    'ollama_request_latency_seconds',
    'Request latency for Ollama API',
    ['model', 'path', 'method', 'session_id']
)
OLLAMA_RUNNER_MEMORY_BYTES = Gauge(
    'ollama_runner_memory_bytes',
    'Memory usage of Ollama runner processes in bytes',
    ['model']
)
OLLAMA_RUNNER_CPU_USAGE_PERCENT = Gauge(
    'ollama_runner_cpu_usage_percent',
    'CPU usage of Ollama runner processes in percent',
    ['model']
)

# --- Model & Resource Metrics ---
OLLAMA_MODEL_INFO = Gauge(
    'ollama_model_info',
    'Detailed information about a loaded model',
    ['model', 'pid', 'quantization', 'params_billion', 'n_ctx', 'gpu_layers']
)
OLLAMA_MODEL_VRAM_BUFFER_BYTES = Gauge(
    'ollama_model_vram_buffer_bytes',
    'VRAM used by the model buffer',
    ['model', 'pid']
)
OLLAMA_MODEL_VRAM_KV_CACHE_BYTES = Gauge(
    'ollama_model_vram_kv_cache_bytes',
    'VRAM used by the KV cache',
    ['model', 'pid']
)
OLLAMA_MODEL_ACTIVE = Gauge(
    'ollama_model_active',
    'Indicates if a model runner process is currently active',
    ['model', 'pid']
)
OLLAMA_MODEL_RAM_USAGE_BYTES = Gauge(
    'ollama_model_ram_usage_bytes',
    'RAM usage of a model runner process',
    ['model', 'pid']
)

# --- GPU Hardware Metrics (from Ollama Logs) ---
OLLAMA_GPU_INFO = Gauge(
    'ollama_gpu_info',
    'Static information about the GPU used by Ollama for inference',
    ['gpu_id', 'gpu_name', 'gpu_library', 'gpu_variant', 'gpu_compute', 'gpu_driver']
)
OLLAMA_GPU_VRAM_TOTAL_BYTES = Gauge(
    'ollama_gpu_vram_total_bytes',
    'Total VRAM on the GPU as reported by Ollama',
    ['gpu_id', 'gpu_name']
)
OLLAMA_GPU_VRAM_AVAILABLE_BYTES = Gauge(
    'ollama_gpu_vram_available_bytes',
    'Available VRAM on the GPU at inference time as reported by Ollama',
    ['gpu_id', 'gpu_name']
)

# --- GPU Hardware Metrics (from NVML) ---
OLLAMA_GPU_UTILIZATION_PERCENT = Gauge(
    'ollama_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_uuid']
)
OLLAMA_GPU_MEMORY_USAGE_PERCENT = Gauge(
    'ollama_gpu_memory_usage_percent',
    'GPU memory usage percentage',
    ['gpu_uuid']
)

# --- Global State ---
# model_map will hold the mapping from { "blob_digest": "model_name" }
model_map = {}
# pid_model_cache will hold the mapping from { "pid": { "model_name": "...", "quantization": "...", ... } }
pid_model_cache = {}
# Global sets to track processed entities
processed_gpus = set()  # GPUs we have already exported info for
# Keep track of active runner processes to clean up metrics
active_runners = set()

# Set locale to handle Russian month names
try:
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Russian_Russia.1251')
    except locale.Error:
        logging.warning("Could not set Russian locale. Log timestamp parsing might fail.")

def parse_duration(duration_str):
    """Converts a duration string (e.g., 5.3s, 10ms) to seconds."""
    val = float(re.findall(r'[\d\.]+', duration_str)[0])
    if 'µs' in duration_str:
        return val / 1_000_000
    if 'ms' in duration_str:
        return val / 1_000
    if 'm' in duration_str:
        return val * 60
    if 'h' in duration_str:
        return val * 3600
    return val # Assumes seconds if no unit

def build_model_map_from_manifests(models_base_path):
    """Builds a map from blob digest to model name by scanning manifest files."""
    global model_map
    temp_model_map = {}

    # Construct the full path to the manifests directory
    manifests_path = os.path.join(os.path.expanduser(models_base_path), "manifests", "registry.ollama.ai", "library")
    logging.info(f"Building model map from manifests in {manifests_path}...")
    try:
        if not os.path.isdir(manifests_path):
            logging.warning(f"Manifests directory not found: {manifests_path}. Skipping map build.")
            return

        for model_name in os.listdir(manifests_path):
            model_dir = os.path.join(manifests_path, model_name)
            if os.path.isdir(model_dir):
                for tag in os.listdir(model_dir):
                    manifest_file = os.path.join(model_dir, tag)
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_data = json.load(f)
                        
                        layers = manifest_data.get('layers', [])
                        for layer in layers:
                            if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                                blob_digest = layer.get('digest', '').replace('sha256:', '')
                                if blob_digest:
                                    full_model_name = f"{model_name}:{tag}"
                                    temp_model_map[blob_digest] = full_model_name
                                    logging.info(f"  Mapped blob {blob_digest[:12]}... to {full_model_name}")
                                break # Assume one model layer per manifest
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON from manifest: {manifest_file}")
                    except Exception as e:
                        logging.error(f"Error processing manifest {manifest_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while scanning manifests: {e}")

    if not temp_model_map:
        logging.warning("No models found in manifests. The map is empty.")
    
    model_map = temp_model_map

def parse_llama_server_line(line):
    """Extracts model details from a 'starting llama server' or related log line."""
    """Extract model and launch parameters from a 'starting llama server' log line.

    Expected example:
        msg="starting llama server" cmd="/usr/local/bin/ollama runner --model /root/.ollama/models/blobs/sha256-<digest> --ctx-size 8192 --batch-size 512 --n-gpu-layers 33 ..."
    """
    details: Dict[str, Any] = {}

    # 1) Try to extract sha256 digest from the --model path
    digest_match = re.search(r'sha256-([0-9a-f]{64})', line)
    if digest_match:
        digest = digest_match.group(1)
        # Map digest to human-readable model name (if present)
        model_name = model_map.get(digest)
        if model_name:
            details['model_name'] = model_name
        details['model_digest'] = digest

    # 2) Tokenise the command part to get flag values
    cmd_match = re.search(r'cmd="([^"]+)"', line)
    if cmd_match:
        cmd_string = cmd_match.group(1)
        try:
            tokens = shlex.split(cmd_string)
            # Iterate tokens to capture flag values
            it = iter(tokens)
            for tok in it:
                if tok == '--ctx-size':
                    details['n_ctx'] = int(next(it, '0'))
                elif tok == '--batch-size':
                    details['batch_size'] = int(next(it, '0'))
                elif tok == '--n-gpu-layers':
                    details['gpu_layers'] = f"{next(it, '0')}/33"  # total layers unknown; store offloaded count only
                elif tok == '--threads':
                    details['threads'] = int(next(it, '0'))
                elif tok == '--parallel':
                    details['parallel'] = int(next(it, '0'))
        except Exception as e:
            logging.debug(f"Failed to parse flags from cmd: {e}")

    return details


def parse_cmd_args(cmd_string):
    """Parse command line arguments from the log string."""
    args = shlex.split(cmd_string)
    params = {}
    try:
        # Find the 'runner' subcommand to start parsing from there
        runner_index = args.index('runner')
        args_to_parse = args[runner_index + 1:]
        i = 0
        while i < len(args_to_parse):
            if args_to_parse[i].startswith('--'):
                key = args_to_parse[i][2:]
                # Check if the next argument is a value or another flag
                if i + 1 < len(args_to_parse) and not args_to_parse[i+1].startswith('--'):
                    params[key] = args_to_parse[i+1]
                    i += 2
                else:
                    # It's a boolean flag
                    params[key] = True
                    i += 1
            else:
                i += 1
    except (ValueError, IndexError):
        logging.warning(f"Could not parse cmd args from: {cmd_string}")
    return params

def parse_inference_compute_line(line):
    """Parses the 'inference compute' log line to get detailed GPU info."""
    details = {}
    try:
        parts = shlex.split(line)
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                # Clean up keys and values
                key = key.strip()
                value = value.strip('"')
                if key == 'total' or key == 'available':
                    # Convert GiB/MiB to bytes
                    size_bytes = 0
                    if 'GiB' in value:
                        size_bytes = float(value.replace('GiB', '').strip()) * (1024**3)
                    elif 'MiB' in value:
                        size_bytes = float(value.replace('MiB', '').strip()) * (1024**2)
                    details[f'{key}_vram'] = size_bytes
                else:
                    details[key] = value
    except Exception as e:
        logging.warning(f"Could not parse inference compute line: '{line}'. Error: {e}")
    return details

def parse_kv_cache_line(line):
    """Parses a log line to extract KV cache size.""" 
    args = shlex.split(line)
    params = {}
    try:
        # Find the 'runner' subcommand to start parsing from there
        runner_index = args.index('runner')
        args_to_parse = args[runner_index + 1:]
        i = 0
        while i < len(args_to_parse):
            if args_to_parse[i].startswith('--'):
                key = args_to_parse[i][2:]
                # Check if the next argument is a value or another flag
                if i + 1 < len(args_to_parse) and not args_to_parse[i+1].startswith('--'):
                    params[key] = args_to_parse[i+1]
                    i += 2
                else:
                    # It's a boolean flag
                    params[key] = True
                    i += 1
            else:
                i += 1
    except (ValueError, IndexError):
        logging.warning(f"Could not parse cmd args from: {line}")
    return params

def find_model_details_in_logs(pid):
    """When a new PID is found, search recent logs to find its model name and parameters."""
    details = {}
    try:
        # This command filters the entire journal for a specific PID to find the
        # "starting llama server" log entry, which contains model details.
        cmd = f"journalctl -u ollama.service --no-pager -o cat _PID={pid}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        log_lines = result.stdout.strip().split('\n')

        if not log_lines:
            return None

        for line in log_lines:
            if 'starting llama server' in line:
                # This is the primary log entry for model details
                details = parse_llama_server_line(line)
            elif 'inference compute' in line:
                # This log entry contains detailed GPU information
                gpu_details = parse_inference_compute_line(line)
                if gpu_details:
                    # Set GPU info metrics, they are static and only need to be set once
                    gpu_id = gpu_details.get('id')
                    if gpu_id and gpu_id not in processed_gpus:
                        OLLAMA_GPU_INFO.labels(
                            gpu_id=gpu_id,
                            gpu_name=gpu_details.get('name'),
                            gpu_library=gpu_details.get('library'),
                            gpu_variant=gpu_details.get('variant'),
                            gpu_compute=gpu_details.get('compute'),
                            gpu_driver=gpu_details.get('driver')
                        ).set(1)
                        OLLAMA_GPU_VRAM_TOTAL_BYTES.labels(
                            gpu_id=gpu_id, gpu_name=gpu_details.get('name')
                        ).set(gpu_details.get('total_vram', 0))
                        OLLAMA_GPU_VRAM_AVAILABLE_BYTES.labels(
                            gpu_id=gpu_id, gpu_name=gpu_details.get('name')
                        ).set(gpu_details.get('available_vram', 0))
                        processed_gpus.add(gpu_id)

        if 'model_name' in details:
            logging.info(f"Discovered details for model '{details['model_name']}' (PID: {pid}): {details}")
            return details

    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning(f"Could not search journal for PID {pid}.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in find_model_details_in_logs: {e}")

    return None

def follow_ollama_logs():
    """Follows the journalctl log, manages state, and updates metrics."""
    global pid_model_cache
    
    gin_log_re = re.compile(r'ollama\[(\d+)\].*?(POST|GET).*?("/api/chat"|"/api/generate")')
    start_server_re = re.compile(r'ollama\[(\d+)\].*msg="starting llama server"')

    try:
        process = subprocess.Popen(shlex.split(f"sudo {LOG_COMMAND}"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in iter(process.stdout.readline, ''):

            if not line:
                break

            # Case 1: A request is being processed
            gin_match = gin_log_re.search(line)
            if gin_match:
                pid, method, endpoint = gin_match.groups()
                model_details = pid_model_cache.get(str(pid))

                # If not in cache, this is the first time we see this PID. Find its details.
                if not model_details:
                    model_details = find_model_details_in_logs(pid)
                    if model_details:
                        # Cache the details
                        pid_model_cache[str(pid)] = model_details
                        # Set the one-time informational metrics
                        OLLAMA_MODEL_INFO.labels(
                            model=model_details.get('model_name', 'unknown'),
                            pid=str(pid),
                            quantization=model_details.get('quantization', 'unknown'),
                            params_billion=model_details.get('params_billion', 0),
                            n_ctx=model_details.get('n_ctx', 0),
                            gpu_layers=model_details.get('gpu_layers', '0/0')
                        ).set(1)
                        OLLAMA_MODEL_VRAM_BUFFER_BYTES.labels(
                            model=model_details.get('model_name', 'unknown'),
                            pid=str(pid)
                        ).set(model_details.get('vram_buffer_bytes', 0))
                        OLLAMA_MODEL_VRAM_KV_CACHE_BYTES.labels(
                            model=model_details.get('model_name', 'unknown'),
                            pid=str(pid)
                        ).set(model_details.get('vram_kv_cache_bytes', 0))

                if model_details:
                    model_name = model_details.get('model_name', 'unknown')
                    # Now, update the request-specific metrics
                    OLLAMA_REQUESTS_TOTAL.labels(endpoint=endpoint.strip('"'), model=model_name, session_id=str(pid)).inc()
                    latency_match = re.search(r'\|\s+([\d\.]+\w*s)\s+\|', line)
                    if latency_match:
                        latency_sec = parse_duration(latency_match.group(1))
                        OLLAMA_REQUEST_LATENCY.labels(method=method, path=endpoint.strip('"'), model=model_name, session_id=str(pid)).observe(latency_sec)
                else:
                    # This can happen if the reverse search fails or if the log appears before the start event
                    logging.warning(f"Request for PID {pid} appeared, but could not determine model.")
                continue

            # Case 2: A new model is being loaded (for cache invalidation)
            start_match = start_server_re.search(line)
            if start_match:
                pid = start_match.group(1)
                # Always use string for cache lookups
                if str(pid) in pid_model_cache:
                    logging.info(f"New model started for PID {pid}. Invalidating cache.")
                    del pid_model_cache[str(pid)]

    except FileNotFoundError:
        logging.error(f"Command '{LOG_COMMAND}' not found. Please ensure 'journalctl' is in the system's PATH.")
    except Exception as e:
        logging.error(f"An error occurred while following logs: {e}")


def background_scheduler(interval, task_func, *args, **kwargs):
    """Runs a task function periodically in a loop."""
    while True:
        try:
            task_func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in background task {task_func.__name__}: {e}")
        time.sleep(interval)

def update_resource_metrics():
    """Periodically updates resource usage metrics for active models and GPUs."""
    while True:
        try:
            # Create a copy of the dictionary to avoid runtime errors during iteration
            pids_to_check = pid_model_cache.copy()
            
            active_pids = []

            for pid, model_details in pids_to_check.items():
                model_name = model_details.get('model_name', 'unknown')

                # psutil requires integer PID
                if psutil.pid_exists(int(pid)):
                    active_pids.append(pid)
                    try:
                        p = psutil.Process(int(pid))
                        OLLAMA_MODEL_ACTIVE.labels(model=model_name, pid=pid).set(1)
                        OLLAMA_MODEL_RAM_USAGE_BYTES.labels(model=model_name, pid=pid).set(p.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process might have just died, handled in the next block
                        pass
                else:
                    # Process is gone, clean up all associated metrics
                    logging.info(f"Process {pid} for model '{model_name}' is no longer active. Cleaning up metrics.")
                    # Use a try-except block for each metric removal to avoid crashes if a label combination was never created
                    try: OLLAMA_MODEL_ACTIVE.remove(model_name, pid) 
                    except KeyError: pass
                    try: OLLAMA_MODEL_RAM_USAGE_BYTES.remove(model_name, pid) 
                    except KeyError: pass
                    try: 
                        OLLAMA_MODEL_INFO.remove(
                            model_name,
                            pid,
                            model_details.get('quantization', 'unknown'),
                            model_details.get('params_billion', 0),
                            model_details.get('n_ctx', 0),
                            model_details.get('gpu_layers', '0/0')
                        )
                    except KeyError: pass
                    try: OLLAMA_MODEL_VRAM_BUFFER_BYTES.remove(model_name, pid) 
                    except KeyError: pass
                    try: OLLAMA_MODEL_VRAM_KV_CACHE_BYTES.remove(model_name, pid) 
                    except KeyError: pass
                    
                    # Remove from cache
                    if str(pid) in pid_model_cache:
                        del pid_model_cache[str(pid)]

        except Exception as e:
            logging.error(f"Error in resource metrics thread: {e}")

        # --- GPU Metrics Collection ---
        try:
            smi_output = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True)
            root = ET.fromstring(smi_output)

            for gpu in root.findall('gpu'):
                gpu_uuid = gpu.find('uuid').text
                utilization = gpu.find('utilization')
                gpu_util = float(utilization.find('gpu_util').text.replace(' %', ''))
                
                memory = gpu.find('fb_memory_usage')
                total_mem = float(memory.find('total').text.replace(' MiB', ''))
                used_mem = float(memory.find('used').text.replace(' MiB', ''))
                mem_util = (used_mem / total_mem * 100) if total_mem > 0 else 0

                OLLAMA_GPU_UTILIZATION_PERCENT.labels(gpu_uuid=gpu_uuid).set(gpu_util)
                OLLAMA_GPU_MEMORY_USAGE_PERCENT.labels(gpu_uuid=gpu_uuid).set(mem_util)

                # Find processes running on this GPU
                for proc in gpu.find('processes').findall('process_info'):
                    pid_str = proc.find('pid').text
                    if pid_str in pid_model_cache:
                        model_details = pid_model_cache.get(pid_str, {})
                        model_name = model_details.get('model_name', 'unknown')
                        used_gpu_memory_mib = float(proc.find('used_memory').text.replace(' MiB', ''))
                        OLLAMA_MODEL_GPU_USAGE_BYTES.labels(model=model_name, pid=pid_str, gpu_uuid=gpu_uuid).set(used_gpu_memory_mib * 1024 * 1024)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # This will fail gracefully if nvidia-smi is not installed or fails
            pass 
        except Exception as e:
            logging.error(f"Error parsing nvidia-smi output: {e}")

        time.sleep(15) # Update interval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ollama Log Exporter for Prometheus.')
    parser.add_argument('--port', type=int, default=9877, help='Port to expose Prometheus metrics on.')
    parser.add_argument('--models-path', type=str, default='~/.ollama/models', help='Path to the Ollama models directory.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initial build of the model map
    build_model_map_from_manifests(args.models_path)

    # Start the Prometheus server in a separate thread
    start_http_server(args.port)
    logging.info(f"Prometheus exporter server started on port {args.port}")

    # Start the background thread for updating resource metrics
    resource_thread = threading.Thread(target=update_resource_metrics, daemon=True)
    resource_thread.start()
    logging.info("Started background thread for resource monitoring.")

    # Start background task to update the model map
    map_updater = threading.Thread(
        target=background_scheduler, 
        args=(MODEL_MAP_UPDATE_INTERVAL, build_model_map_from_manifests, args.models_path),
        daemon=True
    )
    map_updater.start()

    # The main thread will follow logs
    try:
        follow_ollama_logs()
    except KeyboardInterrupt:
        logging.info("Exporter stopped by user.")
