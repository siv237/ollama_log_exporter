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
import shlex
import subprocess
import threading
import time

import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# --- Configuration ---
PROMETHEUS_PORT = 9877
JOURNALCTL_UNIT = "ollama.service"
MODEL_MAP_UPDATE_INTERVAL = 300  # 5 minutes
PROCESS_METRICS_UPDATE_INTERVAL = 5 # 5 seconds
OLLAMA_MANIFESTS_PATH = "/root/.ollama/models/manifests/registry.ollama.ai/library/"
LOG_COMMAND = f"journalctl -u {JOURNALCTL_UNIT} -f -n 0"

# --- Metrics ---
OLLAMA_REQUESTS_TOTAL = Counter(
    'ollama_requests_total',
    'Total number of requests to Ollama API',
    ['model', 'endpoint']
)
OLLAMA_REQUEST_LATENCY = Histogram(
    'ollama_request_latency_seconds',
    'Request latency for Ollama API',
    ['model', 'path', 'method']
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
OLLAMA_MODEL_START_TIMESTAMP = Gauge(
    'ollama_model_start_timestamp',
    'Timestamp of the last model start',
    ['model']
)
OLLAMA_MODEL_CONTEXT_SIZE = Gauge(
    'ollama_model_context_size',
    'Context size (ctx-size) of a loaded model',
    ['model']
)
OLLAMA_MODEL_GPU_LAYERS = Gauge(
    'ollama_model_gpu_layers',
    'Number of GPU layers for a loaded model',
    ['model']
)

# --- Global State ---
# model_map will hold the mapping from { "blob_digest": "model_name" }
model_map = {}
# pid_model_cache will hold the mapping from { "pid": "model_name" }
pid_model_cache = {}
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

def build_model_map_from_manifests():
    """Builds a map from blob digest to model name by scanning manifest files."""
    global model_map
    logging.info(f"Building model map from manifests in {OLLAMA_MANIFESTS_PATH}...")
    new_map = {}
    try:
        if not os.path.isdir(OLLAMA_MANIFESTS_PATH):
            logging.warning(f"Manifests directory not found: {OLLAMA_MANIFESTS_PATH}. Skipping map build.")
            return

        for model_name in os.listdir(OLLAMA_MANIFESTS_PATH):
            model_dir = os.path.join(OLLAMA_MANIFESTS_PATH, model_name)
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
                                    new_map[blob_digest] = full_model_name
                                    logging.info(f"  Mapped blob {blob_digest[:12]}... to {full_model_name}")
                                break # Assume one model layer per manifest
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON from manifest: {manifest_file}")
                    except Exception as e:
                        logging.error(f"Error processing manifest {manifest_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while scanning manifests: {e}")

    if not new_map:
        logging.warning("Model map is empty. No models found in manifests directory.")

    model_map = new_map

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

def find_model_for_pid_in_logs(pid):
    logging.info(f"PID {pid} not in cache. Performing reverse search for model name...")
    try:
        # Search backwards in the journal for the line indicating the model name for this PID
        command = f"journalctl -u ollama --no-pager --since '10 minutes ago' | grep '{pid}' | grep 'general.name' | tail -n 1"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout:
            log_line = result.stdout.strip()
            # Expected format: ... print_info: general.name     = Model-Name
            match = re.search(r'general\.name\s*=\s*(.*)', log_line)
            if match:
                model_name = match.group(1).strip()
                logging.info(f"Found model '{model_name}' for PID {pid}. Caching.")
                pid_model_cache[pid] = model_name
                return model_name

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.error(f"Error during reverse log search for PID {pid}: {e}")
    
    logging.warning(f"Could not find model name for PID {pid} in recent logs.")
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
                model_name = pid_model_cache.get(pid)

                if not model_name:
                    # This is the first time we see this PID for a request.
                    # Let's find the model and update the startup metrics.
                    model_name = find_model_for_pid_in_logs(pid)
                    if model_name:
                        pid_model_cache[pid] = model_name
                
                if model_name:
                    # Now, update the request-specific metrics
                    OLLAMA_REQUESTS_TOTAL.labels(model=model_name, endpoint=endpoint.strip('"')).inc()
                    logging.info(f"Logged request for model '{model_name}' (PID: {pid}) on endpoint '{endpoint}'.")
                    latency_match = re.search(r'\|\s+([\d\.]+\w*s)\s+\|', line)
                    if latency_match:
                        latency_sec = parse_duration(latency_match.group(1))
                        OLLAMA_REQUEST_LATENCY.labels(model=model_name, path=endpoint.strip('"'), method=method).observe(latency_sec)
                else:
                    # This can happen if the reverse search fails or if the log appears before the start event
                    logging.warning(f"Request for PID {pid} appeared, but could not determine model.")
                continue

            # Case 2: A new model is being loaded (for cache invalidation)
            start_match = start_server_re.search(line)
            if start_match:
                pid = start_match.group(1)
                if pid in pid_model_cache:
                    logging.info(f"New model started for PID {pid}. Invalidating cache.")
                    del pid_model_cache[pid]

    except FileNotFoundError:
        logging.error(f"Command '{LOG_COMMAND}' not found. Please ensure 'journalctl' is in the system's PATH.")
    except Exception as e:
        logging.error(f"An error occurred while following logs: {e}")


def background_scheduler(interval, task_func):
    """Runs a task function periodically in a loop."""
    while True:
        try:
            task_func()
        except Exception as e:
            logging.error(f"Error in background task {task_func.__name__}: {e}")
        time.sleep(interval)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    
    # Initial build of the model map
    build_model_map_from_manifests()

    # Start Prometheus server
    start_http_server(PROMETHEUS_PORT)
    logging.info(f"Prometheus exporter server started on port {PROMETHEUS_PORT}")

    # Start background tasks in separate threads
    map_updater = threading.Thread(target=background_scheduler, args=(MODEL_MAP_UPDATE_INTERVAL, build_model_map_from_manifests), daemon=True)
    map_updater.start()



    # The main thread will follow logs
    try:
        follow_ollama_logs()
    except KeyboardInterrupt:
        logging.info("Exporter stopped by user.")
