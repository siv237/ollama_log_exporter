#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Log Exporter for Prometheus (New Version)

Real-time session-based exporter that uses the same stateful logic as the parser
for accurate and comprehensive Prometheus metrics.
"""

import json
import locale
import logging
import os
import re
import time
import threading
import subprocess
import shlex
import argparse
from datetime import datetime
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info

# Set environment for consistent output
os.environ['LANG'] = 'C'
os.environ['LC_ALL'] = 'C'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
JOURNALCTL_UNIT = "ollama.service"
MODEL_MAP_UPDATE_INTERVAL = 300  # 5 minutes
LOG_COMMAND = f"journalctl -u {JOURNALCTL_UNIT} -f -n 10"

# --- Prometheus Metrics ---
# Request metrics
OLLAMA_REQUESTS_TOTAL = Counter(
    'ollama_requests_total',
    'Total number of requests to Ollama API',
    ['model', 'endpoint', 'method', 'status', 'session_id']
)

OLLAMA_REQUEST_LATENCY = Histogram(
    'ollama_request_latency_seconds',
    'Request latency for Ollama API',
    ['model', 'endpoint', 'method', 'session_id']
)

# Session metrics
OLLAMA_SESSION_INFO = Info(
    'ollama_session_info',
    'Information about active Ollama sessions',
    ['session_id', 'model', 'pid']
)

OLLAMA_SESSION_START_TIME = Gauge(
    'ollama_session_start_timestamp_seconds',
    'Unix timestamp when session started',
    ['model', 'session_id', 'pid']
)

OLLAMA_SESSION_RUNNER_START_TIME = Gauge(
    'ollama_session_runner_start_seconds',
    'Time taken to start the runner in seconds',
    ['model', 'session_id', 'pid']
)

# Model configuration metrics
OLLAMA_MODEL_CONTEXT_SIZE = Gauge(
    'ollama_model_context_size',
    'Context size configured for the model',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_BATCH_SIZE = Gauge(
    'ollama_model_batch_size',
    'Batch size configured for the model',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_GPU_LAYERS = Gauge(
    'ollama_model_gpu_layers',
    'Number of GPU layers for the model',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_THREADS = Gauge(
    'ollama_model_threads',
    'Number of threads configured for the model',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_PARALLEL = Gauge(
    'ollama_model_parallel_requests',
    'Number of parallel requests configured',
    ['model', 'session_id', 'pid']
)

# VRAM and memory metrics
OLLAMA_MODEL_VRAM_AVAILABLE_BYTES = Gauge(
    'ollama_model_vram_available_bytes',
    'Available VRAM when model was loaded',
    ['model', 'session_id', 'pid', 'gpu']
)

OLLAMA_MODEL_VRAM_REQUIRED_BYTES = Gauge(
    'ollama_model_vram_required_bytes',
    'VRAM required by the model',
    ['model', 'session_id', 'pid', 'gpu']
)

# Offload metrics
OLLAMA_MODEL_LAYERS_TOTAL = Gauge(
    'ollama_model_layers_total',
    'Total number of model layers',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_LAYERS_OFFLOADED = Gauge(
    'ollama_model_layers_offloaded',
    'Number of layers offloaded to GPU',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_MEMORY_REQUIRED_FULL_BYTES = Gauge(
    'ollama_model_memory_required_full_bytes',
    'Full memory required by model',
    ['model', 'session_id', 'pid']
)

OLLAMA_MODEL_MEMORY_REQUIRED_KV_BYTES = Gauge(
    'ollama_model_memory_required_kv_bytes',
    'Memory required for KV cache',
    ['model', 'session_id', 'pid']
)

# Session activity
OLLAMA_SESSION_ACTIVE = Gauge(
    'ollama_session_active',
    'Whether the session is currently active (1) or not (0)',
    ['model', 'session_id', 'pid']
)

# System info metrics
OLLAMA_SYSTEM_INFO = Info(
    'ollama_system_info',
    'Information about Ollama system and service',
    ['service_status', 'version', 'port']
)

OLLAMA_GPU_INFO = Info(
    'ollama_gpu_info', 
    'Information about GPU used by Ollama',
    ['gpu_id', 'gpu_name', 'library', 'variant', 'compute', 'driver']
)

OLLAMA_GPU_VRAM_TOTAL_BYTES = Gauge(
    'ollama_gpu_vram_total_bytes',
    'Total VRAM on GPU in bytes',
    ['gpu_id', 'gpu_name']
)

OLLAMA_GPU_VRAM_AVAILABLE_BYTES = Gauge(
    'ollama_gpu_vram_available_bytes', 
    'Available VRAM on GPU in bytes',
    ['gpu_id', 'gpu_name']
)

OLLAMA_SERVICE_START_TIME = Gauge(
    'ollama_service_start_timestamp_seconds',
    'Unix timestamp when Ollama service was started'
)

OLLAMA_SERVICE_RESTARTS_TOTAL = Counter(
    'ollama_service_restarts_total',
    'Total number of Ollama service restarts'
)

# Global state
model_map = {}  # SHA256 -> model_name mapping
active_sessions = {}  # session_id -> session_data mapping
processed_requests = set()  # To avoid duplicate request processing
system_info_collected = False  # Flag to avoid re-collecting system info


class OllamaSession:
    """Represents an Ollama session with all its metadata and state"""
    
    def __init__(self, session_id, start_time, pid=None):
        self.session_id = session_id
        self.start_time = start_time
        self.pid = pid
        self.raw_lines = []
        
        # Model information
        self.model_name = None
        self.sha256 = None
        self.model_sha256 = None
        self.architecture = None
        self.size_label = None
        self.params_label = None
        self.file_size_label = None
        
        # Configuration parameters
        self.ctx_size = None
        self.batch_size = None
        self.gpu_layers = None
        self.threads = None
        self.parallel = None
        self.port = None
        
        # VRAM and GPU info
        self.gpu = None
        self.vram_available = None
        self.vram_required = None
        self.model_path = None
        
        # Offload information
        self.offload_info = {}
        
        # Runtime info
        self.runner_start_time = None
        self.gin_requests = []
        
        # State tracking
        self.runner_block_lines = 0
        self.metadata_block_active = False
        self.sha256_extracted_from_vram_loading = False
        
    def get_model_name(self):
        """Get the best available model name"""
        if self.model_name:
            return self.model_name
        
        sha = self.sha256 or self.model_sha256
        if sha and sha in model_map:
            return model_map[sha]
            
        if sha:
            return f"unknown:{sha[:12]}"
            
        return "unknown"
    
    def is_valid_session(self):
        """Check if session has enough data to be considered valid"""
        return bool(
            self.sha256 or 
            self.model_sha256 or 
            self.model_name or 
            self.offload_info
        )
    
    def to_dict(self):
        """Convert session to dictionary for logging/debugging"""
        return {
            'session_id': self.session_id,
            'pid': self.pid,
            'model_name': self.get_model_name(),
            'sha256': self.sha256 or self.model_sha256,
            'ctx_size': self.ctx_size,
            'gpu_layers': self.gpu_layers,
            'vram_available': self.vram_available,
            'requests_count': len(self.gin_requests)
        }


class SessionTracker:
    """Tracks Ollama sessions in real-time using parser logic"""
    
    def __init__(self):
        self.current_session = None
        self.sessions = {}
        self.param_buffer = []
        self.session_counter = 0
        
    def parse_key_value_string(self, s):
        """Parse key=value pairs from log string"""
        pattern = r'(\w+(?:\.\w+)*)=("\[.*?\]"|\[.*?\]|"[^"]*"|[\w\./\:-]+)'
        matches = re.findall(pattern, s)
        return {key: value.strip('"') for key, value in matches}
    
    def parse_duration(self, duration_str):
        """Convert duration string to seconds"""
        val = float(re.findall(r'[\d\.]+', duration_str)[0])
        if 'µs' in duration_str:
            return val / 1_000_000
        if 'ms' in duration_str:
            return val / 1_000
        if 'm' in duration_str:
            return val * 60
        if 'h' in duration_str:
            return val * 3600
        return val
    
    def process_line(self, line):
        """Process a single log line using parser logic"""
        # Buffer model params lines
        if 'print_info: model params' in line:
            self.param_buffer.append(line)
        
        # Extract PID
        m_pid = re.search(r'ollama\[(\d+)\]', line)
        pid = m_pid.group(1) if m_pid else None
        
        # Check for new session triggers
        is_new_session = (
            'msg="starting llama server"' in line or
            'msg="new model will fit in available VRAM in single GPU, loading"' in line
        )
        
        if is_new_session:
            # Finalize current session
            if self.current_session and self.current_session.is_valid_session():
                self.sessions[self.current_session.session_id] = self.current_session
                self._export_session_metrics(self.current_session)
            
            # Create new session
            self.session_counter += 1
            session_id = f"session_{self.session_counter}_{pid or 'unknown'}"
            start_time = line.split()[0] if line.split() else str(datetime.now())
            
            self.current_session = OllamaSession(session_id, start_time, pid)
            
            # Add buffered params to new session
            if self.param_buffer:
                self.current_session.raw_lines.extend(self.param_buffer)
                self.param_buffer = []
            
            # Set runner block processing
            if 'msg="starting llama server"' in line:
                self.current_session.runner_block_lines = 20
                self.current_session.metadata_block_active = True
        
        if not self.current_session:
            return
        
        # Add line to current session
        self.current_session.raw_lines.append(line)
        
        # Update PID if found
        if pid and not self.current_session.pid:
            self.current_session.pid = pid
        
        # Process runner block (20 lines after starting llama server)
        if self.current_session.runner_block_lines > 0:
            self._process_runner_block_line(line)
            self.current_session.runner_block_lines -= 1
        
        # Process GIN requests
        if '[GIN]' in line:
            logging.info(f"About to process GIN request: {line.strip()}")
        self._process_gin_request(line)
    
    def _process_runner_block_line(self, line):
        """Process lines within the runner block (20 lines after server start)"""
        session = self.current_session
        
        # Extract metadata while block is active
        if session.metadata_block_active:
            if 'general.name' in line:
                m = re.search(r'general\.name\s+str\s*=\s*(.+)', line)
                if m and not session.model_name:
                    session.model_name = m.group(1).strip()
            
            elif 'general.size_label' in line:
                m = re.search(r'general\.size_label\s+str\s*=\s*(.+)', line)
                if m and not session.size_label:
                    session.size_label = m.group(1).strip()
            
            elif 'general.architecture' in line:
                m = re.search(r'general\.architecture\s+str\s*=\s*(.+)', line)
                if m and not session.architecture:
                    session.architecture = m.group(1).strip()
            
            elif 'print_info: model params' in line:
                m = re.search(r'print_info: model params\s*=\s*([\d\.]+\s*[BMKbmkg])', line)
                if m and not session.params_label:
                    session.params_label = m.group(1).strip()
            
            elif 'print_info: file size' in line:
                m = re.search(r'print_info: file size\s*=\s*([\d\.]+\s*[GMK]i?B)', line)
                if not m:
                    m = re.search(r'print_info: file size\s*=\s*([^\(]+)', line)
                if m and not session.file_size_label:
                    session.file_size_label = m.group(1).strip().split('(')[0].strip()
            
            # End metadata block if no relevant content
            elif not any(x in line for x in ['general.name', 'general.size_label', 'general.architecture', 'print_info: model params']):
                session.metadata_block_active = False
        
        # Process VRAM loading
        if 'msg="new model will fit in available VRAM in single GPU, loading"' in line:
            m_sha = re.search(r'sha256-([a-f0-9]{64})', line)
            if m_sha:
                session.model_sha256 = m_sha.group(1)
                session.sha256 = m_sha.group(1)
                session.sha256_extracted_from_vram_loading = True
            
            m_gpu = re.search(r'gpu=([\w\-]+)', line)
            if m_gpu:
                session.gpu = m_gpu.group(1)
            
            m_parallel = re.search(r'parallel=(\d+)', line)
            if m_parallel:
                session.parallel = m_parallel.group(1)
            
            m_avail = re.search(r'available=([0-9]+)', line)
            if m_avail:
                session.vram_available = m_avail.group(1)
            
            m_req = re.search(r'required="([^"]+)"', line)
            if m_req:
                session.vram_required = m_req.group(1)
            
            m_model_path = re.search(r'model=([^\s]+)', line)
            if m_model_path:
                session.model_path = m_model_path.group(1)
        
        # Process offload info
        elif 'msg=offload' in line:
            offload_str = line.split('msg=offload ')[1]
            session.offload_info = self.parse_key_value_string(offload_str)
        
        # Process starting llama server
        elif 'msg="starting llama server"' in line:
            cmd_str = line.split('cmd=')[1].strip().strip('"')
            cmd_parts = cmd_str.split()
            params = {}
            
            for j, part in enumerate(cmd_parts):
                if part.startswith('--') and j + 1 < len(cmd_parts):
                    params[part] = cmd_parts[j+1]
            
            model_path = params.get('--model', '')
            sha = model_path.split('sha256-')[-1] if 'sha256-' in model_path else None
            
            session.sha256 = sha
            session.ctx_size = params.get('--ctx-size', 'N/A')
            session.batch_size = params.get('--batch-size', 'N/A')
            session.gpu_layers = params.get('--n-gpu-layers', 'N/A')
            session.threads = params.get('--threads', 'N/A')
            session.parallel = params.get('--parallel', 'N/A')
            session.port = params.get('--port', 'N/A')
        
        # Process runner start time
        elif 'msg="llama runner started' in line:
            time_str = line.split(' in ')[-1].split(' seconds')[0]
            session.runner_start_time = f"{time_str} s"
    
    def _process_gin_request(self, line):
        """Process GIN request lines"""
        if '[GIN]' not in line:
            return
        
        # Extract PID from line
        m_pid = re.search(r'ollama\[(\d+)\]', line)
        pid = m_pid.group(1) if m_pid else None
        
        # Parse GIN request
        gin_pattern = re.compile(r'^([\d\-:T\+]+)\s+[^:]+: \[GIN\]\s+(\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2}) \| (\d+) \| ([^|]+) \|\s*([^|]+) \|\s*(\w+)\s+"([^"]+)"')
        m = gin_pattern.search(line)
        
        if m:
            journal_time = m.group(1).strip()
            status = m.group(3).strip()
            latency = m.group(4).strip()
            ip = m.group(5).strip()
            method = m.group(6).strip()
            path = m.group(7).strip()
            
            request_data = {
                'journal_time': journal_time,
                'status': status,
                'latency': latency,
                'ip': ip,
                'method': method,
                'path': path,
                'raw_line': line.strip()
            }
            
            # Find the session for this PID
            target_session = None
            
            # First try current session
            if self.current_session and self.current_session.pid == pid:
                target_session = self.current_session
            else:
                # Look for existing session with this PID
                for session in self.sessions.values():
                    if session.pid == pid:
                        target_session = session
                        break
            
            # If no session found, create a minimal session for this PID
            if not target_session and pid:
                self.session_counter += 1
                session_id = f"session_{self.session_counter}_{pid}_gin_only"
                target_session = OllamaSession(session_id, journal_time, pid)
                target_session.model_name = "unknown"  # Will be resolved later if possible
                self.sessions[session_id] = target_session
                logging.info(f"Created minimal session {session_id} for GIN request from PID {pid}")
            
            if target_session:
                target_session.gin_requests.append(request_data)
                self._export_request_metrics(target_session, request_data)
            else:
                logging.warning(f"Could not find or create session for GIN request: {line.strip()}")
    
    def _export_session_metrics(self, session):
        """Export session-related metrics to Prometheus"""
        model_name = session.get_model_name()
        session_id = session.session_id
        pid = session.pid or "unknown"
        
        logging.info(f"Exporting metrics for session {session_id}: {model_name}")
        
        # Session info
        OLLAMA_SESSION_ACTIVE.labels(
            model=model_name,
            session_id=session_id,
            pid=pid
        ).set(1)
        
        # Parse start time to timestamp
        try:
            if session.start_time:
                # Handle different timestamp formats
                start_ts = self._parse_timestamp(session.start_time)
                OLLAMA_SESSION_START_TIME.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(start_ts)
        except Exception as e:
            logging.warning(f"Could not parse start time {session.start_time}: {e}")
        
        # Runner start time
        if session.runner_start_time:
            try:
                runner_time = float(session.runner_start_time.replace(' s', ''))
                OLLAMA_SESSION_RUNNER_START_TIME.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(runner_time)
            except:
                pass
        
        # Configuration metrics
        if session.ctx_size and session.ctx_size != 'N/A':
            try:
                OLLAMA_MODEL_CONTEXT_SIZE.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(int(session.ctx_size))
            except:
                pass
        
        if session.batch_size and session.batch_size != 'N/A':
            try:
                OLLAMA_MODEL_BATCH_SIZE.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(int(session.batch_size))
            except:
                pass
        
        if session.gpu_layers and session.gpu_layers != 'N/A':
            try:
                OLLAMA_MODEL_GPU_LAYERS.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(int(session.gpu_layers))
            except:
                pass
        
        if session.threads and session.threads != 'N/A':
            try:
                OLLAMA_MODEL_THREADS.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(int(session.threads))
            except:
                pass
        
        if session.parallel and session.parallel != 'N/A':
            try:
                OLLAMA_MODEL_PARALLEL.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(int(session.parallel))
            except:
                pass
        
        # VRAM metrics
        gpu = session.gpu or "unknown"
        if session.vram_available:
            try:
                vram_bytes = self._parse_memory_size(session.vram_available)
                OLLAMA_MODEL_VRAM_AVAILABLE_BYTES.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid,
                    gpu=gpu
                ).set(vram_bytes)
            except:
                pass
        
        if session.vram_required:
            try:
                vram_bytes = self._parse_memory_size(session.vram_required)
                OLLAMA_MODEL_VRAM_REQUIRED_BYTES.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid,
                    gpu=gpu
                ).set(vram_bytes)
            except:
                pass
        
        # Offload metrics
        if session.offload_info:
            offload = session.offload_info
            
            if 'layers.model' in offload:
                try:
                    OLLAMA_MODEL_LAYERS_TOTAL.labels(
                        model=model_name,
                        session_id=session_id,
                        pid=pid
                    ).set(int(offload['layers.model']))
                except:
                    pass
            
            if 'layers.offload' in offload:
                try:
                    OLLAMA_MODEL_LAYERS_OFFLOADED.labels(
                        model=model_name,
                        session_id=session_id,
                        pid=pid
                    ).set(int(offload['layers.offload']))
                except:
                    pass
            
            if 'memory.required.full' in offload:
                try:
                    mem_bytes = self._parse_memory_size(offload['memory.required.full'])
                    OLLAMA_MODEL_MEMORY_REQUIRED_FULL_BYTES.labels(
                        model=model_name,
                        session_id=session_id,
                        pid=pid
                    ).set(mem_bytes)
                except:
                    pass
            
            if 'memory.required.kv' in offload:
                try:
                    mem_bytes = self._parse_memory_size(offload['memory.required.kv'])
                    OLLAMA_MODEL_MEMORY_REQUIRED_KV_BYTES.labels(
                        model=model_name,
                        session_id=session_id,
                        pid=pid
                    ).set(mem_bytes)
                except:
                    pass
    
    def _export_request_metrics(self, session, request_data):
        """Export request-related metrics to Prometheus"""
        model_name = session.get_model_name()
        session_id = session.session_id
        
        # Increment request counter
        OLLAMA_REQUESTS_TOTAL.labels(
            model=model_name,
            endpoint=request_data['path'],
            method=request_data['method'],
            status=request_data['status'],
            session_id=session_id
        ).inc()
        
        # Record latency
        try:
            latency_sec = self.parse_duration(request_data['latency'])
            OLLAMA_REQUEST_LATENCY.labels(
                model=model_name,
                endpoint=request_data['path'],
                method=request_data['method'],
                session_id=session_id
            ).observe(latency_sec)
        except:
            pass
    
    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp string to Unix timestamp"""
        # Handle different formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.timestamp()
            except:
                continue
        
        # Fallback to current time
        return time.time()
    
    def _parse_memory_size(self, size_str):
        """Parse memory size string to bytes"""
        if not size_str:
            return 0
        
        # Remove brackets and quotes
        size_str = size_str.strip('[]"')
        
        # Extract number and unit
        match = re.match(r'([\d\.]+)\s*([GMK]i?B?)', size_str, re.IGNORECASE)
        if not match:
            return 0
        
        value = float(match.group(1))
        unit = match.group(2).upper()
        
        multipliers = {
            'B': 1,
            'KB': 1024, 'KIB': 1024,
            'MB': 1024**2, 'MIB': 1024**2,
            'GB': 1024**3, 'GIB': 1024**3
        }
        
        return int(value * multipliers.get(unit, 1))


def build_model_map_from_manifests(models_base_path):
    """Build SHA256 -> model_name mapping from manifest files"""
    global model_map
    temp_model_map = {}
    
    manifests_path = os.path.join(os.path.expanduser(models_base_path), "manifests", "registry.ollama.ai", "library")
    logging.info(f"Building model map from manifests in {manifests_path}...")
    
    try:
        if not os.path.isdir(manifests_path):
            logging.warning(f"Manifests directory not found: {manifests_path}")
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
                                break
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON from manifest: {manifest_file}")
                    except Exception as e:
                        logging.error(f"Error processing manifest {manifest_file}: {e}")
    except Exception as e:
        logging.error(f"Error scanning manifests: {e}")
    
    if not temp_model_map:
        logging.warning("No models found in manifests")
    
    model_map = temp_model_map


def find_models_path_from_journal():
    """Scan journal logs for OLLAMA_MODELS env var to find models path"""
    try:
        logging.info("Scanning journal logs for OLLAMA_MODELS path...")
        env = os.environ.copy()
        env['LANG'] = 'C.UTF-8'
        env['LC_ALL'] = 'C.UTF-8'
        
        process = subprocess.Popen(
            ['journalctl', '-u', JOURNALCTL_UNIT, '--no-pager', '--output=cat', '-r'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, 
            encoding='utf-8', errors='replace', env=env
        )
        
        models_path_re = re.compile(r'OLLAMA_MODELS:(\S+)')
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            if 'OLLAMA_MODELS:' in line:
                m = models_path_re.search(line)
                if m:
                    models_path = m.group(1).strip(']"')
                    logging.info(f"Found Ollama models path: {models_path}")
                    process.terminate()
                    return models_path
        
        process.wait()
        logging.warning("Could not find OLLAMA_MODELS path in journal")
        return None
        
    except FileNotFoundError:
        logging.error("journalctl command not found")
        return None
    except Exception as e:
        logging.error(f"Error searching for models path: {e}")
        return None


def collect_system_info():
    """Collect Ollama system information from startup logs (first approach)"""
    global system_info_collected
    
    if system_info_collected:
        return
    
    try:
        logging.info("Collecting Ollama system information from startup logs...")
        
        # Get startup context (like parser does)
        cmd = "journalctl -u ollama.service --no-pager"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        log_lines = result.stdout.splitlines()
        
        # Find last startup context
        last_start_idx = -1
        for i, line in reversed(list(enumerate(log_lines))):
            if "Started ollama.service" in line or "Starting Ollama Service" in line:
                last_start_idx = i
                break
        
        if last_start_idx == -1:
            logging.warning("No Ollama service startup found in logs")
            return
        
        # Process startup context (10 lines before and after startup)
        start_slice = max(0, last_start_idx - 10)
        end_slice = min(len(log_lines), last_start_idx + 50)  # More lines after to catch GPU info
        context_lines = log_lines[start_slice:end_slice]
        
        # Parse system events and Ollama events
        systemd_events = []
        ollama_events = []
        
        for line in context_lines:
            parts = line.strip().split()
            if not parts or len(parts) < 4:
                continue
                
            timestamp_str = " ".join(parts[:3])
            message = " ".join(parts[3:])
            
            # SYSTEMD EVENTS
            if "systemd" in message:
                if "Started ollama.service" in message:
                    # Parse timestamp and set service start time
                    try:
                        start_ts = parse_systemd_timestamp(timestamp_str)
                        OLLAMA_SERVICE_START_TIME.set(start_ts)
                        logging.info(f"Set Ollama service start time: {timestamp_str}")
                    except Exception as e:
                        logging.warning(f"Could not parse service start time: {e}")
                        
                elif "Scheduled restart job" in message:
                    OLLAMA_SERVICE_RESTARTS_TOTAL.inc()
                    logging.info("Incremented service restart counter")
            
            # OLLAMA EVENTS
            elif "ollama[" in message:
                # API Ready - extract version and port
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
                    
                    # Set system info
                    OLLAMA_SYSTEM_INFO.labels(
                        service_status="running",
                        version=version or "unknown",
                        port=port or "unknown"
                    ).info({
                        'startup_time': timestamp_str,
                        'status': 'active'
                    })
                    
                    logging.info(f"Set Ollama system info - Version: {version}, Port: {port}")
                
                # GPU Found - extract detailed GPU information
                elif 'msg="inference compute"' in message:
                    gpu_details = {}
                    
                    # Parse all key=value pairs from the inference compute message
                    for key, val in re.findall(r'(\w+(?:\.\w+)*|projector\.\w+)=((?:"[^"]*")|(?:\[[^\]]*\])|(?:[^\s]+))', message):
                        gpu_details[key] = val.strip('"')
                    
                    if gpu_details:
                        gpu_id = gpu_details.get('id', 'unknown')
                        gpu_name = gpu_details.get('name', 'unknown')
                        
                        # Set GPU info
                        OLLAMA_GPU_INFO.labels(
                            gpu_id=gpu_id,
                            gpu_name=gpu_name,
                            library=gpu_details.get('library', 'unknown'),
                            variant=gpu_details.get('variant', 'unknown'),
                            compute=gpu_details.get('compute', 'unknown'),
                            driver=gpu_details.get('driver', 'unknown')
                        ).info({
                            'detection_time': timestamp_str,
                            'all_details': str(gpu_details)
                        })
                        
                        # Set VRAM metrics if available
                        if 'total' in gpu_details:
                            try:
                                total_vram = parse_vram_size(gpu_details['total'])
                                OLLAMA_GPU_VRAM_TOTAL_BYTES.labels(
                                    gpu_id=gpu_id,
                                    gpu_name=gpu_name
                                ).set(total_vram)
                                logging.info(f"Set GPU total VRAM: {gpu_details['total']} ({total_vram} bytes)")
                            except Exception as e:
                                logging.warning(f"Could not parse total VRAM: {e}")
                        
                        if 'available' in gpu_details:
                            try:
                                available_vram = parse_vram_size(gpu_details['available'])
                                OLLAMA_GPU_VRAM_AVAILABLE_BYTES.labels(
                                    gpu_id=gpu_id,
                                    gpu_name=gpu_name
                                ).set(available_vram)
                                logging.info(f"Set GPU available VRAM: {gpu_details['available']} ({available_vram} bytes)")
                            except Exception as e:
                                logging.warning(f"Could not parse available VRAM: {e}")
                        
                        logging.info(f"Set GPU info - ID: {gpu_id}, Name: {gpu_name}, Library: {gpu_details.get('library')}")
        
        system_info_collected = True
        logging.info("System information collection completed")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Could not read journal for system info: {e}")
    except Exception as e:
        logging.error(f"Error collecting system info: {e}")


def parse_systemd_timestamp(timestamp_str):
    """Parse systemd timestamp to Unix timestamp"""
    # Handle different systemd timestamp formats
    formats = [
        '%b %d %H:%M:%S',  # Jul 16 12:06:26
        '%Y-%m-%d %H:%M:%S',
        '%m-%d %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            # For formats without year, assume current year
            if '%Y' not in fmt:
                current_year = datetime.now().year
                dt = datetime.strptime(f"{current_year} {timestamp_str}", f"%Y {fmt}")
            else:
                dt = datetime.strptime(timestamp_str, fmt)
            return dt.timestamp()
        except:
            continue
    
    # Fallback to current time
    return time.time()


def parse_vram_size(vram_str):
    """Parse VRAM size string like '11.8 GiB' to bytes"""
    if not vram_str:
        return 0
    
    # Extract number and unit
    match = re.match(r'([\d\.]+)\s*([GMK]i?B?)', vram_str, re.IGNORECASE)
    if not match:
        return 0
    
    value = float(match.group(1))
    unit = match.group(2).upper()
    
    multipliers = {
        'B': 1,
        'KB': 1024, 'KIB': 1024,
        'MB': 1024**2, 'MIB': 1024**2,
        'GB': 1024**3, 'GIB': 1024**3
    }
    
    return int(value * multipliers.get(unit, 1))


def load_recent_sessions(session_tracker, hours_back=2):
    """Load recent sessions for each PID (second approach like parser)"""
    try:
        logging.info(f"Loading recent sessions from last {hours_back} hours...")
        
        # Get recent logs (like parser does with dump_recent_logs_to_file)
        cmd = f"journalctl -u ollama.service --since '{hours_back} hours ago' --no-pager -o short-iso"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        logging.info(f"Processing {len(lines)} recent log lines for session detection...")
        
        # Track active PIDs and their sessions
        active_pids = set()
        pid_sessions = {}  # PID -> latest session data
        
        # First pass: identify all active PIDs and their latest sessions
        for line in lines:
            if not line.strip() or 'ollama[' not in line:
                continue
                
            # Extract PID
            m_pid = re.search(r'ollama\[(\d+)\]', line)
            if not m_pid:
                continue
                
            pid = m_pid.group(1)
            active_pids.add(pid)
            
            # Look for session start events
            if ('msg="starting llama server"' in line or 
                'msg="new model will fit in available VRAM in single GPU, loading"' in line):
                
                timestamp = line.split()[0] if line.split() else str(datetime.now())
                
                # Initialize or update session for this PID
                if pid not in pid_sessions:
                    pid_sessions[pid] = {
                        'pid': pid,
                        'start_time': timestamp,
                        'lines': [],
                        'last_activity': timestamp
                    }
                else:
                    # Update with newer session start
                    pid_sessions[pid]['start_time'] = timestamp
                    pid_sessions[pid]['last_activity'] = timestamp
        
        logging.info(f"Found {len(active_pids)} active PIDs: {list(active_pids)}")
        
        # Second pass: collect session data for each PID
        for pid in active_pids:
            if pid not in pid_sessions:
                continue
                
            logging.info(f"Processing session data for PID {pid}...")
            
            # Collect all lines for this PID
            pid_lines = []
            for line in lines:
                if f'ollama[{pid}]' in line:
                    pid_lines.append(line)
            
            if not pid_lines:
                continue
            
            # Process lines to build session (like parser does)
            session_data = _build_session_from_lines(pid, pid_lines, session_tracker)
            
            if session_data:
                # Create session in tracker
                session_id = f"session_recent_{pid}"
                session = OllamaSession(session_id, session_data['start_time'], pid)
                
                # Copy all extracted data
                for key, value in session_data.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                
                # Add to tracker
                session_tracker.sessions[session_id] = session
                
                # Export metrics for this session
                session_tracker._export_session_metrics(session)
                
                logging.info(f"Created session {session_id} for PID {pid}: {session.get_model_name()}")
        
        logging.info(f"Recent session loading complete. Total sessions: {len(session_tracker.sessions)}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Could not read recent logs: {e}")
    except Exception as e:
        logging.error(f"Error loading recent sessions: {e}")


def _build_session_from_lines(pid, lines, session_tracker):
    """Build session data from lines for a specific PID (like parser logic)"""
    session_data = {
        'pid': pid,
        'start_time': None,
        'model_name': None,
        'sha256': None,
        'model_sha256': None,
        'architecture': None,
        'size_label': None,
        'params_label': None,
        'file_size_label': None,
        'ctx_size': None,
        'batch_size': None,
        'gpu_layers': None,
        'threads': None,
        'parallel': None,
        'port': None,
        'gpu': None,
        'vram_available': None,
        'vram_required': None,
        'model_path': None,
        'offload_info': {},
        'runner_start_time': None,
        'gin_requests': []
    }
    
    runner_block_active = False
    metadata_block_active = False
    runner_block_lines = 0
    
    for i, line in enumerate(lines):
        # Extract timestamp for start_time
        if not session_data['start_time']:
            session_data['start_time'] = line.split()[0] if line.split() else str(datetime.now())
        
        # Check for session start events
        if ('msg="starting llama server"' in line or 
            'msg="new model will fit in available VRAM in single GPU, loading"' in line):
            
            session_data['start_time'] = line.split()[0] if line.split() else str(datetime.now())
            
            if 'msg="starting llama server"' in line:
                runner_block_active = True
                runner_block_lines = 20
                metadata_block_active = True
        
        # Process runner block (20 lines after starting llama server)
        if runner_block_active and runner_block_lines > 0:
            
            # Extract metadata while block is active
            if metadata_block_active:
                if 'general.name' in line:
                    m = re.search(r'general\.name\s+str\s*=\s*(.+)', line)
                    if m and not session_data['model_name']:
                        session_data['model_name'] = m.group(1).strip()
                
                elif 'general.size_label' in line:
                    m = re.search(r'general\.size_label\s+str\s*=\s*(.+)', line)
                    if m and not session_data['size_label']:
                        session_data['size_label'] = m.group(1).strip()
                
                elif 'general.architecture' in line:
                    m = re.search(r'general\.architecture\s+str\s*=\s*(.+)', line)
                    if m and not session_data['architecture']:
                        session_data['architecture'] = m.group(1).strip()
                
                elif 'print_info: model params' in line:
                    m = re.search(r'print_info: model params\s*=\s*([\d\.]+\s*[BMKbmkg])', line)
                    if m and not session_data['params_label']:
                        session_data['params_label'] = m.group(1).strip()
                
                elif 'print_info: file size' in line:
                    m = re.search(r'print_info: file size\s*=\s*([\d\.]+\s*[GMK]i?B)', line)
                    if not m:
                        m = re.search(r'print_info: file size\s*=\s*([^\(]+)', line)
                    if m and not session_data['file_size_label']:
                        session_data['file_size_label'] = m.group(1).strip().split('(')[0].strip()
                
                # End metadata block if no relevant content
                elif not any(x in line for x in ['general.name', 'general.size_label', 'general.architecture', 'print_info: model params']):
                    metadata_block_active = False
            
            # Process VRAM loading
            if 'msg="new model will fit in available VRAM in single GPU, loading"' in line:
                m_sha = re.search(r'sha256-([a-f0-9]{64})', line)
                if m_sha:
                    session_data['model_sha256'] = m_sha.group(1)
                    session_data['sha256'] = m_sha.group(1)
                
                m_gpu = re.search(r'gpu=([\w\-]+)', line)
                if m_gpu:
                    session_data['gpu'] = m_gpu.group(1)
                
                m_parallel = re.search(r'parallel=(\d+)', line)
                if m_parallel:
                    session_data['parallel'] = m_parallel.group(1)
                
                m_avail = re.search(r'available=([0-9]+)', line)
                if m_avail:
                    session_data['vram_available'] = m_avail.group(1)
                
                m_req = re.search(r'required="([^"]+)"', line)
                if m_req:
                    session_data['vram_required'] = m_req.group(1)
                
                m_model_path = re.search(r'model=([^\s]+)', line)
                if m_model_path:
                    session_data['model_path'] = m_model_path.group(1)
            
            # Process offload info
            elif 'msg=offload' in line:
                offload_str = line.split('msg=offload ')[1]
                session_data['offload_info'] = session_tracker.parse_key_value_string(offload_str)
            
            # Process starting llama server
            elif 'msg="starting llama server"' in line:
                cmd_str = line.split('cmd=')[1].strip().strip('"')
                cmd_parts = cmd_str.split()
                params = {}
                
                for j, part in enumerate(cmd_parts):
                    if part.startswith('--') and j + 1 < len(cmd_parts):
                        params[part] = cmd_parts[j+1]
                
                model_path = params.get('--model', '')
                sha = model_path.split('sha256-')[-1] if 'sha256-' in model_path else None
                
                session_data['sha256'] = sha
                session_data['ctx_size'] = params.get('--ctx-size', 'N/A')
                session_data['batch_size'] = params.get('--batch-size', 'N/A')
                session_data['gpu_layers'] = params.get('--n-gpu-layers', 'N/A')
                session_data['threads'] = params.get('--threads', 'N/A')
                session_data['parallel'] = params.get('--parallel', 'N/A')
                session_data['port'] = params.get('--port', 'N/A')
            
            # Process runner start time
            elif 'msg="llama runner started' in line:
                time_str = line.split(' in ')[-1].split(' seconds')[0]
                session_data['runner_start_time'] = f"{time_str} s"
            
            runner_block_lines -= 1
            
            if runner_block_lines <= 0:
                runner_block_active = False
        
        # Collect GIN requests for this session
        if '[GIN]' in line:
            gin_pattern = re.compile(r'^([\d\-:T\+]+)\s+[^:]+: \[GIN\]\s+(\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2}) \| (\d+) \| ([^|]+) \|\s*([^|]+) \|\s*(\w+)\s+"([^"]+)"')
            m = gin_pattern.search(line)
            
            if m:
                request_data = {
                    'journal_time': m.group(1).strip(),
                    'status': m.group(3).strip(),
                    'latency': m.group(4).strip(),
                    'ip': m.group(5).strip(),
                    'method': m.group(6).strip(),
                    'path': m.group(7).strip(),
                    'raw_line': line.strip()
                }
                session_data['gin_requests'].append(request_data)
    
    # Resolve model name using model_map
    sha = session_data['sha256'] or session_data['model_sha256']
    if sha and sha in model_map:
        session_data['model_name'] = model_map[sha]
    
    # Return session data if it has meaningful content
    if (session_data['sha256'] or session_data['model_sha256'] or 
        session_data['model_name'] or session_data['offload_info']):
        return session_data
    
    return None


def follow_ollama_logs(session_tracker):
    """Follow journalctl logs and process them with session tracker"""
    try:
        # First load recent sessions (second approach)
        load_recent_sessions(session_tracker)
        
        logging.info(f"Starting to follow new logs with command: sudo {LOG_COMMAND}")
        process = subprocess.Popen(
            shlex.split(f"sudo {LOG_COMMAND}"), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            # Debug: log every line we receive
            if 'ollama[' in line:
                logging.info(f"Processing new log line: {line.strip()}")
            
            try:
                session_tracker.process_line(line.strip())
            except Exception as e:
                logging.error(f"Error processing line: {e}")
                logging.debug(f"Problematic line: {line.strip()}")
        
    except FileNotFoundError:
        logging.error(f"Command not found: {LOG_COMMAND}")
    except Exception as e:
        logging.error(f"Error following logs: {e}")


def background_model_map_updater(models_path):
    """Background thread to periodically update model map"""
    while True:
        try:
            build_model_map_from_manifests(models_path)
        except Exception as e:
            logging.error(f"Error updating model map: {e}")
        
        time.sleep(MODEL_MAP_UPDATE_INTERVAL)


def cleanup_inactive_sessions(session_tracker):
    """Background thread to clean up metrics for inactive sessions"""
    while True:
        try:
            current_time = time.time()
            inactive_sessions = []
            
            for session_id, session in session_tracker.sessions.items():
                # Check if session is still active (has recent requests or process exists)
                if session.pid:
                    try:
                        # Check if process still exists
                        os.kill(int(session.pid), 0)
                        continue  # Process exists, session is active
                    except (OSError, ValueError):
                        # Process doesn't exist
                        pass
                
                # Mark session as inactive
                inactive_sessions.append(session_id)
            
            # Clean up inactive sessions
            for session_id in inactive_sessions:
                session = session_tracker.sessions[session_id]
                model_name = session.get_model_name()
                pid = session.pid or "unknown"
                
                logging.info(f"Cleaning up inactive session {session_id}: {model_name}")
                
                # Set session as inactive
                OLLAMA_SESSION_ACTIVE.labels(
                    model=model_name,
                    session_id=session_id,
                    pid=pid
                ).set(0)
                
                # Remove from active sessions
                del session_tracker.sessions[session_id]
        
        except Exception as e:
            logging.error(f"Error in session cleanup: {e}")
        
        time.sleep(60)  # Check every minute


def main():
    parser = argparse.ArgumentParser(description='Ollama Log Exporter for Prometheus (New Version)')
    parser.add_argument('--port', type=int, default=9877, help='Port to expose Prometheus metrics')
    args = parser.parse_args()
    
    logging.info("Starting Ollama Log Exporter (New Version)")
    
    # Find models path
    models_path = find_models_path_from_journal()
    if not models_path:
        logging.critical("Failed to determine Ollama models path. Exporter cannot start.")
        return 1
    
    logging.info(f"Using models path: {models_path}")
    
    # Initial model map build
    build_model_map_from_manifests(models_path)
    
    # Collect system information (first approach)
    collect_system_info()
    
    # Start Prometheus server
    start_http_server(args.port)
    logging.info(f"Prometheus exporter server started on port {args.port}")
    
    # Create session tracker
    session_tracker = SessionTracker()
    
    # Start background threads
    model_map_thread = threading.Thread(
        target=background_model_map_updater,
        args=(models_path,),
        daemon=True
    )
    model_map_thread.start()
    logging.info("Started background model map updater")
    
    cleanup_thread = threading.Thread(
        target=cleanup_inactive_sessions,
        args=(session_tracker,),
        daemon=True
    )
    cleanup_thread.start()
    logging.info("Started background session cleanup")
    
    # Main log following loop
    try:
        follow_ollama_logs(session_tracker)
    except KeyboardInterrupt:
        logging.info("Exporter stopped by user")
        return 0
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())