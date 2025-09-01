import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    if config_file is None:
        config_file = get_project_root() / "config.json"
    
    default_config = {
        "general": {
            "mode": "auto",
            "bitcoin": true,
            "evm": true,
            "search_type": "all",
            "random": true,
            "incremental": false,
            "start_key": "1",
            "end_key": "",
            "step": 1,
            "batch_size": 100000,
            "address_file": "addresses.txt",
            "output_file": "found.txt",
            "report_interval": 5,
            "disable_progress": false
        },
        "performance": {
            "max_memory": 0,
            "threads": 0,
            "gpu_devices": "all"
        },
        "advanced": {
            "cache_size": 1000,
            "save_interval": 10000,
            "resume": true,
            "verbose": false
        }
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
                for section in default_config:
                    if section in user_config:
                        for key in default_config[section]:
                            if key in user_config[section]:
                                default_config[section][key] = user_config[section][key]
                
                return default_config
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration.")
    
    return default_config

def save_config(config: Dict[str, Any], config_file: Optional[str] = None) -> bool:
    if config_file is None:
        config_file = get_project_root() / "config.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config file: {e}")
        return False

def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        return f"{seconds/3600:.2f} hours"

def format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024
        i += 1
    
    return f"{size:.2f} {size_names[i]}"

def progress_bar(iteration: int, total: int, length: int = 50, prefix: str = '', suffix: str = '') -> str:
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'

def estimate_time_remaining(start_time: float, current: int, total: int) -> str:
    if current == 0:
        return "Unknown"
    
    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = (total - current) / rate
    
    return format_time(remaining)

def create_directory(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def is_running_in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False