#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path

def optimize_jupyter_lab():
    """Optimize Jupyter Lab for GPU operations"""
    print("Optimizing Jupyter Lab for GPU operations...")
    
    jupyter_config_dir = Path.home() / ".jupyter"
    jupyter_config_dir.mkdir(exist_ok=True)
    
    jupyter_config = jupyter_config_dir / "jupyter_notebook_config.py"
    
    config_content = """
# Jupyter Lab GPU Optimization
import os

# Enable GPU acceleration
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False

# Increase memory limits
c.NotebookApp.memory_limit = 1024 * 1024 * 1024  # 1GB

# Set kernel options
c.KernelManager.autorestart = True
c.MappingKernelManager.cull_idle_timeout = 3600
c.MappingKernelManager.cull_interval = 300

# Enable large data support
c.NotebookApp.allow_origin = '*'
c.NotebookApp.disable_check_xsrf = True
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' *"
    }
}

# Add CUDA paths to environment
c.Spawner.env_keep += ['PATH', 'CUDA_PATH', 'LD_LIBRARY_PATH']
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
"""
    
    with open(jupyter_config, 'w') as f:
        f.write(config_content)
    
    print(f"Jupyter Lab configuration saved to: {jupyter_config}")
    
    # Create startup script for Jupyter Lab
    startup_script = """
#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda

# Start Jupyter Lab with optimized settings
jupyter lab --NotebookApp.allow_remote_access=True \
            --NotebookApp.ip='0.0.0.0' \
            --NotebookApp.open_browser=False \
            --NotebookApp.memory_limit=1073741824 \
            --NotebookApp.allow_origin='*' \
            --NotebookApp.disable_check_xsrf=True
"""
    
    startup_file = Path("start_jupyter_gpu.sh")
    with open(startup_file, 'w') as f:
        f.write(startup_script)
    
    # Make executable
    startup_file.chmod(0o755)
    
    print(f"Jupyter Lab startup script created: {startup_file}")
    print("To start Jupyter Lab with GPU optimization, run: ./start_jupyter_gpu.sh")
    
    return True

if __name__ == "__main__":
    optimize_jupyter_lab()