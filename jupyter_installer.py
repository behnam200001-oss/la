#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import importlib
import ctypes
import urllib.request
import zipfile
import tarfile
import time
import json
from pathlib import Path

class JupyterInstaller:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.arch = platform.machine().lower()
        self.cuda_available = False
        self.opencl_available = False
        self.install_dir = Path(__file__).parent.absolute()
        self.requirements_file = self.install_dir / "requirements.txt"
        
    def check_admin_privileges(self):
        try:
            if self.os_type == "windows":
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except:
            return False
            
    def run_command(self, command, check=True, capture_output=True):
        try:
            result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
            if check and result.returncode != 0:
                print(f"Command error: {command}")
                print(f"Error output: {result.stderr}")
                return False, result.stderr
            return True, result.stdout
        except Exception as e:
            print(f"Exception executing command {command}: {e}")
            return False, str(e)
    
    def detect_hardware(self):
        print("Detecting hardware for Jupyter environment...")
        
        if not self.requirements_file.exists():
            print("requirements.txt file not found!")
            return False
        
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.cuda_available = True
                print("NVIDIA GPU with CUDA support detected")
                
                try:
                    result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode == 0:
                        version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
                        cuda_version = version_line.split('release')[-1].strip().split(',')[0]
                        print(f"CUDA Version: {cuda_version}")
                except:
                    pass
            else:
                print("NVIDIA GPU not found")
                
        except Exception as e:
            print(f"Error checking for NVIDIA GPU: {e}")
            
        try:
            result = subprocess.run(['clinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.opencl_available = True
                print("OpenCL support detected")
            else:
                try:
                    import pyopencl as cl
                    platforms = cl.get_platforms()
                    if platforms:
                        self.opencl_available = True
                        print("OpenCL support detected via pyopencl")
                    else:
                        print("OpenCL not found")
                except ImportError:
                    print("OpenCL not available")
        except Exception as e:
            print(f"Error checking for OpenCL: {e}")
    
    def install_jupyter_dependencies(self):
        print("Installing Jupyter and dependencies...")
        
        jupyter_packages = [
            "jupyter",
            "jupyterlab",
            "ipywidgets",
            "ipykernel",
            "matplotlib",
            "seaborn",
            "pandas",
            "numpy",
            "tqdm",
            "psutil"
        ]
        
        for package in jupyter_packages:
            print(f"Installing {package}...")
            success, output = self.run_command(f"{sys.executable} -m pip install {package}")
            if not success:
                print(f"Failed to install {package}: {output}")
        
        return True
    
    def install_gpu_dependencies(self):
        print("Installing GPU dependencies...")
        
        if self.cuda_available:
            cuda_packages = [
                "pycuda",
                "cupy-cuda11x"
            ]
            
            for package in cuda_packages:
                print(f"Installing {package}...")
                success, output = self.run_command(f"{sys.executable} -m pip install {package}")
                if not success:
                    print(f"Failed to install {package}: {output}")
        
        if self.opencl_available:
            opencl_packages = [
                "pyopencl"
            ]
            
            for package in opencl_packages:
                print(f"Installing {package}...")
                success, output = self.run_command(f"{sys.executable} -m pip install {package}")
                if not success:
                    print(f"Failed to install {package}: {output}")
        
        return True
    
    def install_crypto_dependencies(self):
        print("Installing cryptography dependencies...")
        
        crypto_packages = [
            "pycryptodome",
            "coincurve",
            "base58",
            "bech32",
            "cpuinfo",
            "GPUtil"
        ]
        
        for package in crypto_packages:
            print(f"Installing {package}...")
            success, output = self.run_command(f"{sys.executable} -m pip install {package}")
            if not success:
                print(f"Failed to install {package}: {output}")
        
        return True
    
    def setup_jupyter_kernel(self):
        print("Setting up Jupyter kernel...")
        
        kernel_name = "bitcoin_evm_searcher"
        
        success, output = self.run_command(f"{sys.executable} -m ipykernel install --user --name={kernel_name} --display-name=\"Bitcoin/EVM Searcher\"")
        if not success:
            print(f"Failed to setup Jupyter kernel: {output}")
            return False
        
        print(f"Jupyter kernel '{kernel_name}' installed successfully")
        return True
    
    def create_example_notebook(self):
        print("Creating example notebook...")
        
        notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bitcoin/EVM Address Searcher - Example Notebook\\n",
    "\\n",
    "This notebook demonstrates how to use the Bitcoin/EVM Address Searcher tool.\\n",
    "\\n",
    "## Prerequisites\\n",
    "1. Make sure you have installed all dependencies\\n",
    "2. Prepare a file with target addresses (addresses.txt)\\n",
    "3. Ensure you have sufficient hardware resources\\n",
    "\\n",
    "## Basic Usage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.append('../')\\n",
    "\\n",
    "from main import main\\n",
    "import argparse\\n",
    "\\n",
    "# Create mock args\\n",
    "class Args:\\n",
    "    def __init__(self):\\n",
    "        self.mode = 'auto'\\n",
    "        self.bitcoin = True\\n",
    "        self.evm = True\\n",
    "        self.search_type = 'all'\\n",
    "        self.random = True\\n",
    "        self.incremental = False\\n",
    "        self.start_key = '1'\\n",
    "        self.end_key = None\\n",
    "        self.step = 1\\n",
    "        self.batch_size = 100000\\n",
    "        self.max_memory = None\\n",
    "        self.threads = None\\n",
    "        self.address_file = 'addresses.txt'\\n",
    "        self.output_file = 'found.txt'\\n",
    "        self.report_interval = 5\\n",
    "        self.disable_progress = False\\n",
    "\\n",
    "args = Args()\\n",
    "\\n",
    "# Run the searcher\\n",
    "main()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Advanced Usage\\n",
    "\\n",
    "You can customize the search parameters for your specific needs:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Search only for Bitcoin addresses with incremental keys\\n",
    "args.mode = 'hybrid'\\n",
    "args.bitcoin = True\\n",
    "args.evm = False\\n",
    "args.random = False\\n",
    "args.incremental = True\\n",
    "args.start_key = '0x1'\\n",
    "args.end_key = '0x100000'\\n",
    "args.step = 1\\n",
    "args.batch_size = 50000\\n",
    "\\n",
    "print(\"Starting customized search...\")\\n",
    "main()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Performance Monitoring\\n",
    "\\n",
    "Monitor your search performance with built-in tools:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from utils.performance import PerformanceMonitor\\n",
    "from utils.hardware_detection import HardwareDetector\\n",
    "\\n",
    "# Monitor system performance\\n",
    "monitor = PerformanceMonitor()\\n",
    "detector = HardwareDetector()\\n",
    "hardware_info = detector.detect()\\n",
    "\\n",
    "print(\"Hardware Information:\")\\n",
    "print(f\"CPU: {hardware_info['cpu']['name']} ({hardware_info['cpu']['cores']} cores)\")\\n",
    "print(f\"RAM: {hardware_info['memory']['total']} GB\")\\n",
    "\\n",
    "if hardware_info['gpu']['cuda']['available']:\\n",
    "    for i, gpu in enumerate(hardware_info['gpu']['cuda']['devices']):\\n",
    "        print(f\"GPU {i}: {gpu['name']} (CUDA)\")\\n",
    "elif hardware_info['gpu']['opencl']['available']:\\n",
    "    for i, gpu in enumerate(hardware_info['gpu']['opencl']['devices']):\\n",
    "        print(f\"GPU {i}: {gpu['name']} (OpenCL)\")\\n",
    "else:\\n",
    "    print(\"GPU: None detected\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Bitcoin/EVM Searcher",
   "language": "python",
   "name": "bitcoin_evm_searcher"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_excerpt": false,
   "pygments_lexicon": 1,
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
        
        notebook_path = self.install_dir / "example.ipynb"
        with open(notebook_path, 'w') as f:
            f.write(notebook_content)
        
        print(f"Example notebook created: {notebook_path}")
        return True
    
    def run_tests(self):
        print("Running verification tests...")
        
        tests = [
            ("numpy", "import numpy as np; print('NumPy version:', np.__version__)"),
            ("jupyter", "import jupyter; print('Jupyter version:', jupyter.__version__)"),
            ("pycryptodome", "from Crypto.Hash import keccak; print('Keccak available')"),
        ]
        
        if self.cuda_available:
            tests.append(("pycuda", "import pycuda.driver as cuda; cuda.init(); print('CUDA devices:', [d.name() for d in cuda.Device(count=cuda.Device.count()).query_devices()])"))
        
        if self.opencl_available:
            tests.append(("pyopencl", "import pyopencl as cl; platforms = cl.get_platforms(); print('OpenCL platforms:', [p.name for p in platforms])"))
        
        all_passed = True
        for module, test_code in tests:
            try:
                result = subprocess.run([sys.executable, "-c", test_code], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✓ {module} test passed")
                else:
                    print(f"✗ {module} test failed: {result.stderr}")
                    all_passed = False
            except Exception as e:
                print(f"✗ {module} test failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def create_launch_script(self):
        print("Creating Jupyter launch script...")
        
        script_content = f"""#!/bin/bash
cd "{self.install_dir}"
"{sys.executable}" -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
"""
        
        script_path = self.install_dir / "launch_jupyter.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if self.os_type != "windows":
            os.chmod(script_path, 0o755)
        
        print(f"Jupyter launch script created: {script_path}")
        return True
    
    def run(self):
        print("=" * 60)
        print("Jupyter Installer for Bitcoin/EVM Address Searcher")
        print("=" * 60)
        
        # Detect hardware
        self.detect_hardware()
        
        # Install dependencies
        if not self.install_jupyter_dependencies():
            print("Error installing Jupyter dependencies!")
            return False
        
        if not self.install_gpu_dependencies():
            print("Error installing GPU dependencies!")
            return False
        
        if not self.install_crypto_dependencies():
            print("Error installing cryptography dependencies!")
            return False
        
        # Setup Jupyter kernel
        if not self.setup_jupyter_kernel():
            print("Error setting up Jupyter kernel!")
            return False
        
        # Create example notebook
        if not self.create_example_notebook():
            print("Error creating example notebook!")
            return False
        
        # Create launch script
        if not self.create_launch_script():
            print("Error creating launch script!")
            return False
        
        # Run tests
        if not self.run_tests():
            print("Some tests failed! Installation may not work correctly.")
        
        print("=" * 60)
        print("Jupyter installation completed successfully!")
        print("To start Jupyter Lab, run:")
        print("   ./launch_jupyter.sh (Linux/Mac)")
        print("Or run directly with: python -m jupyter lab")
        print("=" * 60)
        
        return True

if __name__ == "__main__":
    installer = JupyterInstaller()
    success = installer.run()
    sys.exit(0 if success else 1)