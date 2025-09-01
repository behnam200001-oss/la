#!/usr/bin/env python3
import subprocess
import sys
import os
import platform
import re
import json
import time
import requests
from pathlib import Path

class GPUSetupTester:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.arch = platform.machine().lower()
        self.cuda_available = False
        self.cuda_version = None
        self.gpu_info = {}
        self.project_root = Path.cwd()
        
    def run_command(self, command, check=True):
        """Execute system commands"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if check and result.returncode != 0:
                print(f"Command error: {command}")
                print(f"Error output: {result.stderr}")
                return False, result.stderr
            return True, result.stdout
        except Exception as e:
            print(f"Exception executing command {command}: {e}")
            return False, str(e)
    
    def check_nvidia_driver(self):
        """Check NVIDIA drivers"""
        print("Checking NVIDIA drivers...")
        
        if self.os_type == "windows":
            try:
                import winreg
                reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
                key = winreg.OpenKey(reg, r"SOFTWARE\NVIDIA Corporation\Global\NVTweak")
                driver_version = winreg.QueryValueEx(key, "Version")[0]
                print(f"NVIDIA driver found: version {driver_version}")
                return True
            except:
                print("NVIDIA driver not found")
                return False
        else:
            success, output = self.run_command("nvidia-smi", check=False)
            if success and "NVIDIA-SMI" in output:
                match = re.search(r"NVIDIA-SMI (\d+\.\d+)", output)
                if match:
                    print(f"NVIDIA driver found: version {match.group(1)}")
                return True
            else:
                print("NVIDIA driver not found")
                return False
    
    def check_cuda_toolkit(self):
        """Check if CUDA Toolkit is installed"""
        print("Checking CUDA Toolkit...")
        
        cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
        if cuda_path and os.path.exists(cuda_path):
            print(f"CUDA Toolkit found at: {cuda_path}")
            
            version_file = Path(cuda_path) / "version.txt"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    self.cuda_version = f.read().strip()
                    print(f"CUDA version: {self.cuda_version}")
            return True
        
        success, output = self.run_command("nvcc --version", check=False)
        if success and "release" in output:
            match = re.search(r"release (\d+\.\d+)", output)
            if match:
                self.cuda_version = match.group(1)
                print(f"CUDA Toolkit found: version {self.cuda_version}")
                return True
        
        print("CUDA Toolkit not found")
        return False
    
    def setup_environment_variables(self):
        """Setup environment variables for CUDA"""
        print("Setting up environment variables...")
        
        cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
        if not cuda_path:
            if self.os_type == "windows":
                possible_paths = [
                    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*",
                    "C:\\CUDA\\v*"
                ]
            else:
                possible_paths = [
                    "/usr/local/cuda*",
                    "/opt/cuda*"
                ]
            
            for pattern in possible_paths:
                paths = list(Path('/').glob(pattern[3:])) if pattern.startswith('/') else list(Path(pattern[:2]).glob(pattern[3:]))
                if paths:
                    cuda_path = str(paths[0])
                    break
        
        if cuda_path:
            bin_path = str(Path(cuda_path) / "bin")
            lib_path = str(Path(cuda_path) / "lib" / "x64") if self.os_type == "windows" else str(Path(cuda_path) / "lib64")
            
            os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
            os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
            os.environ['CUDA_PATH'] = cuda_path
            
            print(f"CUDA paths added to PATH: {bin_path}, {lib_path}")
            return True
        
        print("CUDA path not found")
        return False
    
    def detect_gpu_specs(self):
        """Detect GPU specifications"""
        print("Detecting GPU specifications...")
        
        try:
            import pycuda.driver as cuda
            cuda.init()
            
            device_count = cuda.Device.count()
            print(f"CUDA devices found: {device_count}")
            
            for i in range(device_count):
                device = cuda.Device(i)
                attrs = device.get_attributes()
                
                name = device.name()
                memory = device.total_memory() / (1024**3)
                cuda_cores = attrs.get(cuda.device_attribute.MULTIPROCESSOR_COUNT, 0) * \
                            {2: 128, 3: 192, 5: 128, 6: 64, 7: 64, 8: 64}.get(
                                attrs.get(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR, 0), 64
                            )
                
                self.gpu_info[i] = {
                    'name': name,
                    'memory_gb': round(memory, 2),
                    'cuda_cores': cuda_cores
                }
                
                print(f"GPU {i}: {name}")
                print(f"  Memory: {memory:.2f} GB")
                print(f"  CUDA Cores: {cuda_cores}")
                
            return True
        except Exception as e:
            print(f"Error detecting GPU specs: {e}")
            return False
    
    def configure_project_settings(self):
        """Configure project settings based on GPU specs"""
        print("Configuring project settings for optimal GPU performance...")
        
        if not self.gpu_info:
            print("No GPU info available, using default settings")
            return False
        
        gpu_id = 0
        gpu = self.gpu_info[gpu_id]
        
        config = {
            'gpu_config': {
                'device_id': gpu_id,
                'device_name': gpu['name'],
                'memory_gb': gpu['memory_gb'],
                'cuda_cores': gpu['cuda_cores'],
                'batch_size': self.calculate_optimal_batch_size(gpu['memory_gb']),
                'threads_per_block': 256,
                'blocks_per_grid': 64
            },
            'performance_settings': {
                'max_memory_usage': gpu['memory_gb'] * 0.8,
                'checkpoint_interval': 100000,
                'reporting_interval': 5
            }
        }
        
        config_path = self.project_root / "gpu_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Project configuration saved to: {config_path}")
        
        self.update_cuda_kernels(config['gpu_config'])
        return True
    
    def calculate_optimal_batch_size(self, memory_gb):
        """Calculate optimal batch size based on GPU memory"""
        memory_bytes = memory_gb * 1024**3
        batch_size = int(memory_bytes / (32 * 1024))
        return min(batch_size, 1000000)
    
    def update_cuda_kernels(self, gpu_config):
        """Update CUDA kernel configuration"""
        print("Updating CUDA kernel configuration...")
        
        kernels_path = self.project_root / "kernels" / "cuda"
        if not kernels_path.exists():
            print("Kernels directory not found, creating basic structure")
            kernels_path.mkdir(parents=True, exist_ok=True)
            
            secp256k1_file = kernels_path / "secp256k1.cu"
            secp256k1_content = """
#include <cstdint>

#define THREADS_PER_BLOCK %(threads_per_block)d
#define BLOCKS_PER_GRID %(blocks_per_grid)d

__global__ void generate_keypair(
    const uint8_t* private_keys,
    uint8_t* public_keys,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) {
        return;
    }
    
    // Simplified key generation for demonstration
    const uint8_t* private_key = private_keys + idx * 32;
    uint8_t* public_key = public_keys + idx * 64;
    
    for (int i = 0; i < 32; i++) {
        public_key[i] = private_key[i] ^ i;
        public_key[i + 32] = private_key[i] ^ (i + 32);
    }
}
""" % gpu_config
            
            with open(secp256k1_file, 'w') as f:
                f.write(secp256k1_content)
            
            print(f"CUDA kernel template created at: {secp256k1_file}")
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("Installing required dependencies...")
        
        dependencies = [
            "pycuda",
            "numpy",
            "coincurve",
            "base58",
            "bech32",
            "pycryptodome",
            "requests",
            "tqdm",
            "psutil",
            "GPUtil"
        ]
        
        for package in dependencies:
            print(f"Installing {package}...")
            success, output = self.run_command(f"{sys.executable} -m pip install {package}")
            if success:
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}: {output}")
        
        return True
    
    def run_real_test(self):
        """Run a real test with Bitcoin addresses"""
        print("Running real Bitcoin address test...")
        
        test_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Known Bitcoin address
            "1BitcoinEaterAddressDontSendf59kuE",  # Another known address
            "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",  # P2SH address
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"  # Bech32 address
        ]
        
        test_address_file = self.project_root / "test_addresses.txt"
        with open(test_address_file, 'w') as f:
            for address in test_addresses:
                f.write(f"{address}\n")
        
        print(f"Test addresses saved to: {test_address_file}")
        
        test_script = self.project_root / "run_test.py"
        test_script_content = '''
import os
import sys
import time
from pathlib import Path

project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

from core.address_db import AddressDatabase
from core.key_generator import KeyGenerator
from core.cpu_searcher import CPUSearcher
from utils.hardware_detection import HardwareDetector

def run_test():
    print("Starting Bitcoin address search test...")
    
    address_file = project_root / "test_addresses.txt"
    output_file = project_root / "test_results.txt"
    
    if not address_file.exists():
        print("Test address file not found")
        return False
    
    address_db = AddressDatabase(str(address_file), search_types='all', 
                               bitcoin_search=True, evm_search=False)
    
    print(f"Loaded {address_db.get_count()} addresses for testing")
    
    hardware_info = HardwareDetector().detect()
    
    class Args:
        def __init__(self):
            self.mode = 'cpu'
            self.bitcoin = True
            self.evm = False
            self.search_type = 'all'
            self.random = False
            self.incremental = True
            self.start_key = '0x1'
            self.end_key = '0x1000'
            self.step = 1
            self.batch_size = 10000
            self.max_memory = None
            self.threads = None
            self.address_file = str(address_file)
            self.output_file = str(output_file)
            self.report_interval = 5
            self.disable_progress = False
    
    args = Args()
    
    searcher = CPUSearcher(args, address_db, hardware_info, None)
    
    start_time = time.time()
    found_count = searcher.search()
    end_time = time.time()
    
    print(f"Test completed in {end_time - start_time:.2f} seconds")
    print(f"Found {found_count} matching addresses")
    
    if found_count > 0:
        print("Check test_results.txt for details")
        return True
    else:
        print("No addresses found in the test range")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
'''
        
        with open(test_script, 'w') as f:
            f.write(test_script_content)
        
        print(f"Test script created at: {test_script}")
        
        print("Executing test script...")
        success, output = self.run_command(f"{sys.executable} {test_script}")
        
        if success:
            print("Test completed successfully!")
            print(output)
        else:
            print("Test failed!")
            print(output)
        
        return success
    
    def run_comprehensive_check(self):
        """Run comprehensive environment check"""
        print("=" * 60)
        print("Running Comprehensive GPU Environment Check")
        print("=" * 60)
        
        checks = [
            ("NVIDIA Driver Check", self.check_nvidia_driver),
            ("CUDA Toolkit Check", self.check_cuda_toolkit),
            ("Environment Setup", self.setup_environment_variables),
            ("GPU Detection", self.detect_gpu_specs),
            ("Dependency Installation", self.install_dependencies),
            ("Project Configuration", self.configure_project_settings)
        ]
        
        results = []
        for check_name, check_func in checks:
            print(f"\n{check_name}:")
            try:
                success = check_func()
                status = "PASS" if success else "FAIL"
                print(f"Result: {status}")
                results.append((check_name, success))
            except Exception as e:
                print(f"Error: {e}")
                results.append((check_name, False))
        
        print("\n" + "=" * 60)
        print("Check Summary:")
        print("=" * 60)
        
        all_passed = True
        for check_name, success in results:
            status = "PASS" if success else "FAIL"
            print(f"{check_name}: {status}")
            if not success:
                all_passed = False
        
        return all_passed

def main():
    """Main function"""
    print("Bitcoin/EVM Address Searcher - GPU Setup and Test")
    print("This script will configure your GPU environment and run a test")
    
    setup = GPUSetupTester()
    
    if setup.run_comprehensive_check():
        print("\n" + "=" * 60)
        print("All checks passed! Running real test...")
        print("=" * 60)
        
        setup.run_real_test()
    else:
        print("\nSome checks failed. Please fix the issues and run again.")
        sys.exit(1)

if __name__ == "__main__":
    main()