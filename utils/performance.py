import psutil
import GPUtil
from typing import Dict, Any
import time

class ResourceManager:
    def __init__(self, hardware_info: Dict[str, Any], args):
        self.hardware_info = hardware_info
        self.args = args
        self.setup_resources()
    
    def setup_resources(self):
        total_memory = self.hardware_info['memory']['total']
        
        if self.args.max_memory:
            self.max_memory = min(self.args.max_memory, total_memory * 0.9)
        else:
            self.max_memory = total_memory * 0.8
        
        if self.args.threads:
            self.cpu_threads = min(self.args.threads, self.hardware_info['cpu']['cores'])
        else:
            self.cpu_threads = self.hardware_info['cpu']['cores']
        
        self.gpu_devices = []
        if self.hardware_info['gpu']['cuda']['available']:
            self.gpu_devices.extend([{
                'type': 'cuda',
                'device': device,
                'memory_limit': device['memory_total'] * 0.8
            } for device in self.hardware_info['gpu']['cuda']['devices']])
        
        if self.hardware_info['gpu']['opencl']['available']:
            self.gpu_devices.extend([{
                'type': 'opencl',
                'device': device,
                'memory_limit': device['global_mem_size'] * 0.8
            } for device in self.hardware_info['gpu']['opencl']['devices']])
    
    def get_optimal_batch_size(self, key_size=32, address_size=20):
        memory_per_key = key_size + address_size + 32
        max_keys = (self.max_memory * 1024**3) / memory_per_key
        
        if self.args.batch_size:
            return min(self.args.batch_size, int(max_keys))
        else:
            return int(max_keys * 0.9)
    
    def monitor_resources(self):
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100
        
        cpu_usage = psutil.cpu_percent() / 100
        
        gpu_usage = {}
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_usage[gpu.id] = {
                    'load': gpu.load,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal
                }
        except:
            pass
        
        return {
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'gpu_usage': gpu_usage
        }


class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.keys_processed = 0
        self.found_count = 0
        
    def update(self, keys_processed_delta=0, found_delta=0):
        self.keys_processed += keys_processed_delta
        self.found_count += found_delta
        
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.append(cpu_percent)
        
        memory_info = psutil.virtual_memory()
        self.memory_usage.append(memory_info.percent)
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.gpu_usage.append({
                    'id': gpu.id,
                    'load': gpu.load,
                    'memory': gpu.memoryUtil
                })
        except:
            pass
        
        if len(self.cpu_usage) > 1000:
            self.cpu_usage = self.cpu_usage[-1000:]
        if len(self.memory_usage) > 1000:
            self.memory_usage = self.memory_usage[-1000:]
        if len(self.gpu_usage) > 1000:
            self.gpu_usage = self.gpu_usage[-1000:]
    
    def get_stats(self):
        elapsed = time.time() - self.start_time
        
        if elapsed > 0:
            keys_per_second = self.keys_processed / elapsed
        else:
            keys_per_second = 0
        
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        
        stats = {
            'elapsed_time': str(elapsed),
            'keys_processed': self.keys_processed,
            'keys_per_second': keys_per_second,
            'found_count': self.found_count,
            'avg_cpu_usage': avg_cpu,
            'avg_memory_usage': avg_memory,
            'gpu_usage': self.gpu_usage[-5:] if self.gpu_usage else []
        }
        
        return stats