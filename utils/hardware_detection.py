import platform
import subprocess
import re
import psutil
import cpuinfo

class HardwareDetector:
    def detect(self):
        hardware_info = {
            'cpu': self._detect_cpu(),
            'memory': self._detect_memory(),
            'gpu': self._detect_gpu(),
            'os': self._detect_os()
        }
        
        return hardware_info
    
    def _detect_cpu(self):
        try:
            info = cpuinfo.get_cpu_info()
            
            return {
                'name': info.get('brand_raw', 'Unknown CPU'),
                'cores': psutil.cpu_count(logical=False) or 1,
                'threads': psutil.cpu_count(logical=True) or 1,
                'arch': info.get('arch', 'Unknown'),
                'bits': info.get('bits', 64),
                'hz': info.get('hz_actual', [0, 0])[0],
                'flags': info.get('flags', [])
            }
        except:
            return {
                'name': 'Unknown CPU',
                'cores': 1,
                'threads': 1,
                'arch': 'Unknown',
                'bits': 64,
                'hz': 0,
                'flags': []
            }
    
    def _detect_memory(self):
        try:
            mem = psutil.virtual_memory()
            
            return {
                'total': round(mem.total / (1024 ** 3), 2),
                'available': round(mem.available / (1024 ** 3), 2),
                'used': round(mem.used / (1024 ** 3), 2),
                'percent': mem.percent
            }
        except:
            return {
                'total': 0,
                'available': 0,
                'used': 0,
                'percent': 0
            }
    
    def _detect_gpu(self):
        gpu_info = {
            'cuda': {'available': False, 'devices': []},
            'opencl': {'available': False, 'devices': []}
        }
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.name and 'nvidia' in gpu.name.lower():
                    gpu_info['cuda']['devices'].append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_free': gpu.memoryFree,
                        'memory_used': gpu.memoryUsed,
                        'driver': gpu.driver,
                        'temperature': gpu.temperature
                    })
            
            if gpu_info['cuda']['devices']:
                gpu_info['cuda']['available'] = True
                
        except ImportError:
            pass
        except Exception:
            pass
        
        try:
            if platform.system() == "Windows":
                try:
                    import ctypes
                    ctypes.windll.OpenCL
                except:
                    raise ImportError("OpenCL not available")
            
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    if device.type == cl.device_type.GPU:
                        gpu_info['opencl']['devices'].append({
                            'name': device.name,
                            'vendor': device.vendor,
                            'memory': device.global_mem_size,
                            'driver': 'OpenCL'
                        })
            
            if gpu_info['opencl']['devices']:
                gpu_info['opencl']['available'] = True
                
        except ImportError:
            pass
        except Exception as e:
            print(f"OpenCL detection warning: {e}")
        
        return gpu_info
    
    def _detect_os(self):
        try:
            return {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            }
        except:
            return {
                'system': 'Unknown',
                'release': 'Unknown',
                'version': 'Unknown',
                'machine': 'Unknown',
                'processor': 'Unknown',
                'python_version': 'Unknown'
            }