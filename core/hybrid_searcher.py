import multiprocessing
import threading
import time
import numpy as np
from queue import Queue, Empty
from .base_searcher import BaseSearcher

class HybridSearcher(BaseSearcher):
    def __init__(self, args, address_db, hardware_info, resource_manager):
        super().__init__(args, address_db, hardware_info, resource_manager)
        self.cpu_searcher = None
        self.gpu_searcher = None
        self.setup()
        
    def setup(self):
        print("Setting up hybrid CPU/GPU search...")
        
        from .cpu_searcher import CPUSearcher
        self.cpu_searcher = CPUSearcher(self.args, self.address_db, self.hardware_info, self.resource_manager)
        
        try:
            from .gpu_searcher import GPUSearcher
            self.gpu_searcher = GPUSearcher(self.args, self.address_db, self.hardware_info, self.resource_manager)
            print("GPU searcher initialized successfully")
        except Exception as e:
            print(f"GPU searcher initialization failed: {e}. Falling back to CPU only.")
            self.gpu_searcher = None
        
        self.key_queue = Queue()
        self.result_queue = Queue()
        
        self.cpu_batch_size = self.resource_manager.get_optimal_batch_size() // 2
        if self.gpu_searcher:
            self.gpu_batch_size = self.resource_manager.get_optimal_batch_size() // 2
        else:
            self.gpu_batch_size = 0
        
        print(f"CPU batch size: {self.cpu_batch_size}")
        print(f"GPU batch size: {self.gpu_batch_size}")
    
    def cpu_worker(self):
        while True:
            try:
                keys = self.key_queue.get(timeout=1)
                if keys is None:
                    break
                
                results = self.cpu_searcher.process_batch_cpu(keys)
                self.result_queue.put(('cpu', results))
                
            except Empty:
                if self.stop_event.is_set():
                    break
            except Exception as e:
                print(f"CPU worker error: {e}")
    
    def gpu_worker(self):
        if not self.gpu_searcher:
            return
            
        while True:
            try:
                keys = self.key_queue.get(timeout=1)
                if keys is None:
                    break
                
                private_keys_np = np.array([list(key) for key in keys], dtype=np.uint8)
                
                if hasattr(self.gpu_searcher, 'search_batch_gpu'):
                    result_indices = self.gpu_searcher.search_batch_gpu(private_keys_np)
                elif hasattr(self.gpu_searcher, 'search_batch_cuda'):
                    result_indices = self.gpu_searcher.search_batch_cuda(private_keys_np)
                else:
                    print("GPU searcher doesn't have search method")
                    break
                
                results = []
                for i, result_index in enumerate(result_indices):
                    if result_index != -1:
                        private_key = keys[i]
                        address_type = self.gpu_searcher.address_db.get_address_type(result_index)
                        results.append((private_key, address_type, None))
                
                self.result_queue.put(('gpu', results))
                
            except Empty:
                if self.stop_event.is_set():
                    break
            except Exception as e:
                print(f"GPU worker error: {e}")
    
    def result_processor(self):
        found_count = 0
        
        while True:
            try:
                source, results = self.result_queue.get(timeout=1)
                
                for private_key, addr_type, address in results:
                    self.save_found_address(private_key, addr_type, address)
                    found_count += 1
                
                if self.reporter:
                    self.reporter.update(len(results), len(results))
                
            except Empty:
                if self.stop_event.is_set() and self.key_queue.empty() and self.result_queue.empty():
                    break
            except Exception as e:
                print(f"Result processor error: {e}")
        
        return found_count
    
    def search(self):
        print("Starting hybrid search...")
        
        self.stop_event = threading.Event()
        
        cpu_thread = threading.Thread(target=self.cpu_worker)
        gpu_thread = threading.Thread(target=self.gpu_worker) if self.gpu_searcher else None
        result_thread = threading.Thread(target=self.result_processor)
        
        cpu_thread.start()
        if gpu_thread:
            gpu_thread.start()
        result_thread.start()
        
        key_generator = self.get_key_generator()
        found_count = 0
        
        try:
            while True:
                cpu_keys = []
                for _ in range(self.cpu_batch_size):
                    cpu_keys.append(next(key_generator))
                
                gpu_keys = []
                if self.gpu_searcher:
                    for _ in range(self.gpu_batch_size):
                        gpu_keys.append(next(key_generator))
                
                self.key_queue.put(cpu_keys)
                if gpu_keys:
                    self.key_queue.put(gpu_keys)
                
                try:
                    while not self.result_queue.empty():
                        source, results = self.result_queue.get_nowait()
                        for private_key, addr_type, address in results:
                            self.save_found_address(private_key, addr_type, address)
                            found_count += 1
                except Empty:
                    pass
                
                resource_usage = self.resource_manager.monitor_resources()
                if resource_usage['memory_usage'] > 0.9:
                    print("High memory usage, reducing batch size")
                    self.cpu_batch_size = max(1000, self.cpu_batch_size // 2)
                    if self.gpu_searcher:
                        self.gpu_batch_size = max(1000, self.gpu_batch_size // 2)
                
        except StopIteration:
            pass
        except KeyboardInterrupt:
            print("Search interrupted by user")
        finally:
            self.stop_event.set()
            self.key_queue.put(None)
            if self.gpu_searcher:
                self.key_queue.put(None)
            
            cpu_thread.join()
            if gpu_thread:
                gpu_thread.join()
            
            while not self.result_queue.empty():
                try:
                    source, results = self.result_queue.get_nowait()
                    for private_key, addr_type, address in results:
                        self.save_found_address(private_key, addr_type, address)
                        found_count += 1
                except Empty:
                    break
            
            result_thread.join()
        
        return found_count
    
    def cleanup(self):
        if self.cpu_searcher:
            self.cpu_searcher.cleanup()
        if self.gpu_searcher:
            self.gpu_searcher.cleanup()