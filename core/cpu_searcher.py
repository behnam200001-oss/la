import multiprocessing
import numpy as np
from .base_searcher import BaseSearcher
from utils.crypto_utils import generate_all_addresses

class CPUSearcher(BaseSearcher):
    def __init__(self, args, address_db, hardware_info, resource_manager):
        super().__init__(args, address_db, hardware_info, resource_manager)
        self.setup()
        
    def setup(self):
        print("Setting up CPU...")
        
        self.threads = min(multiprocessing.cpu_count(), 16)
        if self.args.threads:
            self.threads = min(self.threads, self.args.threads)
            
        print(f"Using {self.threads} processing cores")
        
        self.pool = multiprocessing.Pool(self.threads)
    
    def process_batch_cpu(self, private_keys_batch):
        results = []
        
        for private_key in private_keys_batch:
            addresses = generate_all_addresses(private_key)
            
            for addr_type, address in addresses.items():
                if address:
                    hash_bytes = self.address_db._address_to_hash(address, addr_type)
                    if hash_bytes and self.address_db.check_hash(hash_bytes) != -1:
                        results.append((private_key, addr_type, address))
                        break
        
        return results
    
    def search(self):
        found_count = 0
        
        key_generator = self.get_key_generator()
        batch_size = self.resource_manager.get_optimal_batch_size()
        
        try:
            while True:
                private_keys_batch = []
                for _ in range(batch_size):
                    private_keys_batch.append(next(key_generator))
                
                results = self.process_batch_cpu(private_keys_batch)
                
                for private_key, addr_type, address in results:
                    self.save_found_address(private_key, addr_type, address)
                    found_count += 1
                
                if self.reporter:
                    self.reporter.update(batch_size, len(results))
                
                resource_usage = self.resource_manager.monitor_resources()
                if resource_usage['memory_usage'] > 0.9:
                    print("High memory usage, reducing batch size")
                    batch_size = max(1000, batch_size // 2)
                
        except StopIteration:
            pass
        
        return found_count
    
    def cleanup(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()