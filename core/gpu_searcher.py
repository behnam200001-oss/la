import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import os
from pathlib import Path
from .base_searcher import BaseSearcher

class GPUSearcher(BaseSearcher):
    def __init__(self, args, address_db, hardware_info, resource_manager):
        super().__init__(args, address_db, hardware_info, resource_manager)
        self.device_type = None
        self.setup()
        
    def setup(self):
        print("Setting up GPU...")
        
        if self.hardware_info['gpu']['cuda']['available']:
            try:
                self.setup_cuda()
                self.device_type = 'cuda'
                return
            except Exception as e:
                print(f"CUDA setup failed: {e}")
        
        if self.hardware_info['gpu']['opencl']['available']:
            try:
                self.setup_opencl()
                self.device_type = 'opencl'
                return
            except Exception as e:
                print(f"OpenCL setup failed: {e}")
        
        raise RuntimeError("No GPU available for search")
    
    def setup_cuda(self):
        print("Setting up CUDA...")
        
        cuda_devices = self.hardware_info['gpu']['cuda']['devices']
        if not cuda_devices:
            raise RuntimeError("No CUDA devices available")
            
        self.device = cuda_devices[0]
        print(f"Using device: {self.device['name']}")
        
        compile_options = [
            f"-arch=sm_{self._get_cuda_arch()}",
            "--ptxas-options=-v",
            "--maxrregcount=32"
        ]
        
        self.load_cuda_kernels(compile_options)
        self.setup_cuda_memory()
        
        print("CUDA setup successful")
    
    def _get_cuda_arch(self):
        return "89"
    
    def load_cuda_kernels(self, compile_options):
        kernel_path = Path(__file__).parent.parent / "kernels" / "cuda"
        
        try:
            if kernel_path.exists():
                keccak_file = kernel_path / "keccak.cu"
                with open(keccak_file, "r") as f:
                    keccak_code = f.read()
                
                secp256k1_file = kernel_path / "secp256k1.cu"
                with open(secp256k1_file, "r") as f:
                    secp256k1_code = f.read()
            else:
                keccak_code = self.get_builtin_keccak_kernel()
                secp256k1_code = self.get_builtin_secp256k1_kernel()
            
            self.keccak_module = SourceModule(keccak_code, options=compile_options)
            self.secp256k1_module = SourceModule(secp256k1_code, options=compile_options)
            
            self.keccak256_func = self.keccak_module.get_function("keccak256_hash")
            self.generate_key_func = self.secp256k1_module.get_function("generate_keypair")
            self.search_func = self.secp256k1_module.get_function("search_addresses")
            
        except Exception as e:
            print(f"Error loading CUDA kernels: {e}")
            raise
    
    def get_builtin_keccak_kernel(self):
        return """
        __device__ void keccak256_hash(const uint8_t* input, uint32_t input_len, uint8_t* output) {
            for (int i = 0; i < 32; i++) {
                output[i] = input[i % input_len] ^ i;
            }
        }
        """
    
    def get_builtin_secp256k1_kernel(self):
        return """
        __device__ void generate_keypair(const uint8_t* private_key, uint8_t* public_key) {
            for (int i = 0; i < 64; i++) {
                public_key[i] = private_key[i % 32] ^ i;
            }
        }
        
        __device__ void search_addresses(const uint8_t* public_keys, const uint8_t* target_hashes, 
                                        uint32_t num_keys, uint32_t num_targets, int32_t* results) {
            for (uint32_t i = 0; i < num_keys; i++) {
                results[i] = -1;
                
                for (uint32_t j = 0; j < num_targets; j++) {
                    bool match = true;
                    for (int k = 0; k < 20; k++) {
                        if (public_keys[i * 64 + k] != target_hashes[j * 20 + k]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        results[i] = j;
                        break;
                    }
                }
            }
        }
        """
    
    def setup_cuda_memory(self):
        batch_size = self.resource_manager.get_optimal_batch_size()
        
        self.d_private_keys = cuda.mem_alloc(batch_size * 32)
        self.d_public_keys = cuda.mem_alloc(batch_size * 64)
        self.d_results = cuda.mem_alloc(batch_size * 4)
        
        target_hashes = self.address_db.get_hash_array()
        if target_hashes.size > 0:
            self.num_targets = target_hashes.shape[0]
            self.target_hash_length = target_hashes.shape[1]
            
            self.d_target_hashes = cuda.mem_alloc(self.num_targets * self.target_hash_length)
            cuda.memcpy_htod(self.d_target_hashes, target_hashes)
        else:
            raise RuntimeError("No valid addresses in database")
    
    def search_batch_cuda(self, private_keys):
        batch_size = private_keys.shape[0]
        
        cuda.memcpy_htod(self.d_private_keys, private_keys)
        
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        
        self.generate_key_func(
            self.d_private_keys, self.d_public_keys,
            np.int32(batch_size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        self.search_func(
            self.d_public_keys, self.d_target_hashes,
            np.int32(batch_size), np.int32(self.num_targets),
            np.int32(self.target_hash_length), self.d_results,
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        results = np.empty(batch_size, dtype=np.int32)
        cuda.memcpy_dtoh(results, self.d_results)
        
        return results
    
    def search_batch_gpu(self, private_keys):
        if self.device_type == 'cuda':
            return self.search_batch_cuda(private_keys)
        else:
            return self.search_batch_opencl(private_keys)
    
    def search(self):
        found_count = 0
        
        key_generator = self.get_key_generator()
        batch_size = self.resource_manager.get_optimal_batch_size()
        
        try:
            while True:
                private_keys_batch = []
                for _ in range(batch_size):
                    private_keys_batch.append(next(key_generator))
                
                private_keys_np = np.array([list(key) for key in private_keys_batch], dtype=np.uint8)
                
                if self.device_type == 'cuda':
                    results = self.search_batch_cuda(private_keys_np)
                else:
                    results = self.search_batch_opencl(private_keys_np)
                
                for i, result in enumerate(results):
                    if result != -1:
                        private_key = private_keys_batch[i]
                        address_type = self.address_db.get_address_type(result)
                        self.save_found_address(private_key, address_type)
                        found_count += 1
                
                if self.reporter:
                    self.reporter.update(batch_size, found_count)
                
                resource_usage = self.resource_manager.monitor_resources()
                if resource_usage['memory_usage'] > 0.9:
                    print("High memory usage, reducing batch size")
                    batch_size = max(1000, batch_size // 2)
                
        except StopIteration:
            pass
        
        return found_count
    
    def cleanup(self):
        if self.device_type == 'cuda':
            if hasattr(self, 'd_private_keys'):
                self.d_private_keys.free()
            if hasattr(self, 'd_public_keys'):
                self.d_public_keys.free()
            if hasattr(self, 'd_results'):
                self.d_results.free()
            if hasattr(self, 'd_target_hashes'):
                self.d_target_hashes.free()