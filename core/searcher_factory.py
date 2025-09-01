from .cpu_searcher import CPUSearcher

class SearcherFactory:
    @staticmethod
    def create_searcher(args, address_db, hardware_info, resource_manager):
        mode = args.mode.lower()
        
        if mode in ['gpu', 'hybrid'] and not hardware_info['gpu']['cuda']['available'] and not hardware_info['gpu']['opencl']['available']:
            print(f"Warning: {mode} mode requested but no GPU available. Falling back to CPU mode.")
            mode = 'cpu'
        
        if mode == 'auto':
            if hardware_info['gpu']['cuda']['available'] or hardware_info['gpu']['opencl']['available']:
                if hardware_info['cpu']['cores'] > 1:
                    mode = 'hybrid'
                else:
                    mode = 'gpu'
            else:
                mode = 'cpu'
        
        if mode == 'cpu':
            return CPUSearcher(args, address_db, hardware_info, resource_manager)
        elif mode == 'gpu':
            try:
                from .gpu_searcher import GPUSearcher
                return GPUSearcher(args, address_db, hardware_info, resource_manager)
            except ImportError:
                print("GPU searcher not available, falling back to CPU")
                return CPUSearcher(args, address_db, hardware_info, resource_manager)
        elif mode == 'hybrid':
            try:
                from .hybrid_searcher import HybridSearcher
                return HybridSearcher(args, address_db, hardware_info, resource_manager)
            except ImportError:
                print("Hybrid searcher not available, falling back to CPU")
                return CPUSearcher(args, address_db, hardware_info, resource_manager)
        else:
            raise ValueError(f"Unknown search mode: {mode}")