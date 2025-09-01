from .base_searcher import BaseSearcher
from .cpu_searcher import CPUSearcher
from .key_generator import KeyGenerator
from .address_db import AddressDatabase
from .reporting import ProgressReporter

try:
    from .gpu_searcher import GPUSearcher
    from .hybrid_searcher import HybridSearcher
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
    class GPUSearcher:
        def __init__(self, *args, **kwargs):
            raise ImportError("GPU support is not available")
    
    class HybridSearcher:
        def __init__(self, *args, **kwargs):
            raise ImportError("GPU support is not available")

__all__ = [
    'BaseSearcher',
    'CPUSearcher',
    'GPUSearcher',
    'HybridSearcher',
    'KeyGenerator',
    'AddressDatabase',
    'ProgressReporter',
    'GPU_AVAILABLE'
]