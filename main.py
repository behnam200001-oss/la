#!/usr/bin/env python3
"""
Bitcoin/EVM Address Searcher - Main Entry Point
Optimized for Jupyter Lab environment with GPU support
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def load_config(config_file=None):
    """Load configuration from file or use defaults"""
    if config_file is None:
        config_file = project_root / "config.json"
    
    default_config = {
        "mode": "auto",
        "bitcoin": True,
        "evm": True,
        "search_type": "all",
        "random": True,
        "incremental": False,
        "start_key": "1",
        "end_key": "",
        "step": 1,
        "batch_size": 400000,
        "max_memory": 0,
        "threads": 0,
        "gpu_devices": "all",
        "address_file": "addresses.txt",
        "output_file": "found.txt",
        "report_interval": 5,
        "disable_progress": False,
        "cache_size": 40000,
        "save_interval": 40000,
        "resume": True,
        "verbose": False
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key] = value
                print(f"Configuration loaded from {config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration.")
    else:
        print("Config file not found. Using default configuration.")
    
    return default_config

def setup_environment():
    """Setup environment for optimal performance in Jupyter Lab"""
    print("Setting up environment for Jupyter Lab...")
    
    # Add CUDA paths if available
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path and os.path.exists(cuda_path):
        bin_path = str(Path(cuda_path) / "bin")
        lib_path = str(Path(cuda_path) / "lib" / "x64") if os.name == 'nt' else str(Path(cuda_path) / "lib64")
        
        os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
        else:
            os.environ['LD_LIBRARY_PATH'] = lib_path
        
        print(f"CUDA paths added to environment: {bin_path}, {lib_path}")
    
    # Set environment variables for better performance
    os.environ['PYTHONOPTIMIZE'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    print("Environment setup completed")

def main():
    """Main function for Bitcoin/EVM Address Searcher"""
    parser = argparse.ArgumentParser(description='Bitcoin/EVM Address Searcher')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'cpu', 'gpu', 'hybrid'],
                       help='Search mode: auto, cpu, gpu, or hybrid')
    
    # Address type selection
    parser.add_argument('--bitcoin', action='store_true', default=True,
                       help='Search for Bitcoin addresses')
    parser.add_argument('--no-bitcoin', action='store_false', dest='bitcoin',
                       help='Disable Bitcoin address search')
    parser.add_argument('--evm', action='store_true', default=True,
                       help='Search for EVM addresses')
    parser.add_argument('--no-evm', action='store_false', dest='evm',
                       help='Disable EVM address search')
    parser.add_argument('--search-type', type=str, default='all',
                       choices=['all', 'p2pkh', 'p2sh', 'p2wpkh', 'p2wsh', 'p2tr', 'evm'],
                       help='Specific address type to search for')
    
    # Key generation options
    parser.add_argument('--random', action='store_true', default=True,
                       help='Use random key generation')
    parser.add_argument('--incremental', action='store_true', default=False,
                       help='Use incremental key generation')
    parser.add_argument('--start-key', type=str, default='1',
                       help='Start key for incremental generation')
    parser.add_argument('--end-key', type=str, default='',
                       help='End key for incremental generation')
    parser.add_argument('--step', type=int, default=1,
                       help='Step size for incremental generation')
    
    # Performance options
    parser.add_argument('--batch-size', type=int, default=100000,
                       help='Batch size for processing')
    parser.add_argument('--max-memory', type=float, default=0,
                       help='Maximum memory usage in GB (0 for auto)')
    parser.add_argument('--threads', type=int, default=0,
                       help='Number of threads (0 for auto)')
    parser.add_argument('--gpu-devices', type=str, default='all',
                       help='GPU devices to use (comma-separated or "all")')
    
    # File options
    parser.add_argument('--address-file', type=str, default='addresses.txt',
                       help='File containing addresses to search for')
    parser.add_argument('--output-file', type=str, default='found.txt',
                       help='File to save found addresses')
    
    # UI options
    parser.add_argument('--report-interval', type=int, default=5,
                       help='Progress report interval in seconds')
    parser.add_argument('--disable-progress', action='store_true', default=False,
                       help='Disable progress reporting')
    
    # Advanced options
    parser.add_argument('--cache-size', type=int, default=1000,
                       help='Cache size for address database')
    parser.add_argument('--save-interval', type=int, default=10000,
                       help='Save interval for found addresses')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from last position if possible')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                       help='Do not resume from last position')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose output')
    
    # Config file option
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file to use')
    
    args = parser.parse_args()
    
    # Load configuration from file if specified
    config = load_config(args.config)
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    
    # Convert config back to args namespace
    args = argparse.Namespace(**config)
    
    # Setup environment
    setup_environment()
    
    print("=" * 60)
    print("Bitcoin/EVM Address Searcher")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Search types: Bitcoin={args.bitcoin}, EVM={args.evm}")
    print(f"Key generation: Random={args.random}, Incremental={args.incremental}")
    if args.incremental:
        print(f"Key range: {args.start_key} to {args.end_key} (step: {args.step})")
    print(f"Batch size: {args.batch_size}")
    print(f"Address file: {args.address_file}")
    print(f"Output file: {args.output_file}")
    print("=" * 60)
    
    # Import modules after environment setup
    try:
        from utils.hardware_detection import HardwareDetector
        from utils.performance import ResourceManager
        from core.address_db import AddressDatabase
        from core.searcher_factory import SearcherFactory
        
        # Detect hardware
        print("Detecting hardware...")
        hardware_detector = HardwareDetector()
        hardware_info = hardware_detector.detect()
        
        # Print hardware info
        print(f"CPU: {hardware_info['cpu']['name']} ({hardware_info['cpu']['cores']} cores)")
        print(f"RAM: {hardware_info['memory']['total']} GB")
        
        if hardware_info['gpu']['cuda']['available']:
            for i, gpu in enumerate(hardware_info['gpu']['cuda']['devices']):
                print(f"GPU {i}: {gpu['name']} (CUDA)")
        elif hardware_info['gpu']['opencl']['available']:
            for i, gpu in enumerate(hardware_info['gpu']['opencl']['devices']):
                print(f"GPU {i}: {gpu['name']} (OpenCL)")
        else:
            print("GPU: None detected")
        
        # Load address database
        print(f"Loading addresses from {args.address_file}...")
        address_db = AddressDatabase(
            args.address_file, 
            search_types=args.search_type,
            bitcoin_search=args.bitcoin,
            evm_search=args.evm
        )
        
        # Create resource manager
        resource_manager = ResourceManager(hardware_info, args)
        
        # Create searcher
        searcher_factory = SearcherFactory()
        searcher = searcher_factory.create_searcher(
            args, address_db, hardware_info, resource_manager
        )
        
        # Setup progress reporting
        reporter = searcher.setup_reporting(args.report_interval, args.disable_progress)
        
        # Start search
        print("Starting search...")
        found_count = searcher.search()
        
        # Cleanup
        searcher.cleanup()
        
        print("=" * 60)
        print(f"Search completed. Found {found_count} addresses.")
        print(f"Results saved to {args.output_file}")
        print("=" * 60)
        
        return found_count
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Exit with appropriate code
    sys.exit(main())