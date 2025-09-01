import os
import time
from abc import ABC, abstractmethod
from .key_generator import KeyGenerator
from utils.crypto_utils import private_key_to_wif

class BaseSearcher(ABC):
    def __init__(self, args, address_db, hardware_info, resource_manager):
        self.args = args
        self.address_db = address_db
        self.hardware_info = hardware_info
        self.resource_manager = resource_manager
        self.reporter = None
        self.found_count = 0
        
    def get_key_generator(self):
        return KeyGenerator(self.args).get_generator()
    
    def save_found_address(self, private_key, addr_type, address=None):
        if address is None:
            from utils.crypto_utils import (
                private_key_to_public_key,
                public_key_to_address
            )
            
            public_key = private_key_to_public_key(private_key, compressed=True)
            address = public_key_to_address(public_key, addr_type)
        
        with open(self.args.output_file, 'a') as f:
            wif = private_key_to_wif(private_key)
            f.write(f"Address: {address}\n")
            f.write(f"Type: {addr_type}\n")
            f.write(f"Private Key: {private_key.hex()}\n")
            f.write(f"WIF: {wif}\n")
            f.write("-" * 50 + "\n")
        
        print(f"Found: {address} (type: {addr_type})")
        self.found_count += 1
    
    def setup_reporting(self, interval, disable_progress):
        if not disable_progress:
            from .reporting import ProgressReporter
            self.reporter = ProgressReporter(interval)
            self.reporter.start()
        return self.reporter
    
    @abstractmethod
    def setup(self):
        pass
    
    @abstractmethod
    def search(self):
        pass
    
    def cleanup(self):
        if self.reporter:
            self.reporter.stop()