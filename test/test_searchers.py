import unittest
import tempfile
import os
from pathlib import Path
from core.cpu_searcher import CPUSearcher
from core.address_db import AddressDatabase
from utils.hardware_detection import HardwareDetector

class TestSearchers(unittest.TestCase):
    def setUp(self):
        self.test_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        ]
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for addr in self.test_addresses:
            self.temp_file.write(addr + '\n')
        self.temp_file.close()
        
        class Args:
            def __init__(self):
                self.random = True
                self.incremental = False
                self.bitcoin = True
                self.evm = False
                self.batch_size = 1000
                self.output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt').name
        
        self.args = Args()
        self.hardware_info = HardwareDetector().detect()
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
        if os.path.exists(self.args.output_file):
            os.unlink(self.args.output_file)
    
    def test_cpu_searcher_initialization(self):
        db = AddressDatabase(self.temp_file.name)
        searcher = CPUSearcher(self.args, db, self.hardware_info, None)
        self.assertIsNotNone(searcher)
    
    def test_key_generation(self):
        db = AddressDatabase(self.temp_file.name)
        searcher = CPUSearcher(self.args, db, self.hardware_info, None)
        
        generator = searcher.get_key_generator()
        key = next(generator)
        self.assertEqual(len(key), 32)
    
    def test_address_generation(self):
        db = AddressDatabase(self.temp_file.name)
        searcher = CPUSearcher(self.args, db, self.hardware_info, None)
        
        test_private_key = bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000001")
        addresses = searcher.generate_addresses(test_private_key)
        
        self.assertIn('p2pkh', addresses)
        self.assertIn('p2sh', addresses)
        self.assertIn('p2wpkh', addresses)

if __name__ == '__main__':
    unittest.main()