import unittest
import tempfile
import os
from pathlib import Path
from core.address_db import AddressDatabase

class TestAddressDatabase(unittest.TestCase):
    def setUp(self):
        self.test_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "bc1pqyqszqgpqyqszqgpqyqszqgpqyqszqgpqyqszqgpqyqszqgpqyqsyjer9e"
        ]
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for addr in self.test_addresses:
            self.temp_file.write(addr + '\n')
        self.temp_file.close()
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_load_addresses(self):
        db = AddressDatabase(self.temp_file.name)
        self.assertEqual(db.get_count(), 4)
    
    def test_address_detection(self):
        db = AddressDatabase(self.temp_file.name)
        
        addr_types = [db.address_types[i] for i in range(db.get_count())]
        expected_types = ['p2pkh', 'p2sh', 'p2wpkh', 'evm']
        self.assertEqual(addr_types, expected_types)
    
    def test_hash_generation(self):
        db = AddressDatabase(self.temp_file.name)
        
        for i in range(db.get_count()):
            hash_bytes = db.hashes[i]
            self.assertIn(len(hash_bytes), [20, 32])
    
    def test_check_hash(self):
        db = AddressDatabase(self.temp_file.name)
        
        test_hash = db.hashes[0]
        result = db.check_hash(test_hash)
        self.assertEqual(result, 0)
        
        non_existent = b'\x00' * 20
        result = db.check_hash(non_existent)
        self.assertEqual(result, -1)

if __name__ == '__main__':
    unittest.main()