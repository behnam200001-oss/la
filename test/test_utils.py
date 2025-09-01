import unittest
from utils.crypto_utils import *
from utils.address_utils import *

class TestCryptoUtils(unittest.TestCase):
    def test_sha256(self):
        test_data = b"hello world"
        result = sha256(test_data)
        self.assertEqual(len(result), 32)
        self.assertEqual(result.hex(), "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")
    
    def test_ripemd160(self):
        test_data = b"hello world"
        result = ripemd160(test_data)
        self.assertEqual(len(result), 20)
        self.assertEqual(result.hex(), "98c615784ccb5fe5936fbc0cbe9dfdb408d92f0f")
    
    def test_hash160(self):
        test_data = b"hello world"
        result = hash160(test_data)
        self.assertEqual(len(result), 20)
        self.assertEqual(result.hex(), "d7d5ee7824ff93f94c5055b4a5d4c5c5d45e40b7")
    
    def test_keccak256(self):
        test_data = b"hello world"
        result = keccak256(test_data)
        self.assertEqual(len(result), 32)
        self.assertEqual(result.hex(), "47173285a8d7341e5e972fc677286384f802f8ef42a5ec5f03bbfa254cb01fad")
    
    def test_private_key_to_public_key(self):
        private_key = bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000001")
        public_key = private_key_to_public_key(private_key, compressed=True)
        self.assertIsNotNone(public_key)
        self.assertEqual(len(public_key), 33)
        
        public_key_uncompressed = private_key_to_public_key(private_key, compressed=False)
        self.assertIsNotNone(public_key_uncompressed)
        self.assertEqual(len(public_key_uncompressed), 65)
    
    def test_public_key_to_address(self):
        public_key = bytes.fromhex("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
        
        p2pkh_addr = public_key_to_address(public_key, 'p2pkh')
        self.assertEqual(p2pkh_addr, "1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm")
        
        p2sh_addr = public_key_to_address(public_key, 'p2sh')
        self.assertEqual(p2sh_addr, "3JvL6Ymt8MVWiCNHC7o3UJACsUP3q2yB66")
        
        p2wpkh_addr = public_key_to_address(public_key, 'p2wpkh')
        self.assertEqual(p2wpkh_addr, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4")
        
        evm_addr = public_key_to_address(public_key, 'evm')
        self.assertEqual(evm_addr, "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf")

class TestAddressUtils(unittest.TestCase):
    def test_validate_address(self):
        self.assertTrue(validate_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"))
        self.assertTrue(validate_address("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"))
        self.assertTrue(validate_address("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"))
        self.assertTrue(validate_address("0x742d35Cc6634C0532925a3b844Bc454e4438f44e"))
        
        self.assertFalse(validate_address("invalid_address"))
        self.assertFalse(validate_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa0"))
        self.assertFalse(validate_address("0x742d35Cc6634C0532925a3b844Bc454e4438f44"))

if __name__ == '__main__':
    unittest.main()