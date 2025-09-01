import hashlib
import base58
import bech32
from Crypto.Hash import keccak
from coincurve import PrivateKey
from typing import Optional, Dict

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def ripemd160(data: bytes) -> bytes:
    h = hashlib.new('ripemd160')
    h.update(data)
    return h.digest()

def hash160(data: bytes) -> bytes:
    return ripemd160(sha256(data))

def keccak256(data: bytes) -> bytes:
    k = keccak.new(digest_bits=256)
    k.update(data)
    return k.digest()

def private_key_to_public_key(private_key_bytes: bytes, compressed: bool = True) -> Optional[bytes]:
    try:
        priv_key = PrivateKey(private_key_bytes)
        return priv_key.public_key.format(compressed=compressed)
    except Exception as e:
        print(f"Error converting private key to public key: {e}")
        return None

def public_key_to_address(public_key_bytes: bytes, addr_type: str = 'p2pkh') -> Optional[str]:
    if addr_type == 'p2pkh':
        h160 = hash160(public_key_bytes)
        return base58.b58encode_check(b'\x00' + h160).decode()
    
    elif addr_type == 'p2sh':
        h160 = hash160(public_key_bytes)
        return base58.b58encode_check(b'\x05' + h160).decode()
    
    elif addr_type == 'p2wpkh':
        h160 = hash160(public_key_bytes)
        return bech32.encode('bc', 0, h160)
    
    elif addr_type == 'p2wsh':
        h256 = sha256(public_key_bytes)
        return bech32.encode('bc', 0, h256)
    
    elif addr_type == 'p2tr':
        h256 = sha256(public_key_bytes)
        return bech32.encode('bc', 1, h256)
    
    elif addr_type == 'evm':
        if public_key_bytes[0] in [0x02, 0x03, 0x04]:
            public_key_bytes = public_key_bytes[1:]
        
        if len(public_key_bytes) == 64:
            keccak_hash = keccak256(public_key_bytes)
            address = '0x' + keccak_hash[-20:].hex()
            return address
    
    return None

def private_key_to_wif(private_key_bytes: bytes, compressed: bool = True) -> str:
    extended_key = b'\x80' + private_key_bytes
    
    if compressed:
        extended_key += b'\x01'
    
    first_sha = sha256(extended_key)
    second_sha = sha256(first_sha)
    checksum = second_sha[:4]
    
    return base58.b58encode(extended_key + checksum).decode()

def generate_all_addresses(private_key: bytes) -> Dict[str, Optional[str]]:
    addresses = {}
    
    public_key_compressed = private_key_to_public_key(private_key, compressed=True)
    public_key_uncompressed = private_key_to_public_key(private_key, compressed=False)
    
    if public_key_compressed:
        addresses['p2pkh_compressed'] = public_key_to_address(public_key_compressed, 'p2pkh')
        addresses['p2sh_compressed'] = public_key_to_address(public_key_compressed, 'p2sh')
        addresses['p2wpkh_compressed'] = public_key_to_address(public_key_compressed, 'p2wpkh')
        addresses['p2wsh_compressed'] = public_key_to_address(public_key_compressed, 'p2wsh')
        addresses['p2tr_compressed'] = public_key_to_address(public_key_compressed, 'p2tr')
    
    if public_key_uncompressed:
        addresses['p2pkh_uncompressed'] = public_key_to_address(public_key_uncompressed, 'p2pkh')
        addresses['p2sh_uncompressed'] = public_key_to_address(public_key_uncompressed, 'p2sh')
        addresses['evm'] = public_key_to_address(public_key_uncompressed, 'evm')
    
    return addresses