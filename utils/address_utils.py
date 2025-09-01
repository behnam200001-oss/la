import re
from typing import Optional

def validate_address(address: str) -> bool:
    address = address.strip()
    
    if address.startswith('1') or address.startswith('3'):
        if not re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
            return False
        return True
    
    elif address.startswith('bc1') or address.startswith('tb1'):
        if not (len(address) >= 42 and len(address) <= 62):
            return False
        
        if not re.match(r'^(bc1|tb1)[ac-hj-np-z02-9]+$', address.lower()):
            return False
        return True
    
    elif address.startswith('0x') and len(address) == 42:
        if not all(c in '0123456789abcdefABCDEF' for c in address[2:]):
            return False
        return True
    
    elif len(address) == 40:
        if not all(c in '0123456789abcdefABCDEF' for c in address):
            return False
        return True
    
    return False

def normalize_address(address: str) -> Optional[str]:
    if not validate_address(address):
        return None
    
    if (address.startswith('0x') and len(address) == 42) or len(address) == 40:
        clean_addr = address[2:] if address.startswith('0x') else address
        return '0x' + clean_addr.lower()
    
    return address

def detect_address_type(address: str) -> str:
    address = address.strip()
    
    if address.startswith('1'):
        return 'p2pkh'
    elif address.startswith('3'):
        return 'p2sh'
    elif address.startswith('bc1') or address.startswith('tb1'):
        if len(address) <= 42:
            return 'p2wpkh'
        elif len(address) <= 62:
            return 'p2wsh'
        else:
            return 'p2tr'
    elif (address.startswith('0x') and len(address) == 42) or (len(address) == 40 and all(c in '0123456789abcdefABCDEF' for c in address)):
        return 'evm'
    else:
        if re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
            return 'p2pkh'
        elif re.match(r'^[LM][a-km-zA-HJ-NP-Z1-9]{25,34}$', address):
            return 'p2pkh'
        else:
            return 'unknown'

def is_bitcoin_address(address: str) -> bool:
    addr_type = detect_address_type(address)
    return addr_type in ['p2pkh', 'p2sh', 'p2wpkh', 'p2wsh', 'p2tr']

def is_evm_address(address: str) -> bool:
    addr_type = detect_address_type(address)
    return addr_type == 'evm'