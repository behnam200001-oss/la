import secrets
import numpy as np

class KeyGenerator:
    def __init__(self, args):
        self.args = args
        self.current_key = self._parse_key(args.start_key)
        self.end_key = self._parse_key(args.end_key) if args.end_key else None
        self.step = args.step
        self.range_size = args.range_size
        self.incremental = args.incremental
        self.random = args.random
        
    def _parse_key(self, key_str):
        if key_str.startswith('0x'):
            return int(key_str, 16)
        else:
            return int(key_str)
    
    def _int_to_bytes(self, n, length=32):
        return n.to_bytes(length, 'big')
    
    def generate_incremental(self):
        while True:
            if self.end_key and self.current_key > self.end_key:
                break
                
            yield self._int_to_bytes(self.current_key)
            self.current_key += self.step
    
    def generate_random(self):
        while True:
            yield secrets.randbits(256).to_bytes(32, 'big')
    
    def generate_range(self, start, size):
        for i in range(size):
            yield self._int_to_bytes(start + i * self.step)
    
    def get_generator(self):
        if self.incremental:
            return self.generate_incremental()
        elif self.random:
            return self.generate_random()
        else:
            return self.generate_range(self.current_key, self.range_size)
    
    def generate_batch(self, batch_size):
        batch = []
        generator = self.get_generator()
        
        for _ in range(batch_size):
            try:
                batch.append(next(generator))
            except StopIteration:
                break
        
        return batch