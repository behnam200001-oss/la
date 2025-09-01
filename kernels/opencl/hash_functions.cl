__kernel void simple_hash(
    __global const uchar* input,
    uint input_len,
    __global uchar* output,
    uint batch_size
) {
    uint idx = get_global_id(0);
    
    if (idx >= batch_size) {
        return;
    }
    
    __global const uchar* data = input + idx * input_len;
    __global uchar* out = output + idx * 32;
    
    for (int i = 0; i < 32; i++) {
        out[i] = data[i % input_len] ^ i;
    }
}

__kernel void hash160_opencl(
    __global const uchar* input,
    uint input_len,
    __global uchar* output,
    uint batch_size
) {
    uint idx = get_global_id(0);
    
    if (idx >= batch_size) {
        return;
    }
    
    __global const uchar* data = input + idx * input_len;
    __global uchar* out = output + idx * 20;
    
    for (int i = 0; i < 20; i++) {
        out[i] = data[i % input_len] ^ i;
    }
}