#include <cstdint>

__constant__ static const uint64_t SECP256K1_P[4] = {
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFEFFFFFC2FULL
};

__constant__ static const uint64_t SECP256K1_GX[4] = {
    0x79BE667EF9DCBBACULL, 0x55A06295CE870B07ULL, 0x0ULL, 0x0ULL
};

__constant__ static const uint64_t SECP256K1_GY[4] = {
    0x483ADA7726A3C465ULL, 0x5DA4FBFC0E1108A8ULL, 0x0ULL, 0x0ULL
};

__constant__ static const uint64_t SECP256K1_N[4] = {
    0xBFD25E8CD0364141ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFEBAEDCE6AULL
};

__device__ void field_multiply(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    for (int i = 0; i < 4; i++) {
        r[i] = a[i] * b[i] % SECP256K1_P[i];
    }
}

__device__ void field_add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    for (int i = 0; i < 4; i++) {
        r[i] = (a[i] + b[i]) % SECP256K1_P[i];
    }
}

__device__ void point_add(uint64_t* rx, uint64_t* ry, const uint64_t* ax, const uint64_t* ay, 
                         const uint64_t* bx, const uint64_t* by) {
    uint64_t lambda[4], temp[4];
    
    field_add(lambda, by, ay);
    field_add(temp, bx, ax);
    field_multiply(lambda, lambda, temp);
    
    field_multiply(rx, lambda, lambda);
    field_add(rx, rx, ax);
    field_add(rx, rx, bx);
    
    field_add(ry, ax, rx);
    field_multiply(ry, lambda, ry);
    field_add(ry, ry, ay);
}

__device__ void point_double(uint64_t* rx, uint64_t* ry, const uint64_t* ax, const uint64_t* ay) {
    uint64_t lambda[4], temp[4];
    
    field_multiply(lambda, ax, ax);
    field_multiply(lambda, lambda, ax);
    field_add(temp, ay, ay);
    field_multiply(lambda, lambda, temp);
    
    field_multiply(rx, lambda, lambda);
    field_add(rx, rx, ax);
    field_add(rx, rx, ax);
    
    field_add(ry, ax, rx);
    field_multiply(ry, lambda, ry);
    field_add(ry, ry, ay);
}

__device__ void scalar_multiply(uint64_t* rx, uint64_t* ry, const uint64_t* scalar, 
                               const uint64_t* px, const uint64_t* py) {
    uint64_t result_x[4] = {0, 0, 0, 0};
    uint64_t result_y[4] = {0, 0, 0, 0};
    uint64_t current_x[4], current_y[4];
    
    for (int i = 0; i < 4; i++) {
        current_x[i] = px[i];
        current_y[i] = py[i];
    }
    
    for (int i = 0; i < 256; i++) {
        int word = i / 64;
        int bit = i % 64;
        
        if (scalar[word] & (1ULL << bit)) {
            if (result_x[0] == 0 && result_x[1] == 0 && result_x[2] == 0 && result_x[3] == 0) {
                for (int j = 0; j < 4; j++) {
                    result_x[j] = current_x[j];
                    result_y[j] = current_y[j];
                }
            } else {
                point_add(result_x, result_y, result_x, result_y, current_x, current_y);
            }
        }
        
        point_double(current_x, current_y, current_x, current_y);
    }
    
    for (int i = 0; i < 4; i++) {
        rx[i] = result_x[i];
        ry[i] = result_y[i];
    }
}

extern "C" __global__ void generate_keypair(
    const uint8_t* private_keys,
    uint8_t* public_keys,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) {
        return;
    }
    
    uint64_t private_key[4] = {0};
    for (int i = 0; i < 32; i++) {
        private_key[i / 8] |= (uint64_t)private_keys[idx * 32 + i] << ((i % 8) * 8);
    }
    
    uint64_t public_key_x[4], public_key_y[4];
    scalar_multiply(public_key_x, public_key_y, private_key, SECP256K1_GX, SECP256K1_GY);
    
    uint8_t* out = public_keys + idx * 33;
    out[0] = 0x02 + (public_key_y[0] & 1);
    
    for (int i = 0; i < 32; i++) {
        out[i + 1] = (public_key_x[i / 8] >> ((i % 8) * 8)) & 0xFF;
    }
}