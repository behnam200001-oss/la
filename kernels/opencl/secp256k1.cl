__constant ulong SECP256K1_P[4] = {
    0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFEFFFFFC2FUL
};

__constant ulong SECP256K1_GX[4] = {
    0x79BE667EF9DCBBACUL, 0x55A06295CE870B07UL, 0x0UL, 0x0UL
};

__constant ulong SECP256K1_GY[4] = {
    0x483ADA7726A3C465UL, 0x5DA4FBFC0E1108A8UL, 0x0UL, 0x0UL
};

__constant ulong SECP256K1_N[4] = {
    0xBFD25E8CD0364141UL, 0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFEBAEDCE6AUL
};

void field_multiply(ulong* r, __constant ulong* a, __constant ulong* b) {
    for (int i = 0; i < 4; i++) {
        r[i] = a[i] * b[i] % SECP256K1_P[i];
    }
}

void field_add(ulong* r, __constant ulong* a, __constant ulong* b) {
    for (int i = 0; i < 4; i++) {
        r[i] = (a[i] + b[i]) % SECP256K1_P[i];
    }
}

void point_add(ulong* rx, ulong* ry, __constant ulong* ax, __constant ulong* ay, 
              __constant ulong* bx, __constant ulong* by) {
    ulong lambda[4], temp[4];
    
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

void point_double(ulong* rx, ulong* ry, __constant ulong* ax, __constant ulong* ay) {
    ulong lambda[4], temp[4];
    
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

void scalar_multiply(ulong* rx, ulong* ry, __constant ulong* scalar, 
                    __constant ulong* px, __constant ulong* py) {
    ulong result_x[4] = {0, 0, 0, 0};
    ulong result_y[4] = {0, 0, 0, 0};
    ulong current_x[4], current_y[4];
    
    for (int i = 0; i < 4; i++) {
        current_x[i] = px[i];
        current_y[i] = py[i];
    }
    
    for (int i = 0; i < 256; i++) {
        int word = i / 64;
        int bit = i % 64;
        
        if (scalar[word] & (1UL << bit)) {
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

__kernel void generate_keypair(
    __global const uchar* private_keys,
    __global uchar* public_keys,
    int batch_size
) {
    int idx = get_global_id(0);
    
    if (idx >= batch_size) {
        return;
    }
    
    ulong private_key[4] = {0};
    for (int i = 0; i < 32; i++) {
        private_key[i / 8] |= (ulong)private_keys[idx * 32 + i] << ((i % 8) * 8);
    }
    
    ulong public_key_x[4], public_key_y[4];
    scalar_multiply(public_key_x, public_key_y, private_key, SECP256K1_GX, SECP256K1_GY);
    
    __global uchar* out = public_keys + idx * 33;
    out[0] = 0x02 + (public_key_y[0] & 1);
    
    for (int i = 0; i < 32; i++) {
        out[i + 1] = (public_key_x[i / 8] >> ((i % 8) * 8)) & 0xFF;
    }
}