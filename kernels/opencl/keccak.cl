#define KECCAK_ROUNDS 24

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__constant ulong keccakf_rndc[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant int keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant int keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

void keccakf(ulong st[25]) {
    int i, j, round;
    ulong t, bc[5];

    for (round = 0; round < KECCAK_ROUNDS; round++) {
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        t = st[1];
        for (i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        st[0] ^= keccakf_rndc[round];
    }
}

__kernel void keccak256_hash(
    __global const uchar* input,
    uint input_len,
    __global uchar* output,
    uint batch_size
) {
    uint idx = get_global_id(0);
    
    if (idx >= batch_size) {
        return;
    }
    
    ulong st[25] = {0};
    __global const uchar* data = input + idx * input_len;
    
    for (int i = 0; i < input_len; i++) {
        st[i / 8] |= (ulong)data[i] << (8 * (i % 8));
    }
    
    st[input_len / 8] |= (ulong)0x06 << (8 * (input_len % 8));
    st[16] ^= 0x8000000000000000UL;
    
    keccakf(st);
    
    __global uchar* out = output + idx * 32;
    for (int i = 0; i < 4; i++) {
        ulong v = st[i];
        for (int j = 0; j < 8; j++) {
            out[i * 8 + j] = (v >> (j * 8)) & 0xFF;
        }
    }
}