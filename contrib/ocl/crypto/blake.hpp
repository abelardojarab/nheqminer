#pragma once

#include <ocl/crypto/detail/blake.hpp>
#include <cassert>
#include <cstring>

namespace ocl {
namespace crypto {

typedef struct  blake2b_state_s
{
    uint64_t    h[8];
    uint64_t    bytes;
}               blake2b_state_t;

inline void zcash_blake2b_init(blake2b_state_t *st, uint8_t hash_len, uint32_t n, uint32_t k) {
	using namespace crypto_detail;

	assert(n > k);
    assert(hash_len <= 64);
    st->h[0] = blake2b_iv[0] ^ (0x01010000 | hash_len);
    for (uint32_t i = 1; i <= 5; i++)
        st->h[i] = blake2b_iv[i];
    st->h[6] = blake2b_iv[6] ^ *(uint64_t *)"ZcashPoW";
    st->h[7] = blake2b_iv[7] ^ (((uint64_t)k << 32) | n);
    st->bytes = 0;
}

inline void zcash_blake2b_update(blake2b_state_t *st, const uint8_t *_msg, uint32_t msg_len, uint32_t is_final) {
	using namespace crypto_detail;

	const uint64_t      *m = (const uint64_t *)_msg;
    uint64_t            v[16];
    assert(msg_len <= 128);
    assert(st->bytes <= UINT64_MAX - msg_len);
    memcpy(v + 0, st->h, 8 * sizeof (*v));
    memcpy(v + 8, blake2b_iv, 8 * sizeof (*v));
    v[12] ^= (st->bytes += msg_len);
    v[14] ^= is_final ? -1 : 0;
    for (uint32_t round = 0; round < blake2b_rounds; round++)
      {
        const uint8_t   *s = blake2b_sigma[round];
        mix(v + 0, v + 4, v + 8,  v + 12, m[s[0]],  m[s[1]]);
        mix(v + 1, v + 5, v + 9,  v + 13, m[s[2]],  m[s[3]]);
        mix(v + 2, v + 6, v + 10, v + 14, m[s[4]],  m[s[5]]);
        mix(v + 3, v + 7, v + 11, v + 15, m[s[6]],  m[s[7]]);
        mix(v + 0, v + 5, v + 10, v + 15, m[s[8]],  m[s[9]]);
        mix(v + 1, v + 6, v + 11, v + 12, m[s[10]], m[s[11]]);
        mix(v + 2, v + 7, v + 8,  v + 13, m[s[12]], m[s[13]]);
        mix(v + 3, v + 4, v + 9,  v + 14, m[s[14]], m[s[15]]);
      }
    for (uint32_t i = 0; i < 8; i++)
        st->h[i] ^= v[i] ^ v[i + 8];	
}

inline void zcash_blake2b_final(blake2b_state_t *st, uint8_t *out, uint8_t outlen) {
	assert(outlen <= 64);
    memcpy(out, st->h, outlen);
}
	
}
}