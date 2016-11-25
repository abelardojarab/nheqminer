#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sa_cuda_context.hpp"

#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <functional>
#include <vector>
#include <iostream>
#include <stdint.h>


//*blake header */

typedef struct  blake2b_state_s
{
	uint64_t    h[8];
	uint64_t    bytes;
}               blake2b_state_t;

void zcash_blake2b_init(blake2b_state_t *st, uint8_t hash_len, uint32_t n, uint32_t k);
void zcash_blake2b_update(blake2b_state_t *st, const uint8_t *_msg,
	uint32_t msg_len, uint32_t is_final);
void zcash_blake2b_final(blake2b_state_t *st, uint8_t *out, uint8_t outlen);

/* blake.cpp **/

//static const uint32_t   blake2b_block_len = 128;
static const uint32_t   blake2b_rounds = 12;
static const uint64_t   blake2b_iv[8] =
{
	0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};
static const uint8_t    blake2b_sigma[12][16] =
{
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{ 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{ 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{ 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{ 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
};

/*
** Init the state according to Zcash parameters.
*/
void zcash_blake2b_init(blake2b_state_t *st, uint8_t hash_len,
	uint32_t n, uint32_t k)
{
	st->h[0] = blake2b_iv[0] ^ (0x01010000 | hash_len);
	for (uint32_t i = 1; i <= 5; i++)
		st->h[i] = blake2b_iv[i];
	st->h[6] = blake2b_iv[6] ^ *(uint64_t *)"ZcashPoW";
	st->h[7] = blake2b_iv[7] ^ (((uint64_t)k << 32) | n);
	st->bytes = 0;
}

static uint64_t rotr64(uint64_t a, uint8_t bits)
{
	return (a >> bits) | (a << (64 - bits));
}

static inline void mix64(uint64_t *va, uint64_t *vb, uint64_t *vc, uint64_t *vd,
	uint64_t x, uint64_t y)
{
	*va = (*va + *vb + x);
	*vd = rotr64(*vd ^ *va, 32);
	*vc = (*vc + *vd);
	*vb = rotr64(*vb ^ *vc, 24);
	*va = (*va + *vb + y);
	*vd = rotr64(*vd ^ *va, 16);
	*vc = (*vc + *vd);
	*vb = rotr64(*vb ^ *vc, 63);
}

/*
** Process either a full message block or the final partial block.
** Note that v[13] is not XOR'd because st->bytes is assumed to never overflow.
**
** _msg         pointer to message (must be zero-padded to 128 bytes if final block)
** msg_len      must be 128 (<= 128 allowed only for final partial block)
** is_final     indicate if this is the final block
*/
void zcash_blake2b_update(blake2b_state_t *st, const uint8_t *_msg,
	uint32_t msg_len, uint32_t is_final)
{
	const uint64_t      *m = (const uint64_t *)_msg;
	uint64_t            v[16];
	memcpy(v + 0, st->h, 8 * sizeof(*v));
	memcpy(v + 8, blake2b_iv, 8 * sizeof(*v));
	v[12] ^= (st->bytes += msg_len);
	v[14] ^= is_final ? -1 : 0;
	for (uint32_t round = 0; round < blake2b_rounds; round++)
	{
		const uint8_t   *s = blake2b_sigma[round];
		mix64(v + 0, v + 4, v + 8, v + 12, m[s[0]], m[s[1]]);
		mix64(v + 1, v + 5, v + 9, v + 13, m[s[2]], m[s[3]]);
		mix64(v + 2, v + 6, v + 10, v + 14, m[s[4]], m[s[5]]);
		mix64(v + 3, v + 7, v + 11, v + 15, m[s[6]], m[s[7]]);
		mix64(v + 0, v + 5, v + 10, v + 15, m[s[8]], m[s[9]]);
		mix64(v + 1, v + 6, v + 11, v + 12, m[s[10]], m[s[11]]);
		mix64(v + 2, v + 7, v + 8, v + 13, m[s[12]], m[s[13]]);
		mix64(v + 3, v + 4, v + 9, v + 14, m[s[14]], m[s[15]]);
	}
	for (uint32_t i = 0; i < 8; i++)
		st->h[i] ^= v[i] ^ v[i + 8];
}

void zcash_blake2b_final(blake2b_state_t *st, uint8_t *out, uint8_t outlen)
{
	memcpy(out, st->h, outlen);
}

/* end of blake cpp*/

#define PARAM_N 200
#define PARAM_K 9
#define PREFIX (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS (1 << PREFIX);
// Approximate log base 2 of number of elements in hash tables
#define APX_NR_ELMS_LOG (PREFIX + 1)

#define ZCASH_BLOCK_HEADER_LEN		140
#define NR_ROWS_LOG 20
#define OVERHEAD 6
#define NR_ROWS (1 << NR_ROWS_LOG)
#define NR_SLOTS ((1 << (APX_NR_ELMS_LOG - NR_ROWS_LOG)) * OVERHEAD)
// Length of 1 element (slot) in bytes
#define SLOT_LEN 32
#define ZCASH_HASH_LEN  50
#define COLL_DATA_SIZE_PER_TH		(NR_SLOTS * 5)
#define MAX_SOLS 10

#define xi_offset_for_round(round)	(8 + ((round) / 2) * 4)


constexpr uint32_t c_NR_SLOTS = NR_SLOTS;
constexpr uint32_t c_ROW_LEN = c_NR_SLOTS * SLOT_LEN;
//constexpr uint32_t c_NR_ROWS = NR_ROWS;

#define HT_SIZE				(NR_ROWS * NR_SLOTS * SLOT_LEN)

#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))


typedef struct __align__(64) sols_s
{
	uint32_t nr;
	uint32_t likely_invalids;
	uint8_t valid[MAX_SOLS];
	uint32_t values[MAX_SOLS][(1 << PARAM_K)];
} sols_t;


__device__ uint32_t rowCounter0[1 << NR_ROWS_LOG];
__device__ uint32_t rowCounter1[1 << NR_ROWS_LOG];
__device__ uint32_t* rowCounters[2] = { rowCounter0 , rowCounter1 };
__device__ blake2b_state_t blake;
__device__ sols_t sols;


__constant__ uint64_t blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};


__global__ void kernel_init_0()
{
	rowCounter0[(blockDim.x * blockIdx.x) + threadIdx.x] = 0;
}

__global__ void kernel_init_1()
{
	rowCounter1[(blockDim.x * blockIdx.x) + threadIdx.x] = 0;
}


typedef uint64_t ulong;
typedef uint32_t uint;
typedef uint8_t uchar;

__device__ uint ht_store(uint round, char *ht, uint i,
	ulong xi0, ulong xi1, ulong xi2, ulong xi3, uint *rowCounters)
{
	uint    row;
	char       *p;
	uint                cnt;
#if NR_ROWS_LOG == 16
	if (!(round & 1))
		row = (xi0 & 0xffff);
	else
		// if we have in hex: "ab cd ef..." (little endian xi0) then this
		// formula computes the row as 0xdebc. it skips the 'a' nibble as it
		// is part of the PREFIX. The Xi will be stored starting with "ef...";
		// 'e' will be considered padding and 'f' is part of the current PREFIX
		row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
	if (!(round & 1))
		row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
	else
		row = ((xi0 & 0xc0000) >> 2) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
	if (!(round & 1))
		row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
	else
		row = ((xi0 & 0xe0000) >> 1) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
	if (!(round & 1))
		row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
	else
		row = ((xi0 & 0xf0000) >> 0) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
	cnt = atomicAdd(&rowCounters[row], 1);
	if (cnt >= c_NR_SLOTS) {
		// avoid overflows
		atomicSub(&rowCounters[row], 1);
		return 1;
	}
	p = ht + row * c_ROW_LEN;
	p += cnt * SLOT_LEN + xi_offset_for_round(round);
	// store "i" (always 4 bytes before Xi)
	*(uint *)(p - 4) = i;
	if (round == 0 || round == 1)
	{
		// store 24 bytes
		*(ulong *)(p + 0) = xi0;
		*(ulong *)(p + 8) = xi1;
		*(ulong *)(p + 16) = xi2;
	}
	else if (round == 2)
	{
		// store 20 bytes
		*(uint *)(p + 0) = xi0;
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
	}
	else if (round == 3)
	{
		// store 16 bytes
		*(uint *)(p + 0) = xi0;
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(uint *)(p + 12) = (xi1 >> 32);
	}
	else if (round == 4)
	{
		// store 16 bytes
		*(ulong *)(p + 0) = xi0;
		*(ulong *)(p + 8) = xi1;
	}
	else if (round == 5)
	{
		// store 12 bytes
		*(ulong *)(p + 0) = xi0;
		*(uint *)(p + 8) = xi1;
	}
	else if (round == 6 || round == 7)
	{
		// store 8 bytes
		*(uint *)(p + 0) = xi0;
		*(uint *)(p + 4) = (xi0 >> 32);
	}
	else if (round == 8)
	{
		// store 4 bytes
		*(uint *)(p + 0) = xi0;
	}
	return 0;
}

#define rotate(a, bits) ((a) << (bits)) | ((a) >> (64 - (bits)))

#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
vd = rotate((vd ^ va), (ulong)64 - 32); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (ulong)64 - 24); \
va = (va + vb + y); \
vd = rotate((vd ^ va), (ulong)64 - 16); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (ulong)64 - 63);

__global__
void kernel_round0(char *ht, uint32_t inputs_per_thread)
{
	typedef uint64_t ulong;

	uint32_t                tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t				v[16];
	//uint32_t                inputs_per_thread = c_NR_ROWS / (gridDim.x * blockDim.x);
	uint32_t                input = tid * inputs_per_thread;
	uint32_t                input_end = (tid + 1) * inputs_per_thread;
	uint32_t                dropped = 0;

	while (input < input_end) {
		// shift "i" to occupy the high 32 bits of the second ulong word in the
		// message block
		ulong word1 = (ulong)input << 32;
		// init vector v
		v[0] = blake.h[0];
		v[1] = blake.h[1];
		v[2] = blake.h[2];
		v[3] = blake.h[3];
		v[4] = blake.h[4];
		v[5] = blake.h[5];
		v[6] = blake.h[6];
		v[7] = blake.h[7];
		v[8] = blake_iv[0];
		v[9] = blake_iv[1];
		v[10] = blake_iv[2];
		v[11] = blake_iv[3];
		v[12] = blake_iv[4];
		v[13] = blake_iv[5];
		v[14] = blake_iv[6];
		v[15] = blake_iv[7];
		// mix in length of data
		v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
		// last block
		v[14] ^= (ulong)-1;

		// round 1
		mix(v[0], v[4], v[8], v[12], 0, word1);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 2
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], word1, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 3
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, word1);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 4
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, word1);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 5
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, word1);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 6
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], word1, 0);
		// round 7
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], word1, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 8
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, word1);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 9
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], word1, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 10
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], word1, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 11
		mix(v[0], v[4], v[8], v[12], 0, word1);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], 0, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);
		// round 12
		mix(v[0], v[4], v[8], v[12], 0, 0);
		mix(v[1], v[5], v[9], v[13], 0, 0);
		mix(v[2], v[6], v[10], v[14], 0, 0);
		mix(v[3], v[7], v[11], v[15], 0, 0);
		mix(v[0], v[5], v[10], v[15], word1, 0);
		mix(v[1], v[6], v[11], v[12], 0, 0);
		mix(v[2], v[7], v[8], v[13], 0, 0);
		mix(v[3], v[4], v[9], v[14], 0, 0);

		// compress v into the blake state; this produces the 50-byte hash
		// (two Xi values)
		ulong h[7];
		h[0] = blake.h[0] ^ v[0] ^ v[8];
		h[1] = blake.h[1] ^ v[1] ^ v[9];
		h[2] = blake.h[2] ^ v[2] ^ v[10];
		h[3] = blake.h[3] ^ v[3] ^ v[11];
		h[4] = blake.h[4] ^ v[4] ^ v[12];
		h[5] = blake.h[5] ^ v[5] ^ v[13];
		h[6] = (blake.h[6] ^ v[6] ^ v[14]) & 0xffff;

		// store the two Xi values in the hash table
#if ZCASH_HASH_LEN == 50
		dropped += ht_store(0, ht, input * 2,
			h[0],
			h[1],
			h[2],
			h[3], rowCounter0);
		dropped += ht_store(0, ht, input * 2 + 1,
			(h[3] >> 8) | (h[4] << (64 - 8)),
			(h[4] >> 8) | (h[5] << (64 - 8)),
			(h[5] >> 8) | (h[6] << (64 - 8)),
			(h[6] >> 8), rowCounter0);
#else
#error "unsupported ZCASH_HASH_LEN"
#endif

		input++;
	}
#ifdef ENABLE_DEBUG
	debug[tid * 2] = 0;
	debug[tid * 2 + 1] = dropped;
#endif
}

#if NR_ROWS_LOG <= 16 && NR_SLOTS <= (1 << 8)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 16) | ((slot1 & 0xff) << 8) | (slot0 & 0xff))
#define DECODE_ROW(REF)   (REF >> 16)
#define DECODE_SLOT1(REF) ((REF >> 8) & 0xff)
#define DECODE_SLOT0(REF) (REF & 0xff)

#elif NR_ROWS_LOG == 18 && NR_SLOTS <= (1 << 7)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 14) | ((slot1 & 0x7f) << 7) | (slot0 & 0x7f))
#define DECODE_ROW(REF)   (REF >> 14)
#define DECODE_SLOT1(REF) ((REF >> 7) & 0x7f)
#define DECODE_SLOT0(REF) (REF & 0x7f)

#elif NR_ROWS_LOG == 19 && NR_SLOTS <= (1 << 6)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 13) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f)) /* 1 spare bit */
#define DECODE_ROW(REF)   (REF >> 13)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#elif NR_ROWS_LOG == 20 && NR_SLOTS <= (1 << 6)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 12) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f))
#define DECODE_ROW(REF)   (REF >> 12)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#else
#error "unsupported NR_ROWS_LOG"
#endif

/*
** Access a half-aligned long, that is a long aligned on a 4-byte boundary.
*/
__device__ ulong half_aligned_long(ulong *p, uint offset)
{
	return
		(((ulong)*(uint *)((char *)p + offset + 0)) << 0) |
		(((ulong)*(uint *)((char *)p + offset + 4)) << 32);
}

/*
** Access a well-aligned int.
*/
__device__ uint well_aligned_int(ulong *_p, uint offset)
{
	char *p = (char *)_p;
	return *(uint *)(p + offset);
}

/*
** XOR a pair of Xi values computed at "round - 1" and store the result in the
** hash table being built for "round". Note that when building the table for
** even rounds we need to skip 1 padding byte present in the "round - 1" table
** (the "0xAB" byte mentioned in the description at the top of this file.) But
** also note we can't load data directly past this byte because this would
** cause an unaligned memory access which is undefined per the OpenCL spec.
**
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
__device__ uint xor_and_store(uint round, char *ht_dst, uint row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters)
{
	ulong xi0, xi1, xi2;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
	// Note: for NR_ROWS_LOG == 20, for odd rounds, we could optimize by not
	// storing the byte containing bits from the previous PREFIX block for
	if (round == 1 || round == 2)
	{
		// xor 24 bytes
		xi0 = *(a++) ^ *(b++);
		xi1 = *(a++) ^ *(b++);
		xi2 = *a ^ *b;
		if (round == 2)
		{
			// skip padding byte
			xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
			xi1 = (xi1 >> 8) | (xi2 << (64 - 8));
			xi2 = (xi2 >> 8);
		}
	}
	else if (round == 3)
	{
		// xor 20 bytes
		xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
		xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
		xi2 = well_aligned_int(a, 16) ^ well_aligned_int(b, 16);
	}
	else if (round == 4 || round == 5)
	{
		// xor 16 bytes
		xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
		xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
		xi2 = 0;
		if (round == 4)
		{
			// skip padding byte
			xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
			xi1 = (xi1 >> 8);
		}
	}
	else if (round == 6)
	{
		// xor 12 bytes
		xi0 = *a++ ^ *b++;
		xi1 = *(uint *)a ^ *(uint *)b;
		xi2 = 0;
		if (round == 6)
		{
			// skip padding byte
			xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
			xi1 = (xi1 >> 8);
		}
	}
	else if (round == 7 || round == 8)
	{
		// xor 8 bytes
		xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
		xi1 = 0;
		xi2 = 0;
		if (round == 8)
		{
			// skip padding byte
			xi0 = (xi0 >> 8);
		}
	}
	// invalid solutions (which start happenning in round 5) have duplicate
	// inputs and xor to zero, so discard them
	if (!xi0 && !xi1)
		return 0;
#else
#error "unsupported NR_ROWS_LOG"
#endif
	return ht_store(round, ht_dst, ENCODE_INPUTS(row, slot_a, slot_b),
		xi0, xi1, xi2, 0, rowCounters);
}

__device__ void equihash_round_cm3(uint round, char *ht_src, char *ht_dst, uint *rowCountersSrc, uint *rowCountersDst)
{
	uint                tid = blockIdx.x * blockDim.x + threadIdx.x;
	char				*p;
	uint				cnt;
	uint                i, j;
	uint				dropped_stor = 0;
	ulong				*a, *b;
	uint				xi_offset;
	xi_offset = (8 + ((round - 1) / 2) * 4);

	cnt = rowCountersSrc[tid];
	cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
	if (!cnt) {// no elements in row, no collisions
		return;
	}
	// find collisions
	p = (ht_src + tid * c_ROW_LEN) + xi_offset;
	for (i = 0; i < cnt; i++) {
		a = (ulong *)(p + i * 32);
		for (j = i + 1; j < cnt; j++) {
			b = (ulong *)(p + j * 32);
			dropped_stor += xor_and_store(round, ht_dst, tid, i, j, a, b, rowCountersDst);
		}
	}
}

#define KERNEL_ROUND_ODD_OLD(N) \
__global__  \
void kernel_round_cm3_ ## N( char *ht_src,  char *ht_dst) \
{ \
    equihash_round_cm3(N, ht_src, ht_dst, rowCounter0, rowCounter1); \
}


#define KERNEL_ROUND_EVEN_OLD(N) \
__global__  \
void kernel_round_cm3_ ## N(char *ht_src,  char *ht_dst) \
{ \
    equihash_round_cm3(N, ht_src, ht_dst, rowCounter1, rowCounter0); \
}


KERNEL_ROUND_ODD_OLD(1)
KERNEL_ROUND_EVEN_OLD(2)
KERNEL_ROUND_ODD_OLD(3)
KERNEL_ROUND_EVEN_OLD(4)
KERNEL_ROUND_ODD_OLD(5)
KERNEL_ROUND_EVEN_OLD(6)
KERNEL_ROUND_ODD_OLD(7)


__global__
void kernel_round_cm3_8(char *ht_src, char *ht_dst)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	equihash_round_cm3(8, ht_src, ht_dst, rowCounter1, rowCounter0);
	if (!tid) {
		sols.nr = sols.likely_invalids = 0;
	}
}


__device__ uint expand_ref(const char *ht, uint xi_offset, uint row, uint slot)
{
	return *(uint *)(ht + row * NR_SLOTS * SLOT_LEN + slot * SLOT_LEN + xi_offset - 4);
}

/*
** Expand references to inputs. Return 1 if so far the solution appears valid,
** or 0 otherwise (an invalid solution would be a solution with duplicate
** inputs, which can be detected at the last step: round == 0).
*/
__device__ uint expand_refs(uint *ins, uint nr_inputs, const char **htabs, uint round)
{
	const char	*ht = htabs[round & 1];
	uint		i = nr_inputs - 1;
	uint		j = nr_inputs * 2 - 1;
	uint		xi_offset = xi_offset_for_round(round);
	int			dup_to_watch = -1;
	do
	{
		ins[j] = expand_ref(ht, xi_offset,
			DECODE_ROW(ins[i]), DECODE_SLOT1(ins[i]));
		ins[j - 1] = expand_ref(ht, xi_offset,
			DECODE_ROW(ins[i]), DECODE_SLOT0(ins[i]));
		if (!round)
		{
			if (dup_to_watch == -1)
				dup_to_watch = ins[j];
			else if (ins[j] == dup_to_watch || ins[j - 1] == dup_to_watch)
				return 0;
		}
		if (!i)
			break;
		i--;
		j -= 2;
	} while (1);
	return 1;
}

/*
** Verify if a potential solution is in fact valid.
*/
__device__ void potential_sol(const char **htabs, uint ref0, uint ref1)
{
	uint	nr_values;
	uint	values_tmp[(1 << PARAM_K)];
	uint	sol_i;
	uint	i;
	nr_values = 0;
	values_tmp[nr_values++] = ref0;
	values_tmp[nr_values++] = ref1;
	uint round = PARAM_K - 1;
	do
	{
		round--;
		if (!expand_refs(values_tmp, nr_values, htabs, round))
			return;
		nr_values *= 2;
	} while (round > 0);
	// solution appears valid, copy it to sols
	sol_i = atomicAdd(&sols.nr, 1);
	if (sol_i >= MAX_SOLS)
		return;
	for (i = 0; i < (1 << PARAM_K); i++)
		sols.values[sol_i][i] = values_tmp[i];
	sols.valid[sol_i] = 1;
}

/*
** Scan the hash tables to find Equihash solutions.
*/
__global__
void kernel_sols(const char *ht0, const char *ht1)
{
	uint		tid = blockIdx.x * blockDim.x + threadIdx.x;
	const char	*htabs[2] = { ht0, ht1 };
	//uint	*hcounters[2] = { rowCounter0, rowCounter1 };
	uint		ht_i = (PARAM_K - 1) & 1; // table filled at last round
	uint		cnt;
	uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
	uint		i, j;
	const char	*a, *b;
	uint		ref_i, ref_j;
	// it's ok for the collisions array to be so small, as if it fills up
	// the potential solutions are likely invalid (many duplicate inputs)
	ulong		collisions;
	//uint		coll;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
	// in the final hash table, we are looking for a match on both the bits
	// part of the previous PREFIX colliding bits, and the last PREFIX bits.
	uint		mask = 0xffffff;
#else
#error "unsupported NR_ROWS_LOG"
#endif

	a = htabs[ht_i] + tid * NR_SLOTS * SLOT_LEN;
	cnt = rowCounter0[tid];
	cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
	//coll = 0;
	a += xi_offset;
	for (i = 0; i < cnt; i++, a += SLOT_LEN) {
		uint a_data = ((*(uint *)a) & mask);
		ref_i = *(uint *)(a - 4);
		for (j = i + 1, b = a + SLOT_LEN; j < cnt; j++, b += SLOT_LEN) {
			if (a_data == ((*(uint *)b) & mask)) {
				ref_j = *(uint *)(b - 4);
				collisions = ((ulong)ref_i << 32) | ref_j;
				goto exit1;
			}
		}
	}
	return;

exit1:
	potential_sol(htabs, collisions >> 32, collisions & 0xffffffff);
}

static void sort_pair(uint32_t *a, uint32_t len)
{
	uint32_t    *b = a + len;
	uint32_t     tmp, need_sorting = 0;
	for (uint32_t i = 0; i < len; i++)
		if (need_sorting || a[i] > b[i])
		{
			need_sorting = 1;
			tmp = a[i];
			a[i] = b[i];
			b[i] = tmp;
		}
		else if (a[i] < b[i])
			return;
}

static uint32_t verify_sol(sols_t *sols, unsigned sol_i)
{
	uint32_t  *inputs = sols->values[sol_i];
	uint32_t  seen_len = (1 << (PREFIX + 1)) / 8;
	uint8_t seen[(1 << (PREFIX + 1)) / 8];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, seen_len);
	for (i = 0; i < (1 << PARAM_K); i++)
	{
		tmp = seen[inputs[i] / 8];
		seen[inputs[i] / 8] |= 1 << (inputs[i] & 7);
		if (tmp == seen[inputs[i] / 8])
		{
			// at least one input value is a duplicate
			sols->valid[sol_i] = 0;
			return 0;
		}
	}
	// the valid flag is already set by the GPU, but set it again because
	// I plan to change the GPU code to not set it
	sols->valid[sol_i] = 1;
	// sort the pairs in place
	for (uint32_t level = 0; level < PARAM_K; level++)
		for (i = 0; i < (1 << PARAM_K); i += (2 << level))
			sort_pair(&inputs[i], 1 << level);
	return 1;
}

struct __align__(64) c_context {
	char* buf_ht[2], *buf_dbg;
	sols_t	*sols;
	uint32_t nthreads;
	size_t global_ws;

	c_context(const uint32_t n_threads) {
		nthreads = n_threads;
	}
	void* operator new(size_t i) {
		return _mm_malloc(i, 64);
	}
	void operator delete(void* p) {
		_mm_free(p);
	}
};

static void compress(uint8_t *out, uint32_t *inputs, uint32_t n)
{
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
	uint8_t *pOut = out;
	while (byte_pos < n)
	{
		if (bits_left >= 8 - x_bits_used)
		{
			x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used);
			bits_left -= 8 - x_bits_used;
			x_bits_used = 8;
		}
		else if (bits_left > 0)
		{
			uint32_t mask = ~(-1 << (8 - x_bits_used));
			mask = ((~mask) >> bits_left) & mask;
			x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask;
			x_bits_used += bits_left;
			bits_left = 0;
		}
		else if (bits_left <= 0)
		{
			assert(!bits_left);
			byte_pos++;
			bits_left = PREFIX + 1;
		}
		if (x_bits_used == 8)
		{
			*pOut++ = x;
			x = x_bits_used = 0;
		}
	}
}

sa_cuda_context::sa_cuda_context(int tpb, int blocks, int id)
	: threadsperblock(tpb), totalblocks(blocks), device_id(id)
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaDeviceReset());
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	eq = new c_context(threadsperblock * totalblocks);
#ifdef ENABLE_DEBUG
	size_t              dbg_size = NR_ROWS;
#else
	size_t              dbg_size = 1;
#endif

	checkCudaErrors(cudaMalloc((void**)&eq->buf_dbg, dbg_size));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_ht[0], HT_SIZE));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_ht[1], HT_SIZE));
	checkCudaErrors(cudaMallocHost(&eq->sols, sizeof(*eq->sols)));
	checkCudaErrors(cudaDeviceSynchronize());
}

sa_cuda_context::~sa_cuda_context()
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaDeviceReset());
	delete eq;
}

void sa_cuda_context::solve(const char * tequihash_header, unsigned int tequihash_header_len, const char * nonce, unsigned int nonce_len, std::function<bool()> cancelf, std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf, std::function<void(void)> hashdonef)
{
	checkCudaErrors(cudaSetDevice(device_id));

	unsigned char context[140];
	memset(context, 0, 140);
	memcpy(context, tequihash_header, tequihash_header_len);
	memcpy(context + tequihash_header_len, nonce, nonce_len);

	c_context *miner = eq;

	//FUNCTION<<<totalblocks, threadsperblock>>>(ARGUMENTS)

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

	checkCudaErrors(cudaMemcpyToSymbol(blake, &initialCtx, sizeof(blake2b_state_s), 0, cudaMemcpyHostToDevice));

	constexpr uint32_t THREAD_SHIFT = 7;
	constexpr uint32_t THREAD_COUNT = 1 << THREAD_SHIFT;
	constexpr uint32_t DIM_SIZE = NR_ROWS >> THREAD_SHIFT;

	kernel_init_0 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round0<<<1024, 64>>> (miner->buf_ht[0], 16);
	if (cancelf()) return;
	kernel_init_1 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_1 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[0], miner->buf_ht[1]);
	if (cancelf()) return;
	kernel_init_0 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_2 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[1], miner->buf_ht[0]);
	if (cancelf()) return;
	kernel_init_1 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_3 << <DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[0], miner->buf_ht[1]);
	if (cancelf()) return;
	kernel_init_0 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_4 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[1], miner->buf_ht[0]);
	if (cancelf()) return;
	kernel_init_1 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_5 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[0], miner->buf_ht[1]);
	if (cancelf()) return;
	kernel_init_0 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_6 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[1], miner->buf_ht[0]);
	if (cancelf()) return;
	kernel_init_1 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_7 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[0], miner->buf_ht[1]);
	if (cancelf()) return;
	kernel_init_0 << <DIM_SIZE, THREAD_COUNT >> > ();
	kernel_round_cm3_8 << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[1], miner->buf_ht[0]);
	if (cancelf()) return;
	kernel_sols << < DIM_SIZE, THREAD_COUNT >> > (miner->buf_ht[0], miner->buf_ht[1]);

	checkCudaErrors(cudaMemcpyFromSymbol(miner->sols, sols, sizeof(sols_t), 0, cudaMemcpyDeviceToHost));

	if (miner->sols->nr > MAX_SOLS)
		miner->sols->nr = MAX_SOLS;

	for (unsigned sol_i = 0; sol_i < miner->sols->nr; sol_i++) {
		verify_sol(miner->sols, sol_i);
	}


	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i <  miner->sols->nr; i++) {
		if (miner->sols->valid[i]) {
			compress(proof, (uint32_t *)(miner->sols->values[i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}
	}
	hashdonef();

}