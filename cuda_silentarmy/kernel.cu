#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <functional>
#include <vector>
#include <iostream>
#include <stdint.h>

#include "sa_cuda_context.hpp"
#include "param.h"
#include "sa_blake.h"

#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long long  ulong;
typedef unsigned short ushort;
typedef uint32_t u32;

typedef struct sols_s
{
	uint nr;
	uint likely_invalids;
	uchar valid[MAX_SOLS];
	uint values[MAX_SOLS][(1 << PARAM_K)];
} sols_t;

/*
** Assuming NR_ROWS_LOG == 16, the hash table slots have this layout (length in
** bytes in parens):
**
** round 0, table 0: cnt(4) i(4)                     pad(0)   Xi(23.0) pad(1)
** round 1, table 1: cnt(4) i(4)                     pad(0.5) Xi(20.5) pad(3)
** round 2, table 0: cnt(4) i(4) i(4)                pad(0)   Xi(18.0) pad(2)
** round 3, table 1: cnt(4) i(4) i(4)                pad(0.5) Xi(15.5) pad(4)
** round 4, table 0: cnt(4) i(4) i(4) i(4)           pad(0)   Xi(13.0) pad(3)
** round 5, table 1: cnt(4) i(4) i(4) i(4)           pad(0.5) Xi(10.5) pad(5)
** round 6, table 0: cnt(4) i(4) i(4) i(4) i(4)      pad(0)   Xi( 8.0) pad(4)
** round 7, table 1: cnt(4) i(4) i(4) i(4) i(4)      pad(0.5) Xi( 5.5) pad(6)
** round 8, table 0: cnt(4) i(4) i(4) i(4) i(4) i(4) pad(0)   Xi( 3.0) pad(5)
**
** If the first byte of Xi is 0xAB then:
** - on even rounds, 'A' is part of the colliding PREFIX, 'B' is part of Xi
** - on odd rounds, 'A' and 'B' are both part of the colliding PREFIX, but
**   'A' is considered redundant padding as it was used to compute the row #
**
** - cnt is an atomic counter keeping track of the number of used slots.
**   it is used in the first slot only; subsequent slots replace it with
**   4 padding bytes
** - i encodes either the 21-bit input value (round 0) or a reference to two
**   inputs from the previous round
**
** Formula for Xi length and pad length above:
** > for i in range(9):
** >   xi=(200-20*i-NR_ROWS_LOG)/8.; ci=8+4*((i)/2); print xi,32-ci-xi
**
** Note that the fractional .5-byte/4-bit padding following Xi for odd rounds
** is the 4 most significant bits of the last byte of Xi.
*/

__constant__ ulong blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};


__device__ uint32_t rowCounter0[NR_ROWS];
__device__ uint32_t rowCounter1[NR_ROWS];
__device__ blake2b_state_t blake;
__device__ sols_t sols;


/*
** Reset counters in hash table.
*/
__global__
void kernel_init_ht0()
{
	rowCounter0[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

__global__
void kernel_init_ht1()
{
	rowCounter1[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}


/*
** If xi0,xi1,xi2,xi3 are stored consecutively in little endian then they
** represent (hex notation, group of 5 hex digits are a group of PREFIX bits):
**   aa aa ab bb bb cc cc cd dd...  [round 0]
**         --------------------
**      ...ab bb bb cc cc cd dd...  [odd round]
**               --------------
**               ...cc cc cd dd...  [next even round]

**                        -----
** Bytes underlined are going to be stored in the slot. Preceding bytes
** (and possibly part of the underlined bytes, depending on NR_ROWS_LOG) are
** used to compute the row number.
**
** Round 0: xi0,xi1,xi2,xi3 is a 25-byte Xi (xi3: only the low byte matter)
** Round 1: xi0,xi1,xi2 is a 23-byte Xi (incl. the colliding PREFIX nibble)
** TODO: update lines below with padding nibbles
** Round 2: xi0,xi1,xi2 is a 20-byte Xi (xi2: only the low 4 bytes matter)
** Round 3: xi0,xi1,xi2 is a 17.5-byte Xi (xi2: only the low 1.5 bytes matter)
** Round 4: xi0,xi1 is a 15-byte Xi (xi1: only the low 7 bytes matter)
** Round 5: xi0,xi1 is a 12.5-byte Xi (xi1: only the low 4.5 bytes matter)
** Round 6: xi0,xi1 is a 10-byte Xi (xi1: only the low 2 bytes matter)
** Round 7: xi0 is a 7.5-byte Xi (xi0: only the low 7.5 bytes matter)
** Round 8: xi0 is a 5-byte Xi (xi0: only the low 5 bytes matter)
**
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
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
	p = ht + row * NR_SLOTS * SLOT_LEN;
	uint xcnt = atomicAdd(&rowCounters[row], 1);
	//printf("inc index %u round %u\n", rowIdx, round);
	cnt = xcnt;
	//printf("row %u rowOffset %u count is %u\n", rowIdx, rowOffset, cnt);
	if (cnt >= NR_SLOTS) {
		// avoid overflows
		atomicSub(&rowCounters[row], 1);
		return 1;
	}
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

/*
** Execute round 0 (blake).
**
** Note: making the work group size less than or equal to the wavefront size
** allows the OpenCL compiler to remove the barrier() calls, see "2.2 Local
** Memory (LDS) Optimization 2-10" in:
** http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/opencl-optimization-guide/
*/
__global__
void kernel_round0(char *ht, uint *debug)
{
	uint                tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulong               v[16];
	uint                inputs_per_thread = NR_INPUTS / (gridDim.x * blockDim.x);
	uint                input = tid * inputs_per_thread;
	uint                input_end = (tid + 1) * inputs_per_thread;
	uint                dropped = 0;

	while (input < input_end)
	{
		//atomicAdd(&ran, 1);
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

__device__ void equihash_round_cm3(uint round,
	char *ht_src,
	char *ht_dst,
	uint *rowCountersSrc,
	uint *rowCountersDst)
{
	uint                tid = blockIdx.x * blockDim.x + threadIdx.x;
	char				*p;
	uint                cnt;
	uint                i, j;
	uint				dropped_stor = 0;
	ulong				*a, *b;
	uint				xi_offset;
	static uint			size = NR_ROWS;
	static uint			stride = NR_SLOTS * SLOT_LEN;
	xi_offset = (8 + ((round - 1) / 2) * 4);

	for (uint ii = tid; ii < size; ii += (blockDim.x * gridDim.x)) {
		p = ht_src + ii * stride;
		cnt = rowCountersSrc[ii];
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
		if (!cnt) {// no elements in row, no collisions
			continue;
		}
		// find collisions
		for (i = 0; i < cnt; i++) {
			for (j = i + 1; j < cnt; j++)
			{
				a = (ulong *)
					(ht_src + ii * stride + i * 32 + xi_offset);
				b = (ulong *)
					(ht_src + ii * stride + j * 32 + xi_offset);
				dropped_stor += xor_and_store(round, ht_dst, ii, i, j, a, b, rowCountersDst);
			}
		}
		//if (round < 8) {
		// reset the counter in preparation of the next round
		//rowCountersSrc[ii] = 0;//might be doing this already
		//*(uint *)(ht_src + ii * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 6) * 32) = 0;
		//}
	}
}

/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/
__device__ void equihash_round(uint round,
	char *ht_src,
	char *ht_dst,
	uint *debug,
	uint *rowCountersSrc,
	uint *rowCountersDst)
{
	uint		tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint		tlid = threadIdx.x;

	__shared__ uchar first_words_data[(NR_SLOTS + 2) * 64];
	__shared__ uint	collisionsData[COLL_DATA_SIZE_PER_TH * 64];
	__shared__ uint collisionsNum;

	char	*p;
	uint		cnt;
	uchar	*first_words = &first_words_data[(NR_SLOTS + 2)*tlid];
	uchar		mask;
	uint		i, j;
	// NR_SLOTS is already oversized (by a factor of OVERHEAD), but we want to
	// make it even larger
	uint		n;
	uint		dropped_coll = 0;
	uint		dropped_stor = 0;
	ulong	*a, *b;
	uint		xi_offset;
	// read first words of Xi from the previous (round - 1) hash table
	xi_offset = xi_offset_for_round(round - 1);
	// the mask is also computed to read data from the previous round
#if NR_ROWS_LOG == 16
	mask = ((!(round & 1)) ? 0x0f : 0xf0);
#elif NR_ROWS_LOG == 18
	mask = ((!(round & 1)) ? 0x03 : 0x30);
#elif NR_ROWS_LOG == 19
	mask = ((!(round & 1)) ? 0x01 : 0x10);
#elif NR_ROWS_LOG == 20
	mask = 0; /* we can vastly simplify the code below */
#else
#error "unsupported NR_ROWS_LOG"
#endif
	uint thCollNum = 0;
	collisionsNum = 0;
	__syncthreads();
	p = (ht_src + tid * NR_SLOTS * SLOT_LEN);
	cnt = rowCountersSrc[tid];
	cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
	if (!cnt) {
		// no elements in row, no collisions
		goto part2;
	}
	p += xi_offset;
	for (i = 0; i < cnt; i++, p += SLOT_LEN)
		first_words[i] = (*(uchar *)p) & mask;
	// find collisions
	for (i = 0; i < cnt - 1 && thCollNum < COLL_DATA_SIZE_PER_TH; i++)
	{
		uchar data_i = first_words[i];
		uint collision = (tid << 10) | (i << 5) | (i + 1);
		for (j = i + 1; (j + 4) < cnt;)
		{
			{
				uint isColl = ((data_i == first_words[j]) ? 1 : 0);
				if (isColl)
				{
					thCollNum++;
					uint index = atomicAdd(&collisionsNum, 1);
					collisionsData[index] = collision;
				}
				collision++;
				j++;
			}
			{
				uint isColl = ((data_i == first_words[j]) ? 1 : 0);
				if (isColl)
				{
					thCollNum++;
					uint index = atomicAdd(&collisionsNum, 1);
					collisionsData[index] = collision;
				}
				collision++;
				j++;
			}
			{
				uint isColl = ((data_i == first_words[j]) ? 1 : 0);
				if (isColl)
				{
					thCollNum++;
					uint index = atomicAdd(&collisionsNum, 1);
					collisionsData[index] = collision;
				}
				collision++;
				j++;
			}
			{
				uint isColl = ((data_i == first_words[j]) ? 1 : 0);
				if (isColl)
				{
					thCollNum++;
					uint index = atomicAdd(&collisionsNum, 1);
					collisionsData[index] = collision;
				}
				collision++;
				j++;
			}
		}
		for (; j < cnt; j++)
		{
			uint isColl = ((data_i == first_words[j]) ? 1 : 0);
			if (isColl)
			{
				thCollNum++;
				uint index = atomicAdd(&collisionsNum, 1);
				collisionsData[index] = collision;
			}
			collision++;
		}
	}
part2:
	__syncthreads();
	uint totalCollisions = collisionsNum;
	for (uint index = tlid; index < totalCollisions; index += blockDim.x) {
		uint collision = collisionsData[index];
		uint collisionThreadId = collision >> 10;
		uint i = (collision >> 5) & 0x1F;
		uint j = collision & 0x1F;
		uchar *ptr = (uchar*)ht_src + collisionThreadId * NR_SLOTS * SLOT_LEN +
			xi_offset;
		a = (ulong *)(ptr + i * SLOT_LEN);
		b = (ulong *)(ptr + j * SLOT_LEN);
		dropped_stor += xor_and_store(round, ht_dst, collisionThreadId, i, j,
			a, b, rowCountersDst);
	}
}

/*
** This defines kernel_round1, kernel_round2, ..., kernel_round7.
*/
#define KERNEL_ROUND_ODD(N) \
__global__  \
void kernel_round ## N( char *ht_src,  char *ht_dst, uint *debug) \
{ \
    equihash_round(N, ht_src, ht_dst, debug, rowCounter0, rowCounter1); \
}

#define KERNEL_ROUND_EVEN(N) \
__global__  \
void kernel_round ## N( char *ht_src,  char *ht_dst, uint *debug) \
{ \
    equihash_round(N, ht_src, ht_dst, debug, rowCounter1, rowCounter0); \
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


KERNEL_ROUND_ODD(1)
KERNEL_ROUND_EVEN(2)
KERNEL_ROUND_ODD(3)
KERNEL_ROUND_EVEN(4)
KERNEL_ROUND_ODD(5)
KERNEL_ROUND_EVEN(6)
KERNEL_ROUND_ODD(7)

KERNEL_ROUND_ODD_OLD(1)
KERNEL_ROUND_EVEN_OLD(2)
KERNEL_ROUND_ODD_OLD(3)
KERNEL_ROUND_EVEN_OLD(4)
KERNEL_ROUND_ODD_OLD(5)
KERNEL_ROUND_EVEN_OLD(6)
KERNEL_ROUND_ODD_OLD(7)


// kernel_round8 takes an extra argument, "sols"
__global__
void kernel_round8(char *ht_src, char *ht_dst, uint *debug)
{
	uint		tid = blockIdx.x * blockDim.x + threadIdx.x;
	equihash_round(8, ht_src, ht_dst, debug, rowCounter1, rowCounter0);
	if (!tid) {
		sols.nr = sols.likely_invalids = 0;
	}
}

__global__
void kernel_round_cm3_8(char *ht_src, char *ht_dst)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	equihash_round_cm3(8, ht_src, ht_dst, rowCounter1, rowCounter0);
	if (!tid) {
		sols.nr = sols.likely_invalids = 0;
	}
}


__device__ uint expand_ref(char *ht, uint xi_offset, uint row, uint slot)
{
	return *(uint *)(ht + row * NR_SLOTS * SLOT_LEN +
		slot * SLOT_LEN + xi_offset - 4);
}

/*
** Expand references to inputs. Return 1 if so far the solution appears valid,
** or 0 otherwise (an invalid solution would be a solution with duplicate
** inputs, which can be detected at the last step: round == 0).
*/
__device__ uint expand_refs(uint *ins, uint nr_inputs, char **htabs,
	uint round)
{
	char	*ht = htabs[round & 1];
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
__device__ void potential_sol(char **htabs, uint ref0, uint ref1)
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
void kernel_sols(char *ht0, char *ht1)
{
	uint		tid = blockIdx.x * blockDim.x + threadIdx.x;
	char	*htabs[2] = { ht0, ht1 };
	uint		ht_i = (PARAM_K - 1) & 1; // table filled at last round
	uint		cnt;
	uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
	uint		i, j;
	char	*a, *b;
	uint		ref_i, ref_j;
	// it's ok for the collisions array to be so small, as if it fills up
	// the potential solutions are likely invalid (many duplicate inputs)
	ulong		collisions;
	uint		coll;
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
	coll = 0;
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
struct __align__(64) c_context {
	char* buf_ht[2], *buf_dbg;
	//uint *rowCounters[2];
	//sols_t	*sols;
	u32 nthreads;
	size_t global_ws;

	c_context(const u32 n_threads) {
		nthreads = n_threads;
	}
	void* operator new(size_t i) {
		return _mm_malloc(i, 64);
	}
	void operator delete(void* p) {
		_mm_free(p);
	}
};

static size_t select_work_size_blake(void)
{
	size_t              work_size =
		64 * /* thread per wavefront */
		BLAKE_WPS * /* wavefront per simd */
		4 * /* simd per compute unit */
		36;
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	while (NR_INPUTS % work_size)
		work_size += 64;
	//debug("Blake: work size %zd\n", work_size);
	return work_size;
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
	checkCudaErrors(cudaDeviceSynchronize());
	//eq->sols = (sols_t *)malloc(sizeof(sols_t));
}

sa_cuda_context::~sa_cuda_context()
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaDeviceReset());
	delete eq;
}

#define CHECK_LAUNCH() \
checkCudaErrors(cudaPeekAtLastError()); \
checkCudaErrors(cudaDeviceSynchronize());


static inline void solve_new(c_context *miner, unsigned round)
{
	constexpr uint32_t THREAD_SHIFT = 10;
	constexpr uint32_t THREAD_COUNT = 1 << THREAD_SHIFT;
	constexpr uint32_t DIM_SIZE = (1 << 20) >> THREAD_SHIFT;


	// Now on every round!!!!
	switch (round) {
	case 0:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round0 << <1024, 64 >> >(miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 1:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round1 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 2:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round2 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 3:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round3 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 4:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round4 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 5:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round5 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 6:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round6 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 7:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round7 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 8:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round8 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	}
}

static inline void solve_old(unsigned round, c_context *miner)
{
	constexpr uint32_t THREAD_SHIFT = 10;
	constexpr uint32_t THREAD_COUNT = 1 << THREAD_SHIFT;
	constexpr uint32_t DIM_SIZE = (1 << 20) >> THREAD_SHIFT;
	// Now on every round!!!!
	switch (round) {
	case 0:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round0 << <1024, 64 >> >(miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 1:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_1 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 2:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_2 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 3:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_3 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 4:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_4 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 5:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_5 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 6:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_6 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 7:
		kernel_init_ht1 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_7 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 8:
		kernel_init_ht0 << <DIM_SIZE, THREAD_COUNT >> > ();
		kernel_round_cm3_8 << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	}
}

#include <fstream>
void sa_cuda_context::solve(const char * tequihash_header, unsigned int tequihash_header_len, const char * nonce, unsigned int nonce_len, std::function<bool()> cancelf, std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf, std::function<void(void)> hashdonef)
{
	checkCudaErrors(cudaSetDevice(device_id));
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));

	bool bUseOld = prop.major < 5;


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

	for (unsigned round = 0; round < PARAM_K; round++) {
		if (bUseOld) {
			solve_old(round, miner);
		} else {
			solve_new(miner, round);
		}
		if (cancelf()) return;
	}
	kernel_sols << <NR_ROWS >> 6, 64 >> >(miner->buf_ht[0], miner->buf_ht[1]);

	sols_t l_sols;

	checkCudaErrors(cudaMemcpyFromSymbol(&l_sols, sols, sizeof(sols_t), 0, cudaMemcpyDeviceToHost));

	if (l_sols.nr > MAX_SOLS)
		l_sols.nr = MAX_SOLS;

	for (unsigned sol_i = 0; sol_i < l_sols.nr; sol_i++) {
		verify_sol(&l_sols, sol_i);
	}

	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < l_sols.nr; i++) {
		if (l_sols.valid[i]) {
			compress(proof, (uint32_t *)(l_sols.values[i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}
	}
	hashdonef();
}