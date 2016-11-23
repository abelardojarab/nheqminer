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

__device__ char rowCounter0[NR_ROWS];
__device__ char rowCounter1[NR_ROWS];
__device__ sols_t sols;
__device__ blake2b_state_t blake;

__constant__ ulong blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

/*
** Reset counters in hash table.
*/
__global__
void kernel_init_ht0()
{
	((uint*)rowCounter0)[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

__global__
void kernel_init_ht1()
{
	((uint*)rowCounter1)[blockIdx.x * blockDim.x + threadIdx.x] = 0;
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

#define nv64to32(low,high,X) asm volatile( "mov.b64 {%0,%1}, %2; \n\t" : "=r"(low), "=r"(high) : "l"(X))
#define nv32to64(X,low,high) asm volatile( "mov.b64 %0,{%1, %2}; \n\t": "=l"(X) : "r"(low), "r"(high))

__device__ uint ht_store(uint round, char *ht, uint i,
	ulong xi0, ulong xi1, ulong xi2, ulong xi3, uint *rowCounters)
{
	uint    row;
	char       *p;
	uint                cnt;
	uint                tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint                tlid = threadIdx.x;
#if NR_ROWS_LOG == 16
	if (!(round % 2))
		row = (xi0 & 0xffff);
	else
		// if we have in hex: "ab cd ef..." (little endian xi0) then this
		// formula computes the row as 0xdebc. it skips the 'a' nibble as it
		// is part of the PREFIX. The Xi will be stored starting with "ef...";
		// 'e' will be considered padding and 'f' is part of the current PREFIX
		row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
	else
		row = ((xi0 & 0xc0000) >> 2) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
	else
		row = ((xi0 & 0xe0000) >> 1) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
	else
		row = ((xi0 & 0xf0000) >> 0) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif

	/*
	xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
	*/
	//256bit shfr
	if (round == 0 || round == 1 || round == 2)
	{
		xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
		xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
		xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
	}
	else if (round == 3 || round == 4 || round == 5) {
		xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
		xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	}
	else if (round == 6 || round == 7 || round == 8) {
		xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
		/*	uint xi0_0=(uint)(xi0);
		uint xi0_1=(uint)(xi0 >> 32);
		uint xi1_0=(uint)(xi1);
		uint xi1_1=(uint)(xi1 >> 32);
		*/
		/*
		uint xi0l,xi0h,xi1l,xi1h;
		uint _xi0l,_xi0h,_xi1l,_xi1h;
		nv64to32(xi0l,xi0h,xi0);
		nv64to32(xi1l,xi1h,xi1);
		asm("{\n\t"
		"shf.r.clamp.b32 %0,%4,%5,%8; \n\t"
		"shf.r.clamp.b32 %1,%5,%6,%8; \n\t"
		"shf.r.clamp.b32 %2,%6,%7,%8; \n\t"
		"shr.b32 %3,%7,%8; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l), "=r"(_xi1h):  "r"(xi0l), "r"(xi0h),"r"(xi1l), "r"(xi1h) , "r"(16));
		nv32to64(xi0,_xi0l,_xi0h);
		*/

		/*
		uint xi0l,xi0h,xi1l,xi1h;
		uint _xi0l,_xi0h,_xi1l,_xi1h;
		nv64to32(xi0l,xi0h,xi0);
		nv64to32(xi1l,xi1h,xi1);
		asm("{\n\t"
		"shf.r.clamp.b32 %0,%4,%5,%8; \n\t"
		"shf.r.clamp.b32 %1,%5,%6,%8; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l), "=r"(_xi1h):  "r"(xi0l), "r"(xi0h),"r"(xi1l), "r"(xi1h) , "r"(16));
		nv32to64(xi0,_xi0l,_xi0h);
		*/

	}

	p = ht + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	// store "i" (always 4 bytes before Xi)
	if (round == 0 || round == 1)
	{
		// store 24 bytes
		ulonglong2 store0;
		ulonglong2 store1;
		//store0.x=(ulong)i  | (ulong)i << 32;
		nv32to64(store0.x, i, i);
		store0.y = xi0;
		*(ulonglong2 *)(pp) = store0;
		store1.x = xi1;
		store1.y = xi2;
		*(ulonglong2 *)(pp + 16) = store1;

		/*
		ulong2 store;
		store.x=xi1;
		store.y=xi2;
		*( ulong2 *)(p + 8)=store;
		*( ulong *)(p + 0) = xi0;
		*( uint *)(p - 4) = i;
		*/

	}
	else if (round == 2)
	{
		// store 20 bytes
		/*
		*( ulong *)(p - 4) = ((ulong)i) | (xi0 << 32);
		*( ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*( ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
		*/
		*(uint *)(p - 4) = i;
		*(uint *)(p + 0) = xi0;

		/*
		uint xi0l, xi0h, xi1l, xi1h, xi2l, xi2h;
		nv64to32(xi0l, xi0h, xi0);
		nv64to32(xi1l, xi1h, xi1);
		nv64to32(xi2l, xi2h, xi2);
		*( uint *)(p + 0) = xi0l;
		ulong s1,s2;
		nv32to64(s1,xi0h,xi1l);
		*( ulong *)(p + 4) = s1;
		nv32to64(s2,xi1h,xi2l);
		*( ulong *)(p + 12) = s2;
		*/
		/*
		*( uint *)(p + 4) = xi0h;
		*( uint *)(p + 8) = xi1l;
		*( uint *)(p + 12) = xi1h;
		*( uint *)(p + 16) = xi2l;
		*/


		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
	}
	else if (round == 3)
	{
		// store 16 bytes
		//8 byte align	

		*(ulong *)(p - 4) = ((ulong)i) | (xi0 << 32);
		*(ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		*(uint *)(p + 12) = (xi1 >> 32);
		*(uint *)(p - 4) = i;
		uint xi0l, xi0h, xi1l, xi1h;
		nv64to32(xi0l, xi0h, xi0);
		nv64to32(xi1l, xi1h, xi1);
		*(uint *)(p + 0) = xi0l;
		ulong s1, s2;
		nv32to64(s1, xi0h, xi1l);
		*(ulong *)(p + 4) = s1;
		*(uint *)(p + 12) = xi1h;

		//	*( uint *)(p + 0) = xi0;
		//	*( ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
		//	*( uint *)(p + 12) = (xi1 >> 32);
	}
	else if (round == 4)
	{
		// store 16 bytes

		ulong2 store;
		store.x = xi0;
		store.y = xi1;
		*(ulong2 *)(p + 0) = store;
		*(uint *)(p - 4) = i;

		/*
		*( uint *)(p - 4) = i;
		*( ulong *)(p + 0) = xi0;
		*( ulong *)(p + 8) = xi1;
		*/
	}
	else if (round == 5)
	{
		// store 12 bytes
		*(uint *)(p - 4) = i;
		*(ulong *)(p + 0) = xi0;

		*(uint *)(p + 8) = xi1;
	}
	else if (round == 6 || round == 7)
	{
		// store 8 bytes
		/*
		uint xi0l,xi0h;
		nv64to32(xi0l,xi0h,xi0);
		ulong s1;
		nv32to64(s1,i,xi0l);
		*( ulong *)(p - 4) = s1;
		*( uint *)(p + 4) = xi0h;
		*/
		*(ulong *)(p - 4) = ((ulong)i) | (xi0 << 32);
		*(uint *)(p + 4) = (xi0 >> 32);

		/*
		*( uint *)(p - 4) = i;
		*( uint *)(p + 0) = xi0;
		*( uint *)(p + 4) = (xi0 >> 32);
		*/
	}
	else if (round == 8)
	{
		//4 byte align
		*(uint *)(p - 4) = i;
		// store 4 bytes
		*(uint *)(p + 0) = xi0;

	}

	// *( uint *)(p - 4) = i;
	return 0;
}


// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320
__device__ __forceinline__
uint64_t rotate(const uint64_t value, const int offset) {
	uint2 result;
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120
__device__ __forceinline__
uint64_t rotate(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shl.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shr.b64 %0, %1, roff;\n\t"
		"add.u64 %0, lhs, %0;\n\t"
		"}\n"
		: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#else
/* host */
#define rotate(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif

#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
vd = rotate((vd ^ va), (int)64 - 32); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (int)64 - 24); \
va = (va + vb + y); \
vd = rotate((vd ^ va), (int)64 - 16); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (int)64 - 63);

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
			h[3], (uint*)rowCounter0);
		dropped += ht_store(0, ht, input * 2 + 1,
			(h[3] >> 8) | (h[4] << (64 - 8)),
			(h[4] >> 8) | (h[5] << (64 - 8)),
			(h[5] >> 8) | (h[6] << (64 - 8)),
			(h[6] >> 8), (uint*)rowCounter0);
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

__device__ uint read_uint_once(uint *p) {
	uint r;
	asm volatile("ld.global.cg.b32  %0, [%1];\n\t" : "=r"(r) : "l"(p));
	return r;
	//return *p;
}


__device__ ulong read_ulong_once(ulong *p) {
	ulong r;
	asm volatile ("ld.global.lu.b64  %0, [%1];\n\t" : "=l"(r) : "l"(p));
	return r;
	// return *p;
}


/*
** Access a half-aligned long, that is a long aligned on a 4-byte boundary.
*/
__device__ ulong half_aligned_long(ulong *p, uint offset)
{
	return
		(((ulong)*(uint *)((char *)p + offset + 0)) << 0) |
		(((ulong)*(uint *)((char *)p + offset + 4)) << 32);
}

__device__ ulong xor_unaligned_long(ulong *a, ulong *b, uint offset)
{
	uint l1 = *((uint *)((char *)a + offset + 0)) ^ *(uint *)((char *)b + offset + 0);
	uint l2 = *(uint *)((char *)a + offset + 4) ^ *(uint *)((char *)b + offset + 4);
	return ((ulong)l1 << 0 | (ulong)l2 << 32);
}

/*
** Access a well-aligned int.
*/
__device__ uint well_aligned_int(ulong *_p, uint offset)
{
	char *p = (char *)_p;
	//    return *( uint *)(p + offset);
	return read_uint_once((uint *)(p + offset));
}


#if 0
/*
** Access a well-aligned long.
*/
__device__ ulong well_aligned_long(ulong *_p, uint offset)
{
	char *p = (char *)_p;
	return read_ulong_once((ulong *)(p + offset));
}
#endif

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

__device__ uint get_row_nr_4(uint xi0, uint round) {
	uint row;
#if NR_ROWS_LOG == 16
	if (!(round % 2))
		row = (xi0 & 0xffff);
	else
		// if we have in hex: "ab cd ef..." (little endian xi0) then this
		// formula computes the row as 0xdebc. it skips the 'a' nibble as it
		// is part of the PREFIX. The Xi will be stored starting with "ef...";
		// 'e' will be considered padding and 'f' is part of the current PREFIX
		row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
	else
		row = ((xi0 & 0xc0000) >> 2) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
	else
		row = ((xi0 & 0xe0000) >> 1) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
	else
		row = ((xi0 & 0xf0000) >> 0) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	return row;
}

__device__ uint get_row_nr_8(ulong xi0, uint round) {
	uint row;
#if NR_ROWS_LOG == 16
	if (!(round % 2))
		row = (xi0 & 0xffff);
	else
		// if we have in hex: "ab cd ef..." (little endian xi0) then this
		// formula computes the row as 0xdebc. it skips the 'a' nibble as it
		// is part of the PREFIX. The Xi will be stored starting with "ef...";
		// 'e' will be considered padding and 'f' is part of the current PREFIX
		row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
	else
		row = ((xi0 & 0xc0000) >> 2) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
	else
		row = ((xi0 & 0xe0000) >> 1) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
	if (!(round % 2))
		row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
	else
		row = ((xi0 & 0xf0000) >> 0) |
		((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
		((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	return row;
}

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320

__device__ void store8(char *p, ulong store) {
	asm volatile ("st.global.cs.b64  [%0], %1;\n\t" :: "l"(p), "l" (store));
}

__device__ void store4(char *p, uint store) {
	asm volatile ("st.global.cs.b32  [%0], %1;\n\t" :: "l"(p), "r" (store));
}

__device__ void store_ulong2(char *p, ulonglong2 store) {
	asm volatile ("st.global.cs.v2.b64  [%0],{ %1, %2 };\n\t" :: "l"(p), "l" (store.x), "l" (store.y));
}

__device__ void store_uint2(char *p, uint2 store) {
	asm volatile ("st.global.cs.v2.b32  [%0],{ %1, %2 };\n\t" :: "l"(p), "r" (store.x), "r" (store.y));
}

__device__ void store_uint4(char *p, uint4 store) {
	asm volatile ("st.global.cs.v4.b32  [%0],{ %1, %2, %3, %4 };\n\t" :: "l"(p), "r" (store.x), "r" (store.y), "r" (store.z), "r" (store.w));
}

__device__ ulong load8_last(ulong *p, uint offset) {
	p = (ulong *)((char *)p + offset);
	ulong r;
	asm volatile ("ld.global.cs.nc.b64  %0, [%1];\n\t" : "=l"(r) : "l"(p));
	return r;
}

__device__ ulong load8(ulong *p, uint offset) {
	p = (ulong *)((char *)p + offset);
	ulong r;
	asm volatile ("ld.global.cs.nc.b64  %0, [%1];\n\t" : "=l"(r) : "l"(p));
	return r;
}

__device__ ulonglong2 load16l(ulong *p, uint offset) {
	p = (ulong *)((char *)p + offset);
	ulonglong2 r;
	asm volatile ("ld.global.cs.nc.v2.b64  {%0,%1}, [%2];\n\t" : "=l"(r.x), "=l"(r.y) : "l"(p));
	return r;
}

__device__ uint load4_last(ulong *p, uint offset) {
	p = (ulong *)((char *)p + offset);
	uint r;
	asm volatile ("ld.global.cs.nc.b32  %0, [%1];\n\t" : "=r"(r) : "l"(p));
	return r;
}

__device__ uint load4(ulong *p, int offset) {
	p = (ulong *)((char *)p + offset);
	uint r;
	asm volatile ("ld.global.cs.nc.b32  %0, [%1];\n\t" : "=r"(r) : "l"(p));
	return r;
}

__device__ void trigger_err() {
	load8_last((ulong *)-1, 0);
}
#else//COMPUTE 20, 30  this will be slower, unfortunetly

__device__ uint load4(ulong *p, int offset) {
	return *(uint *)((char *)p + offset);
}
__device__ uint load4_last(ulong *p, int offset) {
	return *(uint *)((char *)p + offset);
}
__device__ ulong load8(ulong *p, int offset) {
	return *(ulong *)((char *)p + offset);
}
__device__ ulong load8_last(ulong *p, int offset) {
	return *(ulong *)((char *)p + offset);
}
__device__ ulonglong2 load16l(ulong *p, uint offset) {
	return *(ulonglong2 *)((char *)p + offset);
}

__device__ void store_ulong2(char *p, ulonglong2 store) {
	*(ulonglong2*)(p) = store;
}
__device__  void store_uint2(char* p, uint2 s0) {
	*(uint2*)(p) = s0;
}
__device__ void store4(char *p, uint store) {
	*(uint *)(p) = store;
}
__device__ void store_uint4(char* p, uint4 store0) {
	*(uint4 *)(p) = store0;
}
__device__ void store8(char *p, ulong store) {
	*(ulong *)(p) = store;
}


#endif
#endif


#define ASM_SHF_R_CLAMP32b_24(d,a,b,tmp) \
"shr.b32 " ## tmp ## "," ## a ## ",24;\n\t" \
"shl.b32 " ## d ## "," ## b ## ",8;\n\t" \
"add.u32 " ## d ## "," ## tmp ## "," ## d ## ";\n\t"

#define ASM_SHF_R_CLAMP32b_16(d,a,b,tmp) \
"shr.b32 " ## tmp ## "," ## a ## ",16;\n\t" \
"shl.b32 " ## d ## "," ## b ## ",16;\n\t" \
"add.u32 " ## d ## "," ## tmp ## "," ## d ## ";\n\t"


#define nv64to16(a,b,c,d,X) asm volatile( "mov.b64 {%0,%1,%2,%3}, %4; \n\t" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(X))


// Round 1
__device__ uint xor_and_store1(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	ulong xi0, xi1, xi2, xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD

	ulonglong2 loada, loadb;
	xi0 = load8(a++, 0) ^ load8(b++, 0);
	//	loada = *( ulong2 *)a;
	loada = load16l(a, 0);
	loadb = load16l(b, 0);
	xi1 = loada.x ^ loadb.x;
	xi2 = loada.y ^ loadb.y;


	/*
	xi0 = *(a++) ^ *(b++);
	xi1 = *(a++) ^ *(b++);
	xi2 = *a ^ *b;
	xi3 = 0;
	*/
	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);


	//256bit shift
	asm("{ .reg .b16 a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3; \n\t"
		"mov.b64 {a0,a1,a2,a3}, %4;\n\t"
		"mov.b64 {b0,b1,b2,b3}, %5;\n\t"
		"mov.b64 {c0,c1,c2,c3}, %6;\n\t"

		"mov.b64 %0, {a1,a2,a3,b0};\n\t"
		"mov.b64 %1, {b1,b2,b3,c0};\n\t"
		"mov.b64 %2, {c1,c2,c3,0};\n\t"
		"mov.b32 %3, {a0,a1};\n\t"
		"}\n" : "=l"(xi0), "=l"(xi1), "=l" (xi2), "=r"(_row) : "l"(xi0), "l"(xi1), "l"(xi2));


	//      row = get_row_nr_4((uint)xi0,round);	
	row = get_row_nr_4(_row, round);

	//        xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	//        xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	//        xi2 = (xi2 >> 16);

	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE
	//        *( uint *)(p - 4) = i;
	//        *( ulong *)(p + 0) = xi0;
	//	*( ulong *)(p + 8) = xi1;
	//	*( ulong *)(p + 16) = xi2;


	ulonglong2 store0;
	ulonglong2 store1;
	nv32to64(store0.x, 0, i);
	store0.y = xi0;
	//	*( ulong2 *)(pp)=store0;
	store_ulong2(pp, store0);
	store1.x = xi1;
	store1.y = xi2;
	//	*( ulong2 *)(pp+16)=store1;
	store_ulong2(pp + 16, store1);
	return 0;
}




// Round 2

__device__ uint xor_and_store2(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	ulong xi0, xi1, xi2, xi3;

	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD
	ulonglong2 loada, loadb;
	xi0 = load8(a++, 0) ^ load8(b++, 0);
	loada = load16l(a, 0);
	loadb = load16l(b, 0);
	xi1 = loada.x ^ loadb.x;
	xi2 = loada.y ^ loadb.y;


	/*
	xi0 = *(a++) ^ *(b++);
	xi1 = *(a++) ^ *(b++);
	xi2 = *a ^ *b;
	xi3 = 0;
	*/
	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);


	//256bit shift



	//7 op asm32 4 op + 3 op devectorize

	uint _xi0l, _xi0h, _xi1l, _xi1h, _xi2l, _xi2h;

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320

	asm("{\n\t"
		".reg .b32 a0,a1,b0,b1,c0,c1; \n\t"
		"mov.b64 {a0,a1}, %6;\n\t"
		"mov.b64 {b0,b1}, %7;\n\t"
		"mov.b64 {c0,c1}, %8;\n\t"

		"shr.b32 %5,a0,8;\n\t"
		"shf.r.clamp.b32 %0,a0,a1,24; \n\t"
		"shf.r.clamp.b32 %1,a1,b0,24; \n\t"
		"shf.r.clamp.b32 %2,b0,b1,24; \n\t"
		"shf.r.clamp.b32 %3,b1,c0,24; \n\t"
		"shf.r.clamp.b32 %4,c0,c1,24; \n\t"

		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h), "=r"(_xi2l), "=r"(_row) :
		"l"(xi0), "l"(xi1), "l"(xi2));
#else
	asm("{\n\t"
		".reg .b32 a0,a1,b0,b1,c0,c1,d0;\n\t"
		"mov.b64 {a0,a1}, %6;\n\t"
		"mov.b64 {b0,b1}, %7;\n\t"
		"mov.b64 {c0,c1}, %8;\n\t"
		"shr.b32 %5,a0,8;\n\t"
		ASM_SHF_R_CLAMP32b_24("%0", "a0", "a1", "d0")
		ASM_SHF_R_CLAMP32b_24("%1", "a1", "b0", "d0")
		ASM_SHF_R_CLAMP32b_24("%2", "b0", "b1", "d0")
		ASM_SHF_R_CLAMP32b_24("%3", "b1", "c0", "d0")
		ASM_SHF_R_CLAMP32b_24("%4", "c0", "c1", "d0")
		"}\n\t" : "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h), "=r"(_xi2l), "=r"(_row) :
		"l"(xi0), "l"(xi1), "l"(xi2));

#endif
#endif
	row = get_row_nr_4(_row, round);

	//	xi0 = (xi0 >> 24) | (xi1 << (64 - 24));
	//        xi1 = (xi1 >> 24) | (xi2 << (64 - 24));
	//        xi2 = (xi2 >> 24);
	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		//	*a+=load8_last(( ulong *)-1);
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE 11 op, asm 9 op, or 6op 32bit

	/*
	ulong s0;
	ulong2 store0;

	nv32to64(s0,i,_xi0l);
	nv32to64(store0.x,_xi0h,_xi1l);
	nv32to64(store0.y,_xi1h,_xi2l);
	*( ulong *)(p - 4)=s0;
	*( ulong2 *)(p + 4)=store0;
	*/

	uint2 s0;
	s0.x = i;
	s0.y = _xi0l;
	uint4 store0;
	store0.x = _xi0h;
	store0.y = _xi1l;
	store0.z = _xi1h;
	store0.w = _xi2l;
	//	*( uint2 *)(p - 4)=s0;
	store_uint2(p - 4, s0);
	//        *( uint4 *)(p + 4)=store0;
	store_uint4(p + 4, store0);
	/*
	*( uint *)(p - 4) = i;
	*( uint *)(p + 0) = xi0;
	*( ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*( ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
	*/
	return 0;
}

#define shuffle_rc32(d, a, b, c) d = (b << (32 - c)) | (a >> c);

//Round3
__device__ uint xor_and_store3(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	//	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD
	uint xi0l, xi0h, xi1l, xi1h, xi2l;
	xi0l = load4(a, 0) ^ load4(b, 0);

	if (!xi0l)
		return 0;


	ulong load1, load2;
	load1 = load8(a, 4) ^ load8(b, 4);
	load2 = load8_last(a, 12) ^ load8_last(b, 12);
	nv64to32(xi0h, xi1l, load1);
	nv64to32(xi1h, xi2l, load2);

	//     if(!xi0l )
	//	*a+=load8_last(( ulong *)-1);
	// xor 20 bytes
	//	xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
	//	xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
	//	xi2 = well_aligned_int(a, 16) ^ well_aligned_int(b, 16);
	//	ulong2 loada;
	//	ulong2 loadb;


	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	row = get_row_nr_4(xi0l, round);

	uint _xi0l, _xi0h, _xi1l, _xi1h;

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320
	asm("{\n\t"
		"shf.r.clamp.b32 %0,%4,%5,16; \n\t"
		"shf.r.clamp.b32 %1,%5,%6,16; \n\t"
		"shf.r.clamp.b32 %2,%6,%7,16; \n\t"
		"shf.r.clamp.b32 %3,%7,%8,16; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h), "r"(xi2l));
#else
	asm("{\n\t"
		".reg .b32 a0;\n\t"
		ASM_SHF_R_CLAMP32b_16("%0", "%4", "%5", "a0")
		ASM_SHF_R_CLAMP32b_16("%1", "%5", "%6", "a0")
		ASM_SHF_R_CLAMP32b_16("%2", "%6", "%7", "a0")
		ASM_SHF_R_CLAMP32b_16("%3", "%7", "%8", "a0")
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h), "r"(xi2l));
#endif
#endif

	//        xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	//        xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	//        xi2 = (xi2 >> 16);

	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		//	*a+=load8_last(( ulong *)-1);
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE
	ulong store0, store1;
	nv32to64(store0, i, _xi0l);
	nv32to64(store1, _xi0h, _xi1l);

	//        *( ulong *)(p - 4) = store0;
	store8(p - 4, store0);
	//        *( ulong *)(p + 4) = store1;
	store8(p + 4, store1);
	//        *( uint *)(p + 12) = _xi1h;
	store4(p + 12, _xi1h);

	/*
	*( uint *)(p - 4) = i;
	// store 16 bytes
	*( uint *)(p + 0) = xi0;
	*( ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*( uint *)(p + 12) = (xi1 >> 32);
	*/
	return 0;
}

// Round 4

__device__ uint xor_and_store4(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	ulong xi0, xi1, xi2, xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD

	//	xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
	//	xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);


	uint xi0l, xi0h, xi1l, xi1h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	if (!xi0l)
		return 0;
	xi0h = load4(a, 4) ^ load4(b, 4);
	xi1l = load4(a, 8) ^ load4(b, 8);
	xi1h = load4_last(a, 12) ^ load4_last(b, 12);


	//	xi2 = 0;

	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);

	uint _xi0l, _xi0h, _xi1l, _xi1h, _xi2l, _xi2h;
	//256bit shift
#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320
	asm("{\n\t"
		"shf.r.clamp.b32 %0,%4,%5,24; \n\t"
		"shf.r.clamp.b32 %1,%5,%6,24; \n\t"
		"shf.r.clamp.b32 %2,%6,%7,24; \n\t"
		"shr.b32         %3,%7,24; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h));
#else
	asm("{\n\t"
		".reg .b32 a0;\r\n"
		ASM_SHF_R_CLAMP32b_24("%0", "%4", "%5", "a0")
		ASM_SHF_R_CLAMP32b_24("%1", "%5", "%6", "a0")
		ASM_SHF_R_CLAMP32b_24("%2", "%6", "%7", "a0")
		"shr.b32         %3,%7,24; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h));
#endif
#endif
	row = get_row_nr_4(xi0l >> 8, round);

	//            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
	//	    xi1 = (xi1 >> 8);

	//row = get_row_nr_4((uint)xi0,round);	
	//	row = get_row_nr_4(_row,round);

	//       xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	//       xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	//       xi2 = (xi2 >> 16);

	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE

	//*( uint *)(p - 4) = i;
	store4(p - 4, i);
	//*( ulong *)(p + 0) = xi0;
	//*( ulong *)(p + 8) = xi1;
	uint4 store;
	store.x = _xi0l;
	store.y = _xi0h;
	store.z = _xi1l;
	store.w = _xi1h;
	//*( uint4 *)(p + 0) = store;
	store_uint4(p + 0, store);
	return 0;
}


// Round 5

__device__ uint xor_and_store5(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	//ulong xi0, xi1, xi2, xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD

	//	xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
	//	xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);

	uint xi0l, xi0h, xi1l, xi1h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	if (!xi0l)
		return 0;
	xi0h = load4(a, 4) ^ load4(b, 4);
	xi1l = load4(a, 8) ^ load4(b, 8);
	xi1h = load4_last(a, 12) ^ load4_last(b, 12);


	//	xi2 = 0;

	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);

	uint _xi0l, _xi0h, _xi1l, _xi1h, _xi2l, _xi2h;

	//256bit shift
#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320

	asm("{\n\t"
		"shf.r.clamp.b32 %0,%4,%5,16; \n\t"
		"shf.r.clamp.b32 %1,%5,%6,16; \n\t"
		"shf.r.clamp.b32 %2,%6,%7,16; \n\t"
		"shr.b32         %3,%7,16; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h));
#else
	asm("{\n\t"
		".reg .b32 a0;\n\t"
		ASM_SHF_R_CLAMP32b_16("%0", "%4", "%5", "a0")
		ASM_SHF_R_CLAMP32b_16("%1", "%5", "%6", "a0")
		ASM_SHF_R_CLAMP32b_16("%2", "%6", "%7", "a0")
		"shr.b32         %3,%7,16; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l), "=r"(_xi1h) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h));

#endif
#endif

	row = get_row_nr_4(xi0l, round);

	//            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
	//	    xi1 = (xi1 >> 8);

	//row = get_row_nr_4((uint)xi0,round);	
	//	row = get_row_nr_4(_row,round);

	//       xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
	//       xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
	//       xi2 = (xi2 >> 16);

	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE

	//*( uint *)(p - 4) = i;
	store4(p - 4, i);
	//*( ulong *)(p + 0) = xi0;
	//*( ulong *)(p + 8) = xi1;
	uint4 store;
	store.x = _xi0l;
	store.y = _xi0h;
	store.z = _xi1l;
	store.w = _xi1h;
	//*( uint4 *)(p + 0) = store;
	store_uint4(p + 0, store);
	return 0;
}

// Round 6
__device__ uint xor_and_store6(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	ulong xi0;// , xi1, xi2, xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD
	uint xi0l, xi0h, xi1l;

	xi0 = load8(a++, 0) ^ load8(b++, 0);

	if (!xi0)
		return 0;
	xi1l = load4_last(a, 0) ^ load4_last(b, 0);

	nv64to32(xi0l, xi0h, xi0);

	//	xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
	//	xi1 = (xi1 >> 8);

	//	xi2 = 0;

	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);


	//256bit shift

	uint _xi0l, _xi0h, _xi1l, _xi1h;

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320

	asm("{\n\t"
		"shf.r.clamp.b32 %0,%3,%4,24; \n\t"
		"shf.r.clamp.b32 %1,%4,%5,24; \n\t"
		"shr.b32         %2,%5,24; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l));
#else
	asm("{\n\t"
		".reg .b32 a0;\n\t"
		ASM_SHF_R_CLAMP32b_24("%0", "%3", "%4", "a0")
		ASM_SHF_R_CLAMP32b_24("%1", "%4", "%5", "a0")
		"shr.b32         %2,%5,24; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h), "=r"(_xi1l) :
		"r"(xi0l), "r"(xi0h), "r"(xi1l));
#endif
#endif

	row = get_row_nr_4(xi0l >> 8, round);


	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE

	//	*( uint *)(p - 4) = i;
	ulong store;
	nv32to64(store, i, _xi0l);
	store8(p - 4, store);
	// *( ulong *)(p - 4)= store;
	//	*( uint *)(p + 0) = _xi0l;
	//	*( uint *)(p + 4) = _xi0h;
	store4(p + 4, _xi0h);
	return 0;
}


// Round 7

__device__ uint xor_and_store7(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	//ulong xi0, xi1, xi2, xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD

	uint xi0l, xi0h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	if (!xi0l)
		return 0;
	xi0h = load4_last(a, 4) ^ load4_last(b, 4);
	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);


	//256bit shift
	row = get_row_nr_4(xi0l, round);

	uint _xi0l, _xi0h;

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320

	asm("{\n\t"
		"shf.r.clamp.b32 %0,%2,%3,16; \n\t"
		"shr.b32         %1,%3,16; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h) :
		"r"(xi0l), "r"(xi0h));
	//
#else
	asm("{\n\t"
		".reg .b32 a0;\n\t"
		ASM_SHF_R_CLAMP32b_16("%0", "%2", "%3", "a0")
		"shr.b32         %1,%3,16; \n\t"
		"}\n\t"
		: "=r"(_xi0l), "=r"(_xi0h) :
		"r"(xi0l), "r"(xi0h));

#endif
#endif


	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS)
	{
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE

	uint2 store;
	store.x = i;
	store.y = _xi0l;
	//	*( uint2 *)(p - 4) = store;
	store_uint2(p - 4, store);
	//	*( uint *)(p + 0) = _xi0l;
	//	*( uint *)(p + 4) = _xi0h;
	store4(p + 4, _xi0h);
	return 0;
}

// Round 8

__device__ uint xor_and_store8(uint round, char *ht_dst, uint x_row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters) {

	ulong xi0, xi1, xi2, xi3;
	uint _row;
	uint row;
	char       *p;
	uint                cnt;
	//LOAD
	uint xi0l, xi0h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	if (!xi0l)
		return 0;
	xi0h = load4_last(a, 4) ^ load4_last(b, 4);
	//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	//256bit shift
	row = get_row_nr_4(xi0l >> 8, round);
	uint _xi0l, _xi0h, _xi1l, _xi1h;

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 320
	asm("{\n\t"
		"shf.r.clamp.b32 %0,%1,%2,24; \n\t"
		"}\n\t"
		: "=r"(_xi0l) :
		"r"(xi0l), "r"(xi0h));
#else
	asm("{\n\t"
		".reg .b32 a0;\n\t"
		ASM_SHF_R_CLAMP32b_24("%0", "%1","%2", "a0")
		"}\n\t"
		: "=r"(_xi0l) :
		"r"(xi0l), "r"(xi0h));

#endif
#endif
	//

	p = ht_dst + row * NR_SLOTS * SLOT_LEN;
	uint rowIdx = row / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
	uint xcnt = atomicAdd(rowCounters + rowIdx, 1 << rowOffset);
	xcnt = (xcnt >> rowOffset) & ROW_MASK;
	cnt = xcnt;
	if (cnt >= NR_SLOTS) {
		// avoid overflows
		atomicSub(rowCounters + rowIdx, 1 << rowOffset);
		return 1;
	}
	char       *pp = p + cnt * SLOT_LEN;
	p = pp + xi_offset_for_round(round);
	//

	//STORE

	//	uint2 store;
	//	store.x=i;
	//	store.y=_xi0l;

	//	*( uint *)(p - 4) = i;
	//	*( uint *)(p + 0) = _xi0l;
	store4(p - 4, i);
	store4(p + 0, _xi0l);
	return 0;
}

__device__ uint xor_and_store(uint round, char *ht_dst, uint row,
	uint slot_a, uint slot_b, ulong *a, ulong *b,
	uint *rowCounters)
{
	if (round == 1)
		return xor_and_store1(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 2)
		return xor_and_store2(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 3)
		return xor_and_store3(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 4)
		return xor_and_store4(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 5)
		return xor_and_store5(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 6)
		return xor_and_store6(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 7)
		return xor_and_store7(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
	else if (round == 8)
		return xor_and_store8(round, ht_dst, row, slot_a, slot_b, a, b, rowCounters);
}


/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/
__device__ void equihash_round_cm3(uint round, char *ht_src, char *ht_dst, uint *rowCountersSrc,
	uint *rowCountersDst)
{
	uint        tid =  blockIdx.x * blockDim.x + threadIdx.x;
	uint		tlid = threadIdx.x;
	char       *p;
	uint                cnt;
	uchar		first_words[NR_SLOTS];
	uchar		mask;
	uint                i, j;
	// NR_SLOTS is already oversized (by a factor of OVERHEAD), but we want to
	// make it even larger
	ushort		collisions[NR_SLOTS * 3];
	uint                nr_coll = 0;
	uint                n;
	uint		dropped_coll = 0;
	uint		dropped_stor = 0;
	ulong      *a, *b;
	uint		xi_offset;
	// read first words of Xi from the previous (round - 1) hash table
	xi_offset = xi_offset_for_round(round - 1);
	// the mask is also computed to read data from the previous round
#if NR_ROWS_LOG == 16
	mask = ((!(round % 2)) ? 0x0f : 0xf0);
#elif NR_ROWS_LOG == 18
	mask = ((!(round % 2)) ? 0x03 : 0x30);
#elif NR_ROWS_LOG == 19
	mask = ((!(round % 2)) ? 0x01 : 0x10);
#elif NR_ROWS_LOG == 20
	mask = 0; /* we can vastly simplify the code below */
#else
#error "unsupported NR_ROWS_LOG"
#endif
	p = (ht_src + tid * NR_SLOTS * SLOT_LEN);

	uint rowIdx = tid / ROWS_PER_UINT;
	uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
	cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
	cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
	if (!cnt)
		// no elements in row, no collisions
		return;
#if NR_ROWS_LOG != 20 || !OPTIM_SIMPLIFY_ROUND
	p += xi_offset;
	for (i = 0; i < cnt; i++, p += SLOT_LEN)
		first_words[i] = *(uchar *)p;
#endif
	// find collisions
	for (i = 0; i < cnt; i++)
		for (j = i + 1; j < cnt; j++)
#if NR_ROWS_LOG != 20 || !OPTIM_SIMPLIFY_ROUND
			if ((first_words[i] & mask) ==
				(first_words[j] & mask))
			{
				// collision!
				if (nr_coll >= sizeof(collisions) / sizeof(*collisions))
					dropped_coll++;
				else
#if NR_SLOTS <= (1 << 8)
					// note: this assumes slots can be encoded in 8 bits
					collisions[nr_coll++] =
					((ushort)j << 8) | ((ushort)i & 0xff);
#else
#error "unsupported NR_SLOTS"
#endif
			}
	// XOR colliding pairs of Xi
	for (n = 0; n < nr_coll; n++)
	{
		i = collisions[n] & 0xff;
		j = collisions[n] >> 8;
#else
		{
#endif
			a = (ulong *)
				(ht_src + tid * NR_SLOTS * SLOT_LEN + i * SLOT_LEN + xi_offset);
			b = (ulong *)
				(ht_src + tid * NR_SLOTS * SLOT_LEN + j * SLOT_LEN + xi_offset);
			dropped_stor += xor_and_store(round, ht_dst, tid, i, j, a, b, rowCountersDst);
	}
}

#ifdef  __CUDACC__
#if __CUDA_ARCH__ >= 320
/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/
__device__ void equihash_round(uint round,
	char *ht_src,
	char *ht_dst,
	uint *collisionsData,
	uint *collisionsNum,
	uint *rowCountersSrc,
	uint *rowCountersDst)
{
	uint globalTid = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_ROW;
	uint localRowIdx = threadIdx.x / THREADS_PER_ROW;
	uint localTid = threadIdx.x % THREADS_PER_ROW;
	__shared__ uint slotCountersData[COLLISION_TYPES_NUM*ROWS_PER_WORKGROUP];
	__shared__ ushort slotsData[COLLISION_TYPES_NUM*COLLISION_BUFFER_SIZE*ROWS_PER_WORKGROUP];

	uint *slotCounters = &slotCountersData[COLLISION_TYPES_NUM*localRowIdx];
	ushort *slots = &slotsData[COLLISION_TYPES_NUM*COLLISION_BUFFER_SIZE*localRowIdx];

	char *p;
	uint    cnt;
	uchar   mask;
	uint    shift;
	uint    i, j;
	// NR_SLOTS is already oversized (by a factor of OVERHEAD), but we want to
	// make it even larger
	uint    n;
	uint    dropped_coll = 0;
	uint    dropped_stor = 0;
	ulong  *a, *b;
	uint    xi_offset;
	// read first words of Xi from the previous (round - 1) hash table
	xi_offset = xi_offset_for_round(round - 1);
	// the mask is also computed to read data from the previous round
#if NR_ROWS_LOG <= 16
	mask = ((!(round % 2)) ? 0x0f : 0xf0);
	shift = ((!(round % 2)) ? 0 : 4);
#elif NR_ROWS_LOG == 18
	mask = ((!(round % 2)) ? 0x03 : 0x30);
	shift = ((!(round % 2)) ? 0 : 4);
#elif NR_ROWS_LOG == 19
	mask = ((!(round % 2)) ? 0x01 : 0x10);
	shift = ((!(round % 2)) ? 0 : 4);
#elif NR_ROWS_LOG == 20
	mask = 0; /* we can vastly simplify the code below */
	shift = 0;
#else
#error "unsupported NR_ROWS_LOG"
#endif    

	for (uint chunk = 0; chunk < THREADS_PER_ROW; chunk++) {
		uint tid = globalTid + NR_ROWS / THREADS_PER_ROW*chunk;
		uint gid = tid & ~(ROWS_PER_WORKGROUP - 1);

		uint rowIdx = tid / ROWS_PER_UINT;
		uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
		cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round

		*collisionsNum = 0;
		p = (ht_src + tid * NR_SLOTS * SLOT_LEN);
		p += xi_offset;
		p += SLOT_LEN*localTid;

		for (i = threadIdx.x; i < COLLISION_TYPES_NUM*ROWS_PER_WORKGROUP; i += blockDim.x)
			slotCountersData[i] = 0;
		
		__syncthreads();

		for (i = localTid; i < cnt; i += THREADS_PER_ROW, p += SLOT_LEN*THREADS_PER_ROW) {
			uchar x = (uchar)(load4((ulong *)p, 0) & mask) >> shift;
			uint slotIdx = atomicAdd(&slotCounters[x], 1);
			slotIdx = min(slotIdx, COLLISION_BUFFER_SIZE - 1);
			slots[COLLISION_BUFFER_SIZE*x + slotIdx] = i;
		}

		__syncthreads();

		const uint ct_groupsize = max(1u, THREADS_PER_ROW / COLLISION_TYPES_NUM);
		for (uint collTypeIdx = localTid / ct_groupsize; collTypeIdx < COLLISION_TYPES_NUM; collTypeIdx += THREADS_PER_ROW / ct_groupsize) {
			const uint N = min((uint)slotCounters[collTypeIdx], COLLISION_BUFFER_SIZE);
			for (uint i = 0; i < N; i++) {
				uint collision = (localRowIdx << 24) | (slots[COLLISION_BUFFER_SIZE*collTypeIdx + i] << 12);
				for (uint j = i + 1 + localTid % ct_groupsize; j < N; j += ct_groupsize) {
					uint index = atomicAdd(collisionsNum, 1);
					index = min(index, (uint)(LDS_COLL_SIZE - 1));
					collisionsData[index] = collision | slots[COLLISION_BUFFER_SIZE*collTypeIdx + j];
				}
			}
		}

		__syncthreads();
		uint totalCollisions = *collisionsNum;
		totalCollisions = min(totalCollisions, (uint)LDS_COLL_SIZE);
		for (uint index = threadIdx.x; index < totalCollisions; index += blockDim.x)
		{
			uint collision = collisionsData[index];
			uint collisionThreadId = gid + (collision >> 24);
			uint i = (collision >> 12) & 0xFFF;
			uint j = collision & 0xFFF;
			uchar *ptr = (uchar*)ht_src + collisionThreadId * NR_SLOTS * SLOT_LEN +
				xi_offset;
			a = (ulong *)(ptr + i * SLOT_LEN);
			b = (ulong *)(ptr + j * SLOT_LEN);
			dropped_stor += xor_and_store(round, ht_dst, collisionThreadId, i, j,
				a, b, rowCountersDst);
		}
	}

}


/*
** This defines kernel_round1, kernel_round2, ..., kernel_round7.
*/
#define KERNEL_ROUND_ODD(N) \
__global__  \
void kernel_round ## N( char *ht_src,  char *ht_dst) \
{ \
    __shared__ uint    collisionsData[LDS_COLL_SIZE]; \
    __shared__ uint    collisionsNum; \
    equihash_round(N, ht_src, ht_dst, collisionsData, \
	    &collisionsNum, (uint*)rowCounter0, (uint*)rowCounter1); \
}

#define KERNEL_ROUND_EVEN(N) \
__global__  \
void kernel_round ## N( char *ht_src,  char *ht_dst) \
{ \
    __shared__ uint    collisionsData[LDS_COLL_SIZE]; \
    __shared__ uint    collisionsNum; \
    equihash_round(N, ht_src, ht_dst, collisionsData, \
	    &collisionsNum, (uint*)rowCounter1, (uint*)rowCounter0); \
}



KERNEL_ROUND_ODD(1)
KERNEL_ROUND_EVEN(2)
KERNEL_ROUND_ODD(3)
KERNEL_ROUND_EVEN(4)
KERNEL_ROUND_ODD(5)
KERNEL_ROUND_EVEN(6)
KERNEL_ROUND_ODD(7)


// kernel_round8 takes an extra argument, "sols"
__global__
void kernel_round8(char *ht_src, char *ht_dst)
{
	uint		tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uint	collisionsData[LDS_COLL_SIZE];
	__shared__ uint	collisionsNum;
	equihash_round(8, ht_src, ht_dst, collisionsData, &collisionsNum, (uint*)rowCounter1, (uint*)rowCounter0);
	if (!tid)
		sols.nr = sols.likely_invalids = 0;
}

#else

#define kernel_round1 kernel_round_cm3_1
#define kernel_round2 kernel_round_cm3_2
#define kernel_round3 kernel_round_cm3_3
#define kernel_round4 kernel_round_cm3_4
#define kernel_round5 kernel_round_cm3_5
#define kernel_round6 kernel_round_cm3_6
#define kernel_round7 kernel_round_cm3_7
#define kernel_round8 kernel_round_cm3_8

#endif
#endif

#define KERNEL_ROUND_ODD_OLD(N) \
__global__  \
void kernel_round_cm3_ ## N( char *ht_src,  char *ht_dst) \
{ \
    equihash_round_cm3(N, ht_src, ht_dst, (uint*)rowCounter0, (uint*)rowCounter1); \
}

#define KERNEL_ROUND_EVEN_OLD(N) \
__global__  \
void kernel_round_cm3_ ## N(char *ht_src,  char *ht_dst) \
{ \
    equihash_round_cm3(N, ht_src, ht_dst, (uint*)rowCounter1, (uint*)rowCounter0); \
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
	equihash_round_cm3(8, ht_src, ht_dst, (uint*)rowCounter1, (uint*)rowCounter0);
	if (!tid) {
		sols.nr = sols.likely_invalids = 0;
	}
}

__device__ uint expand_ref(char *ht, uint xi_offset, uint row, uint slot)
{
	return load4((ulong *)(ht + row * NR_SLOTS * SLOT_LEN +
		slot * SLOT_LEN + xi_offset - 4), 0);
}

/*
** Expand references to inputs. Return 1 if so far the solution appears valid,
** or 0 otherwise (an invalid solution would be a solution with duplicate
** inputs, which can be detected at the last step: round == 0).
*/
__device__ uint expand_refs(uint *ins, uint nr_inputs, char **htabs,
	uint round)
{
	char	*ht = htabs[round % 2];
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

constexpr uint c_kernel_sol_counters = 32768 * (THRD / THREADS_PER_ROW);
__device__ uint kernel_sol_counters[c_kernel_sol_counters];

/*
** Scan the hash tables to find Equihash solutions.
*/
__global__
void kernel_sols_cm3(char *ht0, char *ht1)
{
	uint *counters = &kernel_sol_counters[blockIdx.x * (THRD / THREADS_PER_ROW)];
	__shared__ uint refs[NR_SLOTS*(THRD / THREADS_PER_ROW)];
	__shared__ uint data[NR_SLOTS*(THRD / THREADS_PER_ROW)];
	__shared__ uint collisionsNum;
	__shared__ ulong collisions[THRD * 4];

	uint globalTid = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_ROW;
	uint localTid = threadIdx.x / THREADS_PER_ROW;
	uint localGroupId = threadIdx.x % THREADS_PER_ROW;
	uint *refsPtr = &refs[NR_SLOTS*localTid];
	uint *dataPtr = &data[NR_SLOTS*localTid];

	char	*htabs[2] = { ht0, ht1 };
	char	*hcounters[2] = { rowCounter0, rowCounter1 };
	uint		ht_i = (PARAM_K - 1) % 2; // table filled at last round
	uint		cnt;
	uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
	uint		i, j;
	char	*p;
	uint		ref_i, ref_j;
	// it's ok for the collisions array to be so small, as if it fills up
	// the potential solutions are likely invalid (many duplicate inputs)
	//     ulong		collisions;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
	// in the final hash table, we are looking for a match on both the bits
	// part of the previous PREFIX colliding bits, and the last PREFIX bits.
	uint		mask = 0xffffff;
#else
#error "unsupported NR_ROWS_LOG"
#endif

	collisionsNum = 0;

	for (uint chunk = 0; chunk < THREADS_PER_ROW; chunk++) {
		uint tid = globalTid + NR_ROWS / THREADS_PER_ROW * chunk;
		p = htabs[ht_i] + tid * NR_SLOTS * SLOT_LEN;
		uint rowIdx = tid / ROWS_PER_UINT;
		uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
		cnt = (((uint*)rowCounter0)[rowIdx] >> rowOffset) & ROW_MASK;
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
		p += xi_offset;
		p += SLOT_LEN*localGroupId;

		for (i = threadIdx.x; i < THRD / THREADS_PER_ROW; i += blockDim.x)
			counters[i] = 0;
		for (i = localGroupId; i < cnt; i += THREADS_PER_ROW, p += SLOT_LEN*THREADS_PER_ROW) {
			//refsPtr[i] = *( uint *)(p - 4);
			//dataPtr[i] = (*( uint *)p) & mask;
			refsPtr[i] = load4((ulong *)(p), -4);
			dataPtr[i] = load4((ulong *)(p), 0) & mask;
		}
		__syncthreads();

		for (i = 0; i < cnt; i++)
		{
			uint a_data = dataPtr[i];
			ref_i = refsPtr[i];
			for (j = i + 1 + localGroupId; j < cnt; j += THREADS_PER_ROW)
			{
				if (a_data == dataPtr[j])
				{
					if (atomicAdd(&counters[localTid], 1) == 0)
						collisions[atomicAdd(&collisionsNum, 1)] = ((ulong)ref_i << 32) | refsPtr[j];
					goto part2;
				}
			}
		}

	part2:
		continue;
	}

	__syncthreads();
	uint totalCollisions = collisionsNum;
	if (threadIdx.x < totalCollisions) {
		ulong coll = collisions[threadIdx.x];
		potential_sol(htabs, coll >> 32, coll & 0xffffffff);
	}
}


/*
** Scan the hash tables to find Equihash solutions.
*/
__global__
void kernel_sols(char *ht0, char *ht1)
{
	__shared__ uint counters[THRD / THREADS_PER_ROW];
	__shared__ uint refs[NR_SLOTS*(THRD / THREADS_PER_ROW)];
	__shared__ uint data[NR_SLOTS*(THRD / THREADS_PER_ROW)];
	__shared__ uint collisionsNum;
	__shared__ ulong collisions[THRD * 4];

	uint globalTid = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_ROW;
	uint localTid = threadIdx.x / THREADS_PER_ROW;
	uint localGroupId = threadIdx.x % THREADS_PER_ROW;
	uint *refsPtr = &refs[NR_SLOTS*localTid];
	uint *dataPtr = &data[NR_SLOTS*localTid];

	char	*htabs[2] = { ht0, ht1 };
	char	*hcounters[2] = { rowCounter0, rowCounter1 };
	uint		ht_i = (PARAM_K - 1) % 2; // table filled at last round
	uint		cnt;
	uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
	uint		i, j;
	char	*p;
	uint		ref_i, ref_j;
	// it's ok for the collisions array to be so small, as if it fills up
	// the potential solutions are likely invalid (many duplicate inputs)
	//     ulong		collisions;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
	// in the final hash table, we are looking for a match on both the bits
	// part of the previous PREFIX colliding bits, and the last PREFIX bits.
	uint		mask = 0xffffff;
#else
#error "unsupported NR_ROWS_LOG"
#endif

	collisionsNum = 0;

	for (uint chunk = 0; chunk < THREADS_PER_ROW; chunk++) {
		uint tid = globalTid + NR_ROWS / THREADS_PER_ROW*chunk;
		p = htabs[ht_i] + tid * NR_SLOTS * SLOT_LEN;
		uint rowIdx = tid / ROWS_PER_UINT;
		uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
		cnt = (((uint*)rowCounter0)[rowIdx] >> rowOffset) & ROW_MASK;
		cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
		p += xi_offset;
		p += SLOT_LEN*localGroupId;

		for (i = threadIdx.x; i < THRD / THREADS_PER_ROW; i += blockDim.x)
			counters[i] = 0;
		for (i = localGroupId; i < cnt; i += THREADS_PER_ROW, p += SLOT_LEN*THREADS_PER_ROW) {
			//refsPtr[i] = *( uint *)(p - 4);
			//dataPtr[i] = (*( uint *)p) & mask;
			refsPtr[i] = load4((ulong *)(p), -4);
			dataPtr[i] = load4((ulong *)(p), 0) & mask;
		}
		__syncthreads();

		for (i = 0; i < cnt; i++)
		{
			uint a_data = dataPtr[i];
			ref_i = refsPtr[i];
			for (j = i + 1 + localGroupId; j < cnt; j += THREADS_PER_ROW)
			{
				if (a_data == dataPtr[j])
				{
					if (atomicAdd(&counters[localTid], 1) == 0)
						collisions[atomicAdd(&collisionsNum, 1)] = ((ulong)ref_i << 32) | refsPtr[j];
					goto part2;
				}
			}
		}

	part2:
		continue;
	}

	__syncthreads();
	uint totalCollisions = collisionsNum;
	if (threadIdx.x < totalCollisions) {
		ulong coll = collisions[threadIdx.x];
		potential_sol(htabs, coll >> 32, coll & 0xffffffff);
	}
}

struct __align__(64) c_context {
	char* buf_ht[2], *buf_dbg;
	//uint *rowCounters[2];
	sols_t	*sols;
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
		32;
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
	checkCudaErrors(cudaMallocHost(&eq->sols, sizeof(*eq->sols)));
	checkCudaErrors(cudaDeviceSynchronize());
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
	constexpr uint32_t THREAD_SHIFT = 7;
	constexpr uint32_t THREAD_COUNT = 1 << THREAD_SHIFT;
	constexpr uint32_t DIM_SIZE = NR_ROWS >> THREAD_SHIFT;

	constexpr uint32_t INIT_THREADS = 256;
	constexpr uint32_t INIT_DIM = NR_ROWS / ROWS_PER_UINT / INIT_THREADS;

	constexpr uint32_t ROUND_THREADS = THRD;
	constexpr uint32_t ROUND_DIM = NR_ROWS / ROUND_THREADS;
	static uint32_t ROUND0_DIM = select_work_size_blake() / ROUND_THREADS;

	// Now on every round!!!!
	switch (round) {
	case 0:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round0 << <ROUND0_DIM, ROUND_THREADS >> >(miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 1:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round1 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 2:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round2 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 3:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round3 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 4:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round4 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 5:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round5 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 6:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round6 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 7:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round7 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 8:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round8 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	}
}

static inline void solve_old(unsigned round, c_context *miner)
{
	constexpr uint32_t THREAD_SHIFT = 7;
	constexpr uint32_t THREAD_COUNT = 1 << THREAD_SHIFT;
	constexpr uint32_t DIM_SIZE = NR_ROWS >> THREAD_SHIFT;

	constexpr uint32_t INIT_DIM = NR_ROWS / ROWS_PER_UINT / 256;
	constexpr uint32_t INIT_THREADS = 256;
	
	constexpr uint32_t ROUND_THREADS = THRD;
	constexpr uint32_t ROUND_DIM = NR_ROWS / ROUND_THREADS;
	static uint32_t ROUND0_DIM = select_work_size_blake() / ROUND_THREADS;
	
	switch (round) {
	case 0:
		kernel_init_ht0<<<INIT_DIM, INIT_THREADS >> > ();
		kernel_round0<<<ROUND0_DIM, ROUND_THREADS >> >(miner->buf_ht[round & 1], (uint*)miner->buf_dbg);
		break;
	case 1:
		kernel_init_ht1<<<INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_1<< <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;

	case 2:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_2 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 3:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > (); 
		kernel_round_cm3_3 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 4:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_4 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 5:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_5 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 6:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_6 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 7:
		kernel_init_ht1 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_7 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
		break;
	case 8:
		kernel_init_ht0 << <INIT_DIM, INIT_THREADS >> > ();
		kernel_round_cm3_8 << <ROUND_DIM, ROUND_THREADS >> >(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1]);
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

	if (bUseOld) {
		kernel_sols_cm3 << <NR_ROWS / THRD, THRD >> > (miner->buf_ht[0], miner->buf_ht[1]);
	} else {
		kernel_sols << <NR_ROWS / THRD, THRD >> > (miner->buf_ht[0], miner->buf_ht[1]);
	}

	checkCudaErrors(cudaMemcpyFromSymbol(eq->sols, sols, sizeof(sols_t), 0, cudaMemcpyDeviceToHost));

	if (eq->sols->nr > MAX_SOLS)
		eq->sols->nr = MAX_SOLS;

	for (unsigned sol_i = 0; sol_i < eq->sols->nr; sol_i++) {
		verify_sol(eq->sols, sol_i);
	}

	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < eq->sols->nr; i++) {
		if (eq->sols->valid[i]) {
			compress(proof, (uint32_t *)(eq->sols->values[i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}
	}
	hashdonef();
}