// Gateless Gate, a Zcash miner
// Copyright 2016 zawawa @ bitcointalk.org
//
// The initial version of this software was based on:
// SILENTARMY v5
// The MIT License (MIT) Copyright (c) 2016 Marc Bevand, Genoil
//
// This program is free software : you can redistribute it and / or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.If not, see <http://www.gnu.org/licenses/>.

//#define ENABLE_DEBUG

//
// Parameters for Hash Tables
//

// There are PARAM_K - 1 hash tables, and each hash table has NR_ROWS rows.
// Each row contains NR_SLOTS slots.

#define NR_ROWS_LOG            14  // 12, 13, 14, 15, or 16. 12 and 13 are not practically usable.
#define NR_SLOTS               199 // Prime numbers are preferable.
#define LOCAL_WORK_SIZE        64  
#define THREADS_PER_ROW        64
#define LOCAL_WORK_SIZE_SOLS   64
#define THREADS_PER_ROW_SOLS   64
#define GLOBAL_WORK_SIZE_RATIO 512 // global_work_size = GLOBAL_WORK_SIZE_RATIO * nr_compute_units * LOCAL_WORK_SIZE
#define THREADS_PER_WRITE      1  // 1, 2, 4, or 8
#define SLOT_CACHE_SIZE        (NR_SLOTS * ROWS_IN_WORK_ITEM)
#define LDS_COLL_SIZE          (NR_SLOTS * ROWS_IN_WORK_ITEM * 140 / 100)
#define BIN_SIZE               (NR_SLOTS * 6 / 100)
#define EXTRA_BITS_FOR_BINS_SOLS 1
#define BIN_SIZE_SOLS          ((BIN_SIZE >> EXTRA_BITS_FOR_BINS_SOLS) * 250 / 100)

#define PARAM_N				   200
#define PARAM_K			       9
#define PREFIX                 (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS              (1 << PREFIX)
#define NR_ROWS                         (1 << NR_ROWS_LOG)
// Length of 1 element (slot) in byte
#define SLOT_LEN                        32
// Total size of hash table
#define HT_SIZE				(NR_ROWS * NR_SLOTS * SLOT_LEN)
// Length of Zcash block header, nonce (part of header)
#define ZCASH_BLOCK_HEADER_LEN		140
// Offset of nTime in header
#define ZCASH_BLOCK_OFFSET_NTIME        (4 + 3 * 32)
// Length of nonce
#define ZCASH_NONCE_LEN			32
// Length of encoded representation of solution size
#define ZCASH_SOLSIZE_LEN		3
// Solution size (1344 = 0x540) represented as a compact integer, in hex
#define ZCASH_SOLSIZE_HEX               "fd4005"
// Length of encoded solution (512 * 21 bits / 8 = 1344 bytes)
#define ZCASH_SOL_LEN                   ((1 << PARAM_K) * (PREFIX + 1) / 8)
// Last N_ZERO_BYTES of nonce must be zero due to my BLAKE2B optimization
#define N_ZERO_BYTES			12
// Number of bytes Zcash needs out of Blake
#define ZCASH_HASH_LEN                  50
// Number of wavefronts per SIMD for the Blake kernel.
// Blake is ALU-bound (beside the atomic counter being incremented) so we need
// at least 2 wavefronts per SIMD to hide the 2-clock latency of integer
// instructions. 10 is the max supported by the hw.
#define BLAKE_WPS               	10
// Maximum number of solutions reported by kernel to host
#define MAX_SOLS			11
// Length of SHA256 target
#define SHA256_TARGET_LEN               (256 / 8)

#if (NR_SLOTS < 3)
#define BITS_PER_ROW 2
#define ROWS_PER_UINT 16
#define ROW_MASK 0x03
#elif (NR_SLOTS < 7)
#define BITS_PER_ROW 3
#define ROWS_PER_UINT 10
#define ROW_MASK 0x07
#elif (NR_SLOTS < 15)
#define BITS_PER_ROW 4
#define ROWS_PER_UINT 8
#define ROW_MASK 0x0F
#elif (NR_SLOTS < 31)
#define BITS_PER_ROW 5
#define ROWS_PER_UINT 6
#define ROW_MASK 0x1F
#elif (NR_SLOTS < 63)
#define BITS_PER_ROW 6
#define ROWS_PER_UINT 5
#define ROW_MASK 0x3F
#elif (NR_SLOTS < 255)
#define BITS_PER_ROW 8
#define ROWS_PER_UINT 4
#define ROW_MASK 0xFF
#else
#define BITS_PER_ROW 16
#define ROWS_PER_UINT 2
#define ROW_MASK 0xFFFF
#endif
#define RC_SIZE (NR_ROWS * 4 / ROWS_PER_UINT)

/*
** Return the offset of Xi in bytes from the beginning of the slot.
*/
#define xi_offset_for_round(round)	4

// An (uncompressed) solution stores (1 << PARAM_K) 32-bit values
#define SOL_SIZE			((1 << PARAM_K) * 4)
typedef struct	sols_s
{
	uint	nr;
	uint	likely_invalids;
	uchar	valid[MAX_SOLS];
	uint	values[MAX_SOLS][(1 << PARAM_K)];
}		sols_t;

#if NR_ROWS_LOG <= 12 && NR_SLOTS <= (1 << 10)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 20) | ((slot1 & 0x3ff) << 10) | (slot0 & 0x3ff))
#define DECODE_ROW(REF)   (REF >> 20)
#define DECODE_SLOT1(REF) ((REF >> 10) & 0x3ff)
#define DECODE_SLOT0(REF) (REF & 0x3ff)

#elif NR_ROWS_LOG <= 14 && NR_SLOTS <= (1 << 9)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 18) | ((slot1 & 0x1ff) << 9) | (slot0 & 0x1ff))
#define DECODE_ROW(REF)   (REF >> 18)
#define DECODE_SLOT1(REF) ((REF >> 9) & 0x1ff)
#define DECODE_SLOT0(REF) (REF & 0x1ff)

#elif NR_ROWS_LOG <= 16 && NR_SLOTS <= (1 << 8)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 16) | ((slot1 & 0xff) << 8) | (slot0 & 0xff))
#define DECODE_ROW(REF)   (REF >> 16)
#define DECODE_SLOT1(REF) ((REF >> 8) & 0xff)
#define DECODE_SLOT0(REF) (REF & 0xff)

#elif NR_ROWS_LOG <= 18 && NR_SLOTS <= (1 << 7)

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

// Windows only for now
#define DEFAULT_NUM_MINING_MODE_THREADS 1
#define MAX_NUM_MINING_MODE_THREADS 16

#if THREADS_PER_WRITE == 1
#define ADJUSTED_SLOT_LEN(round) (((round) <= 5) ? SLOT_LEN : SLOT_LEN - 16)
#else
#define ADJUSTED_SLOT_LEN(round) SLOT_LEN
#endif

#define OPENCL_BUILD_OPTIONS_AMD "-I.. -I. -O5"
#define OPENCL_BUILD_OPTIONS     "-I.. -I."

#define NEXT_PRIME_NO(n) \
	(((n) <= 2) ? 2 : \
     ((n) <= 3) ? 3 : \
     ((n) <= 5) ? 5 : \
     ((n) <= 7) ? 7 : \
     ((n) <= 11) ? 11 : \
     ((n) <= 13) ? 13 : \
     ((n) <= 17) ? 17 : \
     ((n) <= 19) ? 19 : \
     ((n) <= 23) ? 23 : \
     ((n) <= 29) ? 29 : \
     ((n) <= 31) ? 31 : \
     ((n) <= 37) ? 37 : \
     ((n) <= 41) ? 41 : \
     ((n) <= 43) ? 43 : \
     ((n) <= 47) ? 47 : \
     ((n) <= 53) ? 53 : \
     ((n) <= 59) ? 59 : \
     ((n) <= 61) ? 61 : \
     ((n) <= 67) ? 67 : \
     ((n) <= 71) ? 71 : \
     ((n) <= 73) ? 73 : \
     ((n) <= 79) ? 79 : \
     ((n) <= 83) ? 83 : \
     ((n) <= 89) ? 89 : \
     ((n) <= 97) ? 97 : \
     ((n) <= 101) ? 101 : \
     ((n) <= 103) ? 103 : \
     ((n) <= 107) ? 107 : \
     ((n) <= 109) ? 109 : \
     ((n) <= 113) ? 113 : \
     ((n) <= 127) ? 127 : \
     ((n) <= 131) ? 131 : \
     ((n) <= 137) ? 137 : \
     ((n) <= 139) ? 139 : \
     ((n) <= 149) ? 149 : \
     ((n) <= 151) ? 151 : \
     ((n) <= 157) ? 157 : \
     ((n) <= 163) ? 163 : \
     ((n) <= 167) ? 167 : \
     ((n) <= 173) ? 173 : \
     ((n) <= 179) ? 179 : \
     ((n) <= 181) ? 181 : \
     ((n) <= 191) ? 191 : \
     ((n) <= 193) ? 193 : \
     ((n) <= 197) ? 197 : \
     ((n) <= 199) ? 199 : \
     ((n) <= 211) ? 211 : \
     ((n) <= 223) ? 223 : \
     ((n) <= 227) ? 227 : \
     ((n) <= 229) ? 229 : \
     ((n) <= 233) ? 233 : \
     ((n) <= 239) ? 239 : \
     ((n) <= 241) ? 241 : \
     ((n) <= 251) ? 251 : \
     ((n) <= 257) ? 257 : \
     ((n) <= 263) ? 263 : \
     ((n) <= 269) ? 269 : \
     ((n) <= 271) ? 271 : \
     ((n) <= 277) ? 277 : \
     ((n) <= 281) ? 281 : \
     ((n) <= 283) ? 283 : \
     ((n) <= 293) ? 293 : \
     ((n) <= 307) ? 307 : \
     ((n) <= 311) ? 311 : \
     ((n) <= 313) ? 313 : \
     ((n) <= 317) ? 317 : \
     ((n) <= 331) ? 331 : \
     ((n) <= 337) ? 337 : \
     ((n) <= 347) ? 347 : \
     ((n) <= 349) ? 349 : \
     ((n) <= 353) ? 353 : \
     ((n) <= 359) ? 359 : \
     ((n) <= 367) ? 367 : \
     ((n) <= 373) ? 373 : \
     ((n) <= 379) ? 379 : \
     ((n) <= 383) ? 383 : \
     ((n) <= 389) ? 389 : \
     ((n) <= 397) ? 397 : \
     ((n) <= 401) ? 401 : \
     ((n) <= 409) ? 409 : \
     ((n) <= 419) ? 419 : \
     ((n) <= 421) ? 421 : \
     ((n) <= 431) ? 431 : \
     ((n) <= 433) ? 433 : \
     ((n) <= 439) ? 439 : \
     ((n) <= 443) ? 443 : \
     ((n) <= 449) ? 449 : \
     ((n) <= 457) ? 457 : \
     ((n) <= 461) ? 461 : \
     ((n) <= 463) ? 463 : \
     ((n) <= 467) ? 467 : \
     ((n) <= 479) ? 479 : \
     ((n) <= 487) ? 487 : \
     ((n) <= 491) ? 491 : \
     ((n) <= 499) ? 499 : \
     ((n) <= 503) ? 503 : \
     ((n) <= 509) ? 509 : \
     ((n) <= 521) ? 521 : \
     ((n) <= 523) ? 523 : \
     ((n) <= 541) ? 541 : \
     ((n) <= 547) ? 547 : \
     ((n) <= 557) ? 557 : \
     ((n) <= 563) ? 563 : \
     ((n) <= 569) ? 569 : \
     ((n) <= 571) ? 571 : \
     ((n) <= 577) ? 577 : \
     ((n) <= 587) ? 587 : \
     ((n) <= 593) ? 593 : \
     ((n) <= 599) ? 599 : \
     ((n) <= 601) ? 601 : \
     ((n) <= 607) ? 607 : \
     ((n) <= 613) ? 613 : \
     ((n) <= 617) ? 617 : \
     ((n) <= 619) ? 619 : \
     ((n) <= 631) ? 631 : \
     ((n) <= 641) ? 641 : \
     ((n) <= 643) ? 643 : \
     ((n) <= 647) ? 647 : \
     ((n) <= 653) ? 653 : \
     ((n) <= 659) ? 659 : \
     ((n) <= 661) ? 661 : \
     ((n) <= 673) ? 673 : \
     ((n) <= 677) ? 677 : \
     ((n) <= 683) ? 683 : \
     ((n) <= 691) ? 691 : \
     ((n) <= 701) ? 701 : \
     ((n) <= 709) ? 709 : \
     ((n) <= 719) ? 719 : \
     ((n) <= 727) ? 727 : \
     ((n) <= 733) ? 733 : \
     ((n) <= 739) ? 739 : \
     ((n) <= 743) ? 743 : \
     ((n) <= 751) ? 751 : \
     ((n) <= 757) ? 757 : \
     ((n) <= 761) ? 761 : \
     ((n) <= 769) ? 769 : \
     ((n) <= 773) ? 773 : \
     ((n) <= 787) ? 787 : \
     ((n) <= 797) ? 797 : \
     ((n) <= 809) ? 809 : \
     ((n) <= 811) ? 811 : \
     ((n) <= 821) ? 821 : \
     ((n) <= 823) ? 823 : \
     ((n) <= 827) ? 827 : \
     ((n) <= 829) ? 829 : \
     ((n) <= 839) ? 839 : \
     ((n) <= 853) ? 853 : \
     ((n) <= 857) ? 857 : \
     ((n) <= 859) ? 859 : \
     ((n) <= 863) ? 863 : \
     ((n) <= 877) ? 877 : \
     ((n) <= 881) ? 881 : \
     ((n) <= 883) ? 883 : \
     ((n) <= 887) ? 887 : \
     ((n) <= 907) ? 907 : \
     ((n) <= 911) ? 911 : \
     ((n) <= 919) ? 919 : \
     ((n) <= 929) ? 929 : \
     ((n) <= 937) ? 937 : \
     ((n) <= 941) ? 941 : \
     ((n) <= 947) ? 947 : \
     ((n) <= 953) ? 953 : \
     ((n) <= 967) ? 967 : \
     ((n) <= 971) ? 971 : \
     ((n) <= 977) ? 977 : \
     ((n) <= 983) ? 983 : \
     ((n) <= 991) ? 991 : \
     ((n) <= 997) ? 997 : \
     ((n) <= 1009) ? 1009 : \
		             (n))

#define ROWS_IN_WORK_ITEM      (LOCAL_WORK_SIZE      / THREADS_PER_ROW     )
#define ROWS_IN_WORK_ITEM_SOLS (LOCAL_WORK_SIZE_SOLS / THREADS_PER_ROW_SOLS)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef union {
	struct {
		uint i;
		uint xi[6];
		uint padding;
	} slot;
	uint8 ui8;
	uint4 ui4[2];
	uint2 ui2[4];
	uint  ui[8];
} slot_t;

#if THREADS_PER_WRITE != 1
typedef __global slot_t *global_pointer_to_slot_t;
#endif


/*
** The new hash table has this layout (length in bytes in parens):
**
** round 0, table 0: i(4) pad(0) Xi(24) pad(4)
** round 1, table 1: i(4) pad(3) Xi(20) pad(5)
** round 2, table 2: i(4) pad(0) Xi(19) pad(9)
** round 3, table 3: i(4) pad(3) Xi(15) pad(10)
** round 4, table 4: i(4) pad(0) Xi(14) pad(14)
** round 5, table 5: i(4) pad(3) Xi(10) pad(15)
** round 6, table 6: i(4) pad(0) Xi( 9) pad(19)
** round 7, table 7: i(4) pad(3) Xi( 5) pad(20)
** round 8, table 8: i(4) pad(0) Xi( 4) pad(24)
**
*/

__constant ulong blake_iv_const[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

/*
** Reset counters in hash table.
*/
__kernel
void kernel_init_ht(__global char *ht, __global uint *rowCounters)
{
	rowCounters[get_global_id(0)] = 0;
}

/*
** OBSOLETE
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

__global char *get_slot_ptr(__global char *ht, uint round, uint row, uint slot)
{
#if 1
	return ht + (row * NR_SLOTS + slot) * ADJUSTED_SLOT_LEN(round);
#else
	return ht + (slot * NR_ROWS + row) * ADJUSTED_SLOT_LEN(round);
#endif
}

__global char *get_xi_ptr(__global char *ht, uint round, uint row, uint slot)
{
	return get_slot_ptr(ht, round, row, slot) + xi_offset_for_round(round);
}

void get_row_counters_index(uint *rowIdx, uint *rowOffset, uint row)
{
	*rowIdx = row / ROWS_PER_UINT;
	*rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
}

uint get_row(uint round, uint xi0)
{
	uint           row;
#if NR_ROWS_LOG == 12
	if (!(round % 2))
		row = (xi0 & 0xfff);
	else
		row = ((xi0 & 0x0f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 13
	if (!(round % 2))
		row = (xi0 & 0x1fff);
	else
		row = ((xi0 & 0x1f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 14
	if (!(round % 2))
		row = (xi0 & 0x3fff);
	else
		row = ((xi0 & 0x3f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 15
	if (!(round % 2))
		row = (xi0 & 0x7fff);
	else
		row = ((xi0 & 0x7f0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#elif NR_ROWS_LOG == 16
	if (!(round % 2))
		row = (xi0 & 0xffff);
	else
		row = ((xi0 & 0xff0f00) >> 8) | ((xi0 & 0xf0000000) >> 24);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	return row;
}

uint inc_row_counter(__global uint *rowCounters, uint row)
{
	uint rowIdx, rowOffset;
	get_row_counters_index(&rowIdx, &rowOffset, row);
	uint cnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
	cnt = (cnt >> rowOffset) & ROW_MASK;
	if (cnt >= NR_SLOTS) {
		// avoid overflows
		atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	}
	return cnt;
}

uint ht_store(uint round, __global char *ht, uint i,
	uint xi0, uint xi1, uint xi2, uint xi3, uint xi4, uint xi5, uint xi6, __global uint *rowCounters)
{
	uint row = get_row(round, xi0);
	uint cnt = inc_row_counter(rowCounters, row);
	if (cnt >= NR_SLOTS)
		return 1;
	__global char *p = get_slot_ptr(ht, round, row, cnt);
	slot_t slot;
	slot.slot.i = i;
	slot.slot.xi[0] = ((xi1 << 24) | (xi0 >> 8));
	slot.slot.xi[1] = ((xi2 << 24) | (xi1 >> 8));
	slot.slot.xi[2] = ((xi3 << 24) | (xi2 >> 8));
	slot.slot.xi[3] = ((xi4 << 24) | (xi3 >> 8));
	slot.slot.xi[4] = ((xi5 << 24) | (xi4 >> 8));
	slot.slot.xi[5] = ((xi6 << 24) | (xi5 >> 8));
	if (round <= 5) {
		*(__global uint8 *)p = slot.ui8;
	}
	else {
		*(__global uint4 *)p = slot.ui4[0];
	}
	return 0;
}

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
__kernel __attribute__((reqd_work_group_size(LOCAL_WORK_SIZE, 1, 1)))
void kernel_round0(__constant ulong *blake_state_const, __global char *ht,
	__global uint *rowCounters, __global uint *debug)
{
	__local ulong blake_state[64];
	__local ulong blake_iv[8];
	uint                tid = get_global_id(0);
	ulong               v[16];
	uint                inputs_per_thread = NR_INPUTS / get_global_size(0);
	uint                input = tid * inputs_per_thread;
	uint                input_end = (tid + 1) * inputs_per_thread;
	uint                dropped = 0;
	if (get_local_id(0) < 64)
		blake_state[get_local_id(0)] = blake_state_const[get_local_id(0)];
	if (get_local_id(0) < 8)
		blake_iv[get_local_id(0)] = blake_iv_const[get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	while (input < input_end) {
		// shift "i" to occupy the high 32 bits of the second ulong word in the
		// message block
		ulong word1 = (ulong)input << 32;
		// init vector v
		v[0] = blake_state[0];
		v[1] = blake_state[1];
		v[2] = blake_state[2];
		v[3] = blake_state[3];
		v[4] = blake_state[4];
		v[5] = blake_state[5];
		v[6] = blake_state[6];
		v[7] = blake_state[7];
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
		h[0] = blake_state[0] ^ v[0] ^ v[8];
		h[1] = blake_state[1] ^ v[1] ^ v[9];
		h[2] = blake_state[2] ^ v[2] ^ v[10];
		h[3] = blake_state[3] ^ v[3] ^ v[11];
		h[4] = blake_state[4] ^ v[4] ^ v[12];
		h[5] = blake_state[5] ^ v[5] ^ v[13];
		h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;

		// store the two Xi values in the hash table
#if ZCASH_HASH_LEN == 50
		dropped += ht_store(0, ht, input * 2,
			h[0] & 0xffffffff, h[0] >> 32,
			h[1] & 0xffffffff, h[1] >> 32,
			h[2] & 0xffffffff, h[2] >> 32,
			h[3] & 0xffffffff,
			rowCounters);
		dropped += ht_store(0, ht, input * 2 + 1,
			((h[3] >> 8) | (h[4] << (64 - 8))) & 0xffffffff,
			((h[3] >> 8) | (h[4] << (64 - 8))) >> 32,
			((h[4] >> 8) | (h[5] << (64 - 8))) & 0xffffffff,
			((h[4] >> 8) | (h[5] << (64 - 8))) >> 32,
			((h[5] >> 8) | (h[6] << (64 - 8))) & 0xffffffff,
			((h[5] >> 8) | (h[6] << (64 - 8))) >> 32,
			(h[6] >> 8) & 0xffffffff,
			rowCounters);
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
// single-thread reads, parallel writes
uint xor_and_store(uint round, __global char *ht_dst, uint row,
	uint slot_a, uint slot_b, __local uint *ai, __local uint *bi,
	__global uint *rowCounters
#if THREADS_PER_WRITE != 1
	, __local slot_t *slot_write_cache,
	__local global_pointer_to_slot_t *slot_ptrs
#endif
) {
	uint ret = 0;
	uint xi0, xi1, xi2, xi3, xi4, xi5;
	uint thread_index = get_local_id(0) % THREADS_PER_WRITE;

#if NR_ROWS_LOG < 8 && NR_ROWS_LOG > 20
#error "unsupported NR_ROWS_LOG"
#endif
	slot_t slot;
	__global slot_t *p = 0;
#if THREADS_PER_WRITE != 1
	slot_ptrs[get_local_id(0)] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	if (ai && bi) {
		xi0 = *(ai++);
		xi1 = *(ai++);
		if (round <= 7) xi2 = *(ai++);
		if (round <= 6) xi3 = *(ai++);
		if (round <= 4) xi4 = *(ai++);
		if (round <= 2) xi5 = *ai;

		xi0 ^= *(bi++);
		xi1 ^= *(bi++);
		if (round <= 7) xi2 ^= *(bi++);
		if (round <= 6) xi3 ^= *(bi++);
		if (round <= 4) xi4 ^= *(bi++);
		if (round <= 2) xi5 ^= *bi;

		if (!(round & 0x1)) {
			// skip padding bytes
			xi0 = (xi0 >> 24) | (xi1 << (32 - 24));
			xi1 = (xi1 >> 24) | (xi2 << (32 - 24));
			if (round <= 7) xi2 = (xi2 >> 24) | (xi3 << (32 - 24));
			if (round <= 6) xi3 = (xi3 >> 24) | (xi4 << (32 - 24));
			if (round <= 4) xi4 = (xi4 >> 24) | (xi5 << (32 - 24));
			if (round <= 2) xi5 = (xi5 >> 24);
		}

		// invalid solutions (which start happenning in round 5) have duplicate
		// inputs and xor to zero, so discard them
		if (xi0 || xi1) {
			uint new_row = get_row(round, xi0);
			uint new_slot_index = inc_row_counter(rowCounters, new_row);
			if (new_slot_index >= NR_SLOTS) {
				ret = 1;
			}
			else {
#if THREADS_PER_WRITE == 1
				p
#else
				slot_ptrs[get_local_id(0)]
#endif
					= (__global slot_t *)get_slot_ptr(ht_dst, round, new_row, new_slot_index);
			}
		}
	}

#if THREADS_PER_WRITE == 1
	if (p) {
		slot.slot.i = ENCODE_INPUTS(row, slot_a, slot_b);
		slot.slot.xi[0] = ((xi1 << 24) | (xi0 >> 8));
		slot.slot.xi[1] = ((xi2 << 24) | (xi1 >> 8));
		slot.slot.xi[2] = ((xi3 << 24) | (xi2 >> 8));
		slot.slot.xi[3] = ((xi4 << 24) | (xi3 >> 8));
		slot.slot.xi[4] = ((xi5 << 24) | (xi4 >> 8));
		slot.slot.xi[5] = ((xi5 >> 8));
		if (round <= 5)
			*(__global uint8 *)p = slot.ui8;
		else
			*(__global uint4 *)p = slot.ui4[0];
	}
#else
	barrier(CLK_LOCAL_MEM_FENCE);
	if (slot_ptrs[get_local_id(0)]) {
		slot_write_cache[get_local_id(0)].slot.i = ENCODE_INPUTS(row, slot_a, slot_b);
		slot_write_cache[get_local_id(0)].slot.xi[0] = ((xi1 << 24) | (xi0 >> 8));
		slot_write_cache[get_local_id(0)].slot.xi[1] = ((xi2 << 24) | (xi1 >> 8));
		slot_write_cache[get_local_id(0)].slot.xi[2] = ((xi3 << 24) | (xi2 >> 8));
		slot_write_cache[get_local_id(0)].slot.xi[3] = ((xi4 << 24) | (xi3 >> 8));
		slot_write_cache[get_local_id(0)].slot.xi[4] = ((xi5 << 24) | (xi4 >> 8));
		slot_write_cache[get_local_id(0)].slot.xi[5] = ((xi5 >> 8));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	uint local_id_base = get_local_id(0) - get_local_id(0) % THREADS_PER_WRITE;
	for (uint write_index = local_id_base; write_index < local_id_base + THREADS_PER_WRITE; ++write_index) {
		if (slot_ptrs[write_index]) {
#if THREADS_PER_WRITE == 2 
			* ((__global uint4 *)slot_ptrs[write_index] + thread_index) = slot_write_cache[write_index].ui4[thread_index];
#elif THREADS_PER_WRITE == 4
			* ((__global uint2 *)slot_ptrs[write_index] + thread_index) = slot_write_cache[write_index].ui2[thread_index];
#elif THREADS_PER_WRITE == 8
			* ((__global uint *)slot_ptrs[write_index] + thread_index) = slot_write_cache[write_index].ui[thread_index];
#else
#error "unsupported THREADS_PER_WRITE"
#endif
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif
	return ret;
}

/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/

#define UINTS_IN_XI(round) (((round) == 0) ? 6 : \
                            ((round) == 1) ? 6 : \
                            ((round) == 2) ? 5 : \
                            ((round) == 3) ? 5 : \
                            ((round) == 4) ? 4 : \
                            ((round) == 5) ? 4 : \
                            ((round) == 6) ? 3 : \
                            ((round) == 7) ? 2 : \
                                             1)

#define RESERVED_FOR_XI(round) (((round) == 0) ? 6 : \
                            ((round) == 1) ? 6 : \
                            ((round) == 2) ? 6 : \
                            ((round) == 3) ? 6 : \
                            ((round) == 4) ? 4 : \
                            ((round) == 5) ? 4 : \
                            ((round) == 6) ? 4 : \
                            ((round) == 7) ? 2 : \
                                             2)

void equihash_round(uint round,
	__global char *ht_src,
	__global char *ht_dst,
	__global uint *debug,
	__local uint  *slot_cache,
	__local uint *collisionsData,
	__local uint *collisionsNum,
	__global uint *rowCountersSrc,
	__global uint *rowCountersDst,
	uint threadsPerRow,
	__local uint *nr_slots_array,
	__local uchar *bins_data,
	__local uint *bin_counters_data)
{
	uint globalTid = get_global_id(0) / threadsPerRow;
	uint localTid = get_local_id(0) / threadsPerRow;
	uint localGroupId = get_local_id(0) % threadsPerRow;
	__global char *p;
	uint     i, j;
	uint     dropped_coll = 0;
	uint     dropped_stor = 0;
	__local uint  *a, *b;
	// the mask is also computed to read data from the previous round
#define BIN_MASK(round)        ((((round) + 1) % 2) ? 0xf000 : 0xf0000)
#define BIN_MASK_OFFSET(round) ((((round) + 1) % 2) ? 3 * 4 : 4 * 4)
#if NR_ROWS_LOG == 12
#define BIN_MASK2(round)        ((((round) + 1) % 2) ? 0x00f0 : 0xf000)
#define BIN_MASK2_OFFSET(round) ((((round) + 1) % 2) ? 0 : 8)
#elif NR_ROWS_LOG == 13
#define BIN_MASK2(round)        ((((round) + 1) % 2) ? 0x00e0 : 0xe000)
#define BIN_MASK2_OFFSET(round) ((((round) + 1) % 2) ? 1 : 9)
#elif NR_ROWS_LOG == 14
#define BIN_MASK2(round)        ((((round) + 1) % 2) ? 0x00c0 : 0xc000)
#define BIN_MASK2_OFFSET(round) ((((round) + 1) % 2) ? 2 : 10)
#elif NR_ROWS_LOG == 15
#define BIN_MASK2(round)        ((((round) + 1) % 2) ? 0x0080 : 0x8000)
#define BIN_MASK2_OFFSET(round) ((((round) + 1) % 2) ? 3 : 11)
#elif NR_ROWS_LOG == 16
#define BIN_MASK2(round)        0
#define BIN_MASK2_OFFSET(round) 0
#else
#error "unsupported NR_ROWS_LOG"
#endif    
#define NR_BINS (256 >> (NR_ROWS_LOG - 12))
	__local uchar *bins = &bins_data[localTid * BIN_SIZE * NR_BINS];
	__local uint *bin_counters = &bin_counters_data[localTid * NR_BINS];
#if THREADS_PER_WRITE != 1
	__local slot_t slot_write_cache[LOCAL_WORK_SIZE];
	__local global_pointer_to_slot_t slot_ptrs[LOCAL_WORK_SIZE];
#endif

	uint rows_per_work_item = (NR_ROWS + get_global_size(0) / threadsPerRow - 1) / (get_global_size(0) / threadsPerRow);
	uint rows_per_chunk = get_global_size(0) / threadsPerRow;

	for (uint chunk = 0; chunk < rows_per_work_item; chunk++) {
		uint cnt = 0;
		uint tid = globalTid + rows_per_chunk * chunk;
		uint gid = tid & ~(get_local_size(0) / threadsPerRow - 1);

		if (!get_local_id(0))
			*collisionsNum = 0;
		for (i = localGroupId; i < NR_BINS; i += threadsPerRow)
			bin_counters[i] = 0;
		if (tid < NR_ROWS && localGroupId == 0) {
			uint rowIdx, rowOffset;
			get_row_counters_index(&rowIdx, &rowOffset, tid);
			cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
			cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
			nr_slots_array[localTid] = cnt;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tid < NR_ROWS && localGroupId) {
			cnt = nr_slots_array[localTid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform a radix sort as slots get loaded into LDS.
		uint xi_first_bytes;
		uint bin_to_use;
		uint bin_counter_copy;
		// Make sure all the work items in the work group enter the loop.
		uint i_max = cnt + (get_local_size(0) - cnt % get_local_size(0)) - 1;
		for (i = localGroupId; i < i_max; i += threadsPerRow) {
			if (tid < NR_ROWS && i < cnt) {
				xi_first_bytes = *(__global uint *)get_xi_ptr(ht_src, round - 1, tid, i);
				slot_cache[(localTid * NR_SLOTS + i) * RESERVED_FOR_XI(round - 1)] = xi_first_bytes;
				for (j = 1; j < UINTS_IN_XI(round - 1); ++j)
					slot_cache[(localTid * NR_SLOTS + i) * RESERVED_FOR_XI(round - 1) + j] = *((__global uint *)get_xi_ptr(ht_src, round - 1, tid, i) + j);

				bin_to_use =
					((xi_first_bytes & BIN_MASK(round - 1)) >> BIN_MASK_OFFSET(round - 1))
					| ((xi_first_bytes & BIN_MASK2(round - 1)) >> BIN_MASK2_OFFSET(round - 1));
				bin_counter_copy = atomic_inc(&bin_counters[bin_to_use]);
				if (bin_counter_copy >= BIN_SIZE) {
					atomic_dec(&bin_counters[bin_to_use]);
					++dropped_coll;
				}
				else {
					bins[bin_to_use * BIN_SIZE + bin_counter_copy] = i;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			if (tid < NR_ROWS && i < cnt) {
				for (j = 0; j < bin_counter_copy; ++j) {
					uint index = atomic_inc(collisionsNum);
					if (index >= LDS_COLL_SIZE) {
						atomic_dec(collisionsNum);
						++dropped_coll;
					}
					else {
						collisionsData[index] = (localTid << 24) | (i << 12) | bins[bin_to_use * BIN_SIZE + j];
					}
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		uint totalCollisions = *collisionsNum;
		// Make sure all the work items in the work group enter and leave the loop at the same time.
		uint max_index = totalCollisions + (LOCAL_WORK_SIZE - totalCollisions % LOCAL_WORK_SIZE) - 1;
		for (uint index = get_local_id(0); index <= max_index; index += LOCAL_WORK_SIZE) {
			uint collision, collisionLocalThreadId, collisionThreadId;
			uint i, j, slot_cache_index_i, slot_cache_index_j;
			a = 0;
			b = 0;

			if (tid < NR_ROWS && index < totalCollisions) {
				collision = collisionsData[index];
				collisionLocalThreadId = collision >> 24;
				collisionThreadId = gid + collisionLocalThreadId;
				i = (collision >> 12) & 0xfff;
				j = collision & 0xfff;
				a = (__local uint *)&slot_cache[(collisionLocalThreadId * NR_SLOTS + i) * RESERVED_FOR_XI(round - 1)];
				b = (__local uint *)&slot_cache[(collisionLocalThreadId * NR_SLOTS + j) * RESERVED_FOR_XI(round - 1)];
			}

			dropped_stor += xor_and_store(
				round,
				ht_dst,
				collisionThreadId,
				i, j,
				a, b,
				rowCountersDst
#if THREADS_PER_WRITE == 1
			);
#else
				, slot_write_cache, slot_ptrs);
#endif
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

#ifdef ENABLE_DEBUG
	uint tid = get_global_id(0);
	debug[tid * 2] = dropped_coll;
	debug[tid * 2 + 1] = dropped_stor;
#endif
}

/*
** This defines kernel_round1, kernel_round2, ..., kernel_round7.
*/
#define KERNEL_ROUND(N) \
__kernel __attribute__((reqd_work_group_size(LOCAL_WORK_SIZE, 1, 1))) \
void kernel_round ## N(__global char *ht_src, __global char *ht_dst, \
	__global uint *rowCountersSrc, __global uint *rowCountersDst, \
       	__global uint *debug) \
{ \
    __local uint    slot_cache[NEXT_PRIME_NO(RESERVED_FOR_XI(N - 1) * SLOT_CACHE_SIZE)]; \
    __local uint    collisionsData[NEXT_PRIME_NO(LDS_COLL_SIZE)]; \
    __local uint    collisionsNum; \
	__local uint    nr_slots_array[NEXT_PRIME_NO(LOCAL_WORK_SIZE / THREADS_PER_ROW)]; \
	__local uchar   bins_data[NEXT_PRIME_NO((LOCAL_WORK_SIZE / THREADS_PER_ROW) * BIN_SIZE * NR_BINS)]; \
	__local uint    bin_counters_data[NEXT_PRIME_NO((LOCAL_WORK_SIZE / THREADS_PER_ROW) * NR_BINS)]; \
	equihash_round(N, ht_src, ht_dst, debug, slot_cache, collisionsData, \
	    &collisionsNum, rowCountersSrc, rowCountersDst, THREADS_PER_ROW, nr_slots_array, bins_data, bin_counters_data); \
}
KERNEL_ROUND(1)
KERNEL_ROUND(2)
KERNEL_ROUND(3)
KERNEL_ROUND(4)
KERNEL_ROUND(5)
KERNEL_ROUND(6)
KERNEL_ROUND(7)
KERNEL_ROUND(8)

uint expand_ref(__global char *ht, uint round, uint row, uint slot)
{
	return ((__global slot_t *)get_slot_ptr(ht, round, row, slot))->slot.i;
}

/*
** Expand references to inputs. Return 1 if so far the solution appears valid,
** or 0 otherwise (an invalid solution would be a solution with duplicate
** inputs, which can be detected at the last step: round == 0).
*/
uint expand_refs(__local uint *ins, uint nr_inputs, __global char **htabs,
	uint round)
{
	__global char	*ht = htabs[round];
	uint		i = nr_inputs - 1;
	uint		j = nr_inputs * 2 - 1;
	int			dup_to_watch = -1;
	do {
		ins[j] = expand_ref(ht, round,
			DECODE_ROW(ins[i]), DECODE_SLOT1(ins[i]));
		ins[j - 1] = expand_ref(ht, round,
			DECODE_ROW(ins[i]), DECODE_SLOT0(ins[i]));
		if (!round) {
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
void potential_sol(__global char **htabs, __global sols_t *sols,
	uint ref0, uint ref1, __local uint *values_tmp)
{
	uint	nr_values;
	uint	sol_i;
	uint	i;
	nr_values = 0;
	values_tmp[nr_values++] = ref0;
	values_tmp[nr_values++] = ref1;
	uint round = PARAM_K - 1;
	do {
		round--;
		if (!expand_refs(values_tmp, nr_values, htabs, round))
			return;
		nr_values *= 2;
	} while (round > 0);
	// solution appears valid, copy it to sols
	sol_i = atomic_inc(&sols->nr);
	if (sol_i >= MAX_SOLS)
		return;
	for (i = 0; i < (1 << PARAM_K); i++)
		sols->values[sol_i][i] = values_tmp[i];
	sols->valid[sol_i] = 1;
}

/*
** Scan the hash tables to find Equihash solutions.
*/
__kernel __attribute__((reqd_work_group_size(LOCAL_WORK_SIZE_SOLS, 1, 1)))
void kernel_sols(__global char *ht0,
	__global char *ht1,
	__global char *ht2,
	__global char *ht3,
	__global char *ht4,
	__global char *ht5,
	__global char *ht6,
	__global char *ht7,
	__global char *ht8,
	__global sols_t *sols,
	__global uint *rowCountersSrc)
{
	__local uint refs[NEXT_PRIME_NO(NR_SLOTS*(LOCAL_WORK_SIZE_SOLS / THREADS_PER_ROW_SOLS))];
	__local uint data[NEXT_PRIME_NO(NR_SLOTS*(LOCAL_WORK_SIZE_SOLS / THREADS_PER_ROW_SOLS))];
	__local uint	values_tmp[NEXT_PRIME_NO(1 << PARAM_K)];
	__local uint    semaphoe;

	uint globalTid = get_global_id(0) / THREADS_PER_ROW_SOLS;
	uint localTid = get_local_id(0) / THREADS_PER_ROW_SOLS;
	uint localGroupId = get_local_id(0) % THREADS_PER_ROW_SOLS;
	__local uint *refsPtr = &refs[NR_SLOTS*localTid];
	__local uint *dataPtr = &data[NR_SLOTS*localTid];

	__global char	*htabs[] = { ht0, ht1, ht2, ht3, ht4, ht5, ht6, ht7, ht8 };
	uint		ht_i = (PARAM_K - 1); // table filled at last round
	uint		cnt;
	uint		i, j;
	__global char	*p;
	uint		ref_i, ref_j;
	__local uchar   bins_data[(LOCAL_WORK_SIZE_SOLS / THREADS_PER_ROW_SOLS) * NR_SLOTS * NR_BINS];
	__local uint    bin_counters_data[(LOCAL_WORK_SIZE_SOLS / THREADS_PER_ROW_SOLS) * NR_BINS];
	__local uchar *bins = &bins_data[localTid * NR_SLOTS * NR_BINS];
	__local uint *bin_counters = &bin_counters_data[localTid * NR_BINS];

	if (!get_global_id(0))
		sols->nr = sols->likely_invalids = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);

	uint rows_per_work_item = (NR_ROWS + get_global_size(0) / THREADS_PER_ROW_SOLS - 1) / (get_global_size(0) / THREADS_PER_ROW_SOLS);
	uint rows_per_chunk = get_global_size(0) / THREADS_PER_ROW_SOLS;

	for (uint chunk = 0; chunk < rows_per_work_item; chunk++) {
		uint tid = globalTid + rows_per_chunk * chunk;
		uint gid = tid & ~(get_local_size(0) / THREADS_PER_ROW_SOLS - 1);

		__local uint nr_slots_array[LOCAL_WORK_SIZE_SOLS / THREADS_PER_ROW_SOLS];
		if (tid < NR_ROWS) {
			if (!get_local_id(0))
				semaphoe = 0;
			for (i = localGroupId; i < NR_BINS; i += THREADS_PER_ROW_SOLS)
				bin_counters[i] = 0;
			if (localGroupId == 0) {
				uint rowIdx, rowOffset;
				get_row_counters_index(&rowIdx, &rowOffset, tid);
				cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
				cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
				nr_slots_array[localTid] = cnt;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tid < NR_ROWS) {
			if (localGroupId)
				cnt = nr_slots_array[localTid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// in the final hash table, we are looking for a match on both the bits
		// part of the previous PREFIX colliding bits, and the last PREFIX bits.
		__local ulong coll;
		if (tid < NR_ROWS) {
			for (i = localGroupId; i < cnt && !semaphoe; i += THREADS_PER_ROW_SOLS) {
				p = get_slot_ptr(htabs[ht_i], PARAM_K - 1, tid, i);
				refsPtr[i] = ((__global slot_t *)p)->slot.i;
				uint xi_first_bytes = dataPtr[i] = ((__global slot_t *)p)->slot.xi[0];
				uint bin_to_use =
					((xi_first_bytes & BIN_MASK(PARAM_K - 1)) >> BIN_MASK_OFFSET(PARAM_K - 1))
					| ((xi_first_bytes & BIN_MASK2(PARAM_K - 1)) >> BIN_MASK2_OFFSET(PARAM_K - 1));
				uint bin_counter_copy = atomic_inc(&bin_counters[bin_to_use]);
				bins[bin_to_use * NR_SLOTS + bin_counter_copy] = i;
				if (bin_counter_copy) {
					for (j = 0; j < bin_counter_copy && !semaphoe; ++j) {
						uint slot_index_j = bins[bin_to_use * NR_SLOTS + j];
						if (xi_first_bytes == dataPtr[slot_index_j]) {
							if (atomic_inc(&semaphoe) == 0)
								coll = ((ulong)refsPtr[i] << 32) | refsPtr[slot_index_j];
						}
					}
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		if (tid < NR_ROWS) {
			if (get_local_id(0) == 0 && semaphoe)
				potential_sol(htabs, sols, coll >> 32, coll & 0xffffffff, values_tmp);
		}
	}
}