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

#define NR_ROWS_LOG            15
#define NR_SLOTS               120
#define LOCAL_WORK_SIZE        256
#define THREADS_PER_ROW        256
#define LOCAL_WORK_SIZE_SOLS   64
#define THREADS_PER_ROW_SOLS   64
#define GLOBAL_WORK_SIZE_RATIO 512
#define SLOT_CACHE_SIZE        (NR_SLOTS * (LOCAL_WORK_SIZE/THREADS_PER_ROW) * 75 / 100)
#define LDS_COLL_SIZE          (NR_SLOTS * (LOCAL_WORK_SIZE / THREADS_PER_ROW) * 120 / 100)

#define SLOT_CACHE_INDEX_TYPE uchar

#define PARAM_N				   200
#define PARAM_K			       9
#define PREFIX                 (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS              (1 << PREFIX)
// Approximate log base 2 of number of elements in hash tables
#define APX_NR_ELMS_LOG        PREFIX + 1)

// Setting this to 1 might make Gateless Gate faster, see TROUBLESHOOTING.md
#define OPTIM_SIMPLIFY_ROUND   1

// Ratio of time of sleeping before rechecking if task is done (0-1)
#define SLEEP_RECHECK_RATIO 0.60
// Ratio of time to busy wait for the solution (0-1)
// The higher value the higher CPU usage with Nvidia
#define SLEEP_SKIP_RATIO 0.005

// Make hash tables OVERHEAD times larger than necessary to store the average
// number of elements per row. The ideal value is as small as possible to
// reduce memory usage, but not too small or else elements are dropped from the
// hash tables.
//
// The actual number of elements per row is closer to the theoretical average
// (less variance) when NR_ROWS_LOG is small. So accordingly OVERHEAD can be
// smaller.
//
// Even (as opposed to odd) values of OVERHEAD sometimes significantly decrease
// performance as they cause VRAM channel conflicts.
#if NR_ROWS_LOG <= 16
#define OVERHEAD                        2
#elif NR_ROWS_LOG == 17
#define OVERHEAD                        3
#elif NR_ROWS_LOG == 18
#define OVERHEAD                        3
#elif NR_ROWS_LOG == 19
#define OVERHEAD                        5
#elif NR_ROWS_LOG == 20 && OPTIM_SIMPLIFY_ROUND
#define OVERHEAD                        6
#elif NR_ROWS_LOG == 20
#define OVERHEAD                        9
#endif

#define NR_ROWS                         (1 << NR_ROWS_LOG)
#ifndef NR_SLOTS
#define NR_SLOTS                        (((1 << (APX_NR_ELMS_LOG - NR_ROWS_LOG)) * OVERHEAD))
#endif
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
#define MAX_SOLS			10
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

#if NR_ROWS_LOG <= 16 && NR_SLOTS <= (1 << 8)

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

#define ADJUSTED_SLOT_LEN(round) (((round) <= 5) ? SLOT_LEN : SLOT_LEN / 2)
#define OPENCL_BUILD_OPTIONS_AMD "-I.. -I. -O1"
#define OPENCL_BUILD_OPTIONS     "-I.. -I."