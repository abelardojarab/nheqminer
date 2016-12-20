#include "ocl_gatelessgate.hpp"

#pragma comment(lib, "winmm.lib")
#define _CRT_RAND_S 


//#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <sys/types.h>
//#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <unistd.h>
//#include <getopt.h>
#include <errno.h>

#include <opencl.h>

#include <fstream>

#include "gettimeofday.h"
#include <TimeAPI.h>

#include <blake/blake.hpp>
using namespace blake;
#include <sha256/sha256.hpp>

#include <inttypes.h>

typedef uint8_t		uchar;
typedef uint32_t	uint;
typedef uint64_t	ulong;
#include "param.h"

#define MIN(A, B)	(((A) < (B)) ? (A) : (B))
#define MAX(A, B)	(((A) > (B)) ? (A) : (B))

#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))

struct timeval kern_avg_run_time;

typedef struct  debug_s
{
	uint32_t    dropped_coll;
	uint32_t    dropped_stor;
}               debug_t;

struct OclGGContext {
	cl_context _context;
	cl_program _program;
	cl_device_id _dev_id;

	cl_platform_id platform_id = 0;

	cl_command_queue queue;

	cl_kernel k_init_ht;
	cl_kernel k_rounds[PARAM_K];
	cl_kernel k_sols;

	cl_mem buf_ht[9], buf_sols, buf_dbg, rowCounters[2];
	size_t global_ws;
	size_t local_work_size = 64;

	sols_t	*sols;

	bool init(cl_device_id dev, unsigned threadsNum, unsigned threadsPerBlock);

	~OclGGContext() {
		clReleaseMemObject(buf_dbg);
		clReleaseMemObject(buf_ht[0]);
		clReleaseMemObject(buf_ht[1]);
		clReleaseMemObject(rowCounters[0]);
		clReleaseMemObject(rowCounters[1]);
		free(sols);
	}
};

cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
	void *host_ptr);

bool OclGGContext::init(
	cl_device_id dev,
	unsigned int threadsNum,
	unsigned int threadsPerBlock)
{
	cl_int error;

	queue = clCreateCommandQueue(_context, dev, 0, &error);

#ifdef ENABLE_DEBUG
	size_t              dbg_size = NR_ROWS;
#else
	size_t              dbg_size = 1;
#endif

	buf_dbg = check_clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, dbg_size, NULL);
	buf_ht[0] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[1] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[2] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[3] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[4] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[5] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[6] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[7] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_ht[8] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	buf_sols = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, sizeof(sols_t), NULL);

	rowCounters[0] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, RC_SIZE, NULL);
	rowCounters[1] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, RC_SIZE, NULL);

	fprintf(stderr, "Hash tables will use %.1f MB\n", 9 * HT_SIZE / 1e6);

	k_init_ht = clCreateKernel(_program, "kernel_init_ht", &error);

	if (error != CL_SUCCESS) {
		printf("kernel error\n");
	}

	for (unsigned i = 0; i < WK; i++) {
		char kernelName[128];
		sprintf(kernelName, "kernel_round%d", i);
		k_rounds[i] = clCreateKernel(_program, kernelName, &error);
		if (error != CL_SUCCESS) {
			printf("kernel round error\n");
		}
	}

	sols = (sols_t *)malloc(sizeof(*sols));

	k_sols = clCreateKernel(_program, "kernel_sols", &error);
	if (error != CL_SUCCESS) {
		printf("kernel sols error\n");
	}

	return true;
}

///
static int             verbose = 0;
static uint32_t	show_encoded = 0;

static cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
	void *host_ptr)
{
	cl_int	status;
	cl_mem	ret;
	ret = clCreateBuffer(ctx, flags, size, host_ptr, &status);
	if (status != CL_SUCCESS || !ret)
		printf("clCreateBuffer (%d)\n", status);
	return ret;
}

static void check_clSetKernelArg(cl_kernel k, cl_uint a_pos, cl_mem *a)
{
	cl_int	status;
	status = clSetKernelArg(k, a_pos, sizeof(*a), a);
	if (status != CL_SUCCESS)
		printf("clSetKernelArg (%d)\n", status);
}

/*static void check_clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel k, cl_uint
	work_dim, const size_t *global_work_offset, const size_t
	*global_work_size, const size_t *local_work_size, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_uint	status;
	status = clEnqueueNDRangeKernel(queue, k, work_dim, global_work_offset,
		global_work_size, local_work_size, num_events_in_wait_list,
		event_wait_list, event);
	OCL(status);
}*/

static void check_clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool
	blocking_read, size_t offset, size_t size, void *ptr, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_int	status;
	status = clEnqueueReadBuffer(queue, buffer, blocking_read, offset,
		size, ptr, num_events_in_wait_list, event_wait_list, event);
	if (status != CL_SUCCESS)
		printf("clEnqueueReadBuffer (%d)\n", status);
	OCL(status);
}

static void hexdump(uint8_t *a, uint32_t a_len)
{
	for (uint32_t i = 0; i < a_len; i++)
		fprintf(stderr, "%02x", a[i]);
}

static char* s_hexdump(const void *_a, uint32_t a_len)
{
	const uint8_t	*a = (uint8_t	*)_a;
	static char		buf[1024];
	uint32_t		i;
	for (i = 0; i < a_len && i + 2 < sizeof(buf); i++)
		sprintf(buf + i * 2, "%02x", a[i]);
	buf[i * 2] = 0;
	return buf;
}

static uint8_t hex2val(const char *base, size_t off)
{
	const char          c = base[off];
	if (c >= '0' && c <= '9')           return c - '0';
	else if (c >= 'a' && c <= 'f')      return 10 + c - 'a';
	else if (c >= 'A' && c <= 'F')      return 10 + c - 'A';
	printf("Invalid hex char at offset %zd: ...%c...\n", off, c);
	return 0;
}

static unsigned nr_compute_units(const char *gpu)
{
	if (!strcmp(gpu, "rx480")) return 36;
	fprintf(stderr, "Unknown GPU: %s\n", gpu);
	return 0;
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

static void get_program_build_log(cl_program program, cl_device_id device)
{
	cl_int		status;
	char	        val[2 * 1024 * 1024];
	size_t		ret = 0;
	status = clGetProgramBuildInfo(program, device,
		CL_PROGRAM_BUILD_LOG,
		sizeof(val),	// size_t param_value_size
		&val,		// void *param_value
		&ret);		// size_t *param_value_size_ret
	if (status != CL_SUCCESS)
		printf("clGetProgramBuildInfo (%d)\n", status);
	fprintf(stderr, "%s\n", val);
}

static size_t select_work_size_blake(void)
{
	size_t              work_size =
		64 * /* thread per wavefront */
		BLAKE_WPS * /* wavefront per simd */
		4 * /* simd per compute unit */
		nr_compute_units("rx480");
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	while (NR_INPUTS % work_size)
		work_size += 64;
	//debug("Blake: work size %zd\n", work_size);
	return work_size;
}

static void init_ht(cl_command_queue queue, cl_kernel k_init_ht, cl_mem buf_ht, cl_mem rowCounters)
{
	size_t      global_ws = RC_SIZE / sizeof(cl_uint);
	size_t      local_ws = 256;
	cl_int      status;
#if 0
	uint32_t    pat = -1;
	status = clEnqueueFillBuffer(queue, buf_ht, &pat, sizeof(pat), 0,
		NR_ROWS * NR_SLOTS * SLOT_LEN,
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL);	// cl_event	*event
	if (status != CL_SUCCESS)
		fatal("clEnqueueFillBuffer (%d)\n", status);
#endif
	status = clSetKernelArg(k_init_ht, 0, sizeof(buf_ht), &buf_ht);
	status = clSetKernelArg(k_init_ht, 1, sizeof(rowCounters), &rowCounters);
	if (status != CL_SUCCESS)
		printf("clSetKernelArg (%d)\n", status);
	OCL(clEnqueueNDRangeKernel(queue, k_init_ht,
		1,		// cl_uint	work_dim
		NULL,	// size_t	*global_work_offset
		&global_ws,	// size_t	*global_work_size
		&local_ws,	// size_t	*local_work_size
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL));	// cl_event	*event
}


/*
** Sort a pair of binary blobs (a, b) which are consecutive in memory and
** occupy a total of 2*len 32-bit words.
**
** a            points to the pair
** len          number of 32-bit words in each pair
*/
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


#define SEEN_LEN (1 << (PREFIX + 1)) / 8

static uint32_t verify_sol(sols_t *sols, unsigned sol_i)
{
	uint32_t  *inputs = sols->values[sol_i];
	//uint32_t  seen_len = (1 << (PREFIX + 1)) / 8;
	//uint8_t seen[(1 << (PREFIX + 1)) / 8];
	uint8_t	seen[SEEN_LEN];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, SEEN_LEN);
	for (i = 0; i < (1 << PARAM_K); i++)
	{
		if (inputs[i] / 8 >= SEEN_LEN)
		{
			printf("Invalid input retrieved from device: %d\n", inputs[i]);
			sols->valid[sol_i] = 0;
			return 0;
		}
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


static struct timeval time_diff(struct timeval start, struct timeval end)
{
	struct timeval temp;
	if ((end.tv_usec - start.tv_usec)<0) {
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_usec = 1000000 + end.tv_usec - start.tv_usec;
	}
	else {
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_usec = end.tv_usec - start.tv_usec;
	}
	return temp;
}

/*
** Write ZCASH_SOL_LEN bytes representing the encoded solution as per the
** Zcash protocol specs (512 x 21-bit inputs).
**
** out		ZCASH_SOL_LEN-byte buffer where the solution will be stored
** inputs	array of 32-bit inputs
** n		number of elements in array
*/
static void store_encoded_sol(uint8_t *out, uint32_t *inputs, uint32_t n)
{
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
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
			*out++ = x;
			x = x_bits_used = 0;
		}
	}
}

/*
** Compare two 256-bit values interpreted as little-endian 256-bit integers.
*/
static int32_t cmp_target_256(void *_a, void *_b)
{
	uint8_t	*a = static_cast<uint8_t*>(_a);
	uint8_t	*b = static_cast<uint8_t*>(_b);
	int32_t	i;
	for (i = SHA256_TARGET_LEN - 1; i >= 0; i--)
		if (a[i] != b[i])
			return (int32_t)a[i] - b[i];
	return 0;
}

/*
** Verify if the solution's block hash is under the target, and if yes print
** it formatted as:
** "sol: <job_id> <ntime> <nonce_rightpart> <solSize+sol>"
**
** Return 1 iff the block hash is under the target.
*/
uint32_t print_solver_line(uint32_t *values, uint8_t *header,
	size_t fixed_nonce_bytes, uint8_t *target, char *job_id)
{
	uint8_t	buffer[ZCASH_BLOCK_HEADER_LEN + ZCASH_SOLSIZE_LEN +
		ZCASH_SOL_LEN];
	uint8_t	hash0[SHA256_DIGEST_SIZE];
	uint8_t	hash1[SHA256_DIGEST_SIZE];
	uint8_t	*p;
	p = buffer;
	memcpy(p, header, ZCASH_BLOCK_HEADER_LEN);
	p += ZCASH_BLOCK_HEADER_LEN;
	memcpy(p, "\xfd\x40\x05", ZCASH_SOLSIZE_LEN);
	p += ZCASH_SOLSIZE_LEN;
	store_encoded_sol(p, values, 1 << PARAM_K);
	sha256::Sha256_Onestep(buffer, sizeof(buffer), hash0);
	sha256::Sha256_Onestep(hash0, sizeof(hash0), hash1);
	// compare the double SHA256 hash with the target
	if (cmp_target_256(target, hash1) < 0)
	{
		printf("Hash is above target\n");
		return 0;
	}
	printf("Hash is under target\n");
	printf("sol: %s ", job_id);
	p = header + ZCASH_BLOCK_OFFSET_NTIME;
	printf("%02x%02x%02x%02x ", p[0], p[1], p[2], p[3]);
	printf("%s ", s_hexdump(header + ZCASH_BLOCK_HEADER_LEN - ZCASH_NONCE_LEN +
		fixed_nonce_bytes, ZCASH_NONCE_LEN - fixed_nonce_bytes));
	printf("%s%s\n", ZCASH_SOLSIZE_HEX,
		s_hexdump(buffer + ZCASH_BLOCK_HEADER_LEN + ZCASH_SOLSIZE_LEN,
			ZCASH_SOL_LEN));
	fflush(stdout);
	return 1;
}

int sol_cmp(const void *_a, const void *_b)
{
	const uint32_t	*a = static_cast<const uint32_t*>(_a);
	const uint32_t	*b = static_cast<const uint32_t*>(_b);
	for (uint32_t i = 0; i < (1 << PARAM_K); i++)
	{
		if (*a != *b)
			return *a - *b;
		a++;
		b++;
	}
	return 0;
}

/*
** Print on stdout a hex representation of the encoded solution as per the
** zcash protocol specs (512 x 21-bit inputs).
**
** inputs	array of 32-bit inputs
** n		number of elements in array
*/
static void print_encoded_sol(uint32_t *inputs, uint32_t n)
{
	uint8_t	sol[ZCASH_SOL_LEN];
	uint32_t	i;
	store_encoded_sol(sol, inputs, n);
	for (i = 0; i < sizeof(sol); i++)
		printf("%02x", sol[i]);
	printf("\n");
	fflush(stdout);
}

static void print_sol(uint32_t *values, uint64_t *nonce)
{
	uint32_t	show_n_sols;
	show_n_sols = (1 << PARAM_K);
	if (verbose < 2)
		show_n_sols = MIN(10, show_n_sols);
	fprintf(stderr, "Soln:");
	// for brievity, only print "small" nonces
	if (*nonce < (1ULL << 32))
		fprintf(stderr, " 0x%" PRIx64 ":", *nonce);
	for (unsigned i = 0; i < show_n_sols; i++)
		fprintf(stderr, " %x", values[i]);
	fprintf(stderr, "%s\n", (show_n_sols != (1 << PARAM_K) ? "..." : ""));
}

/*
** Print all solutions.
**
** In mining mode, return the number of shares, that is the number of solutions
** that were under the target.
*/
static uint32_t print_sols(sols_t *all_sols, uint64_t *nonce, uint32_t nr_valid_sols,
	uint8_t *header, size_t fixed_nonce_bytes, uint8_t *target,
	char *job_id)
{
	uint8_t		*valid_sols;
	uint32_t		counted;
	uint32_t		shares = 0;
	valid_sols = static_cast<uint8_t*>(malloc(nr_valid_sols * SOL_SIZE));
	if (!valid_sols)
		printf("malloc: %s\n", strerror(errno));
	counted = 0;
	for (uint32_t i = 0; i < all_sols->nr; i++)
		if (all_sols->valid[i])
		{
			if (counted >= nr_valid_sols)
				printf("Bug: more than %d solutions\n", nr_valid_sols);
			memcpy(valid_sols + counted * SOL_SIZE, all_sols->values[i],
				SOL_SIZE);
			counted++;
		}
	assert(counted == nr_valid_sols);
	// sort the solutions amongst each other, to make the solver's output
	// deterministic and testable
	qsort(valid_sols, nr_valid_sols, SOL_SIZE, sol_cmp);
	for (uint32_t i = 0; i < nr_valid_sols; i++)
	{
		uint32_t	*inputs = (uint32_t *)(valid_sols + i * SOL_SIZE);
		if (show_encoded)
			print_encoded_sol(inputs, 1 << PARAM_K);
		if (verbose)
			print_sol(inputs, nonce);
		if (true)
			shares += print_solver_line(inputs, header, fixed_nonce_bytes,
				target, job_id);
	}
	free(valid_sols);
	return shares;
}

/*
** Return the number of valid solutions.
*/
static uint32_t verify_sols(cl_command_queue queue, cl_mem buf_sols, uint64_t *nonce,
	uint8_t *header, size_t fixed_nonce_bytes, uint8_t *target,
	char *job_id, uint32_t *shares, struct timeval *start_time, bool is_amd)
{
	sols_t	*sols;
	uint32_t	nr_valid_sols;
	sols = (sols_t *)malloc(sizeof(*sols));
	if (!sols)
		printf("malloc: %s\n", strerror(errno));
#ifdef WIN32
	timeBeginPeriod(1);
	DWORD duration = (DWORD)kern_avg_run_time.tv_sec * 1000 + (DWORD)kern_avg_run_time.tv_usec / 1000;
	if (!is_amd && duration < 1000)
		Sleep(duration);
#endif
	check_clEnqueueReadBuffer(queue, buf_sols,
		CL_TRUE,	// cl_bool	blocking_read
		0,		// size_t	offset
		sizeof(*sols),	// size_t	size
		sols,	// void		*ptr
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL);	// cl_event	*event
	struct timeval curr_time;
	gettimeofday(&curr_time, NULL);

	struct timeval t_diff = time_diff(*start_time, curr_time);

	double a_diff = t_diff.tv_sec * 1e6 + t_diff.tv_usec;
	double kern_avg = kern_avg_run_time.tv_sec * 1e6 + kern_avg_run_time.tv_usec;
	if (kern_avg == 0)
		kern_avg = a_diff;
	else
		kern_avg = kern_avg * 70 / 100 + a_diff * 28 / 100; // it is 2% less than average
															// thus allowing time to reduce

	kern_avg_run_time.tv_sec = (time_t)(kern_avg / 1e6);
	kern_avg_run_time.tv_usec = ((long)kern_avg) % 1000000;

	if (sols->nr > MAX_SOLS)
	{
		fprintf(stderr, "%d (probably invalid) solutions were dropped!\n",
			sols->nr - MAX_SOLS);
		sols->nr = MAX_SOLS;
	}
	printf("Retrieved %d potential solutions\n", sols->nr);
	nr_valid_sols = 0;
	for (unsigned sol_i = 0; sol_i < sols->nr; sol_i++)
		nr_valid_sols += verify_sol(sols, sol_i);
	uint32_t sh = print_sols(sols, nonce, nr_valid_sols, header, fixed_nonce_bytes, target, job_id);
	if (shares)
		*shares = sh;
	printf("Stats: %d likely invalids\n", sols->likely_invalids);
	free(sols);
	return nr_valid_sols;
}


ocl_gatelessgate::ocl_gatelessgate(int platf_id, int dev_id) {
	platform_id = platf_id;
	device_id = dev_id;
	// TODO 
	threadsNum = 8192;
	wokrsize = 128; // 256;
}

std::string ocl_gatelessgate::getdevinfo() {
	static auto devices = GetAllDevices(platform_id);
	auto device = devices[device_id];
	std::vector<char> name(256, 0);
	size_t nActualSize = 0;
	std::string gpu_name;

	cl_int rc = clGetDeviceInfo(device, CL_DEVICE_NAME, name.size(), &name[0], &nActualSize);

	gpu_name.assign(&name[0], nActualSize);

	return "GPU_ID( " + gpu_name + ")";
}

// STATICS START
int ocl_gatelessgate::getcount() {
	static auto devices = GetAllDevices();
	return devices.size();
}

void ocl_gatelessgate::getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version) {
	static auto devices = GetAllDevices(platf_id);

	if (devices.size() <= d_id) {
		return;
	}
	auto device = devices[d_id];

	std::vector<char> name(256, 0);
	cl_uint compute_units = 0;

	size_t nActualSize = 0;
	cl_int rc = clGetDeviceInfo(device, CL_DEVICE_NAME, name.size(), &name[0], &nActualSize);

	if (rc == CL_SUCCESS) {
		gpu_name.assign(&name[0], nActualSize);
	}

	rc = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(cl_uint), &compute_units, &nActualSize);
	if (rc == CL_SUCCESS) {
		sm_count = (int)compute_units;
	}

	memset(&name[0], 0, name.size());
	rc = clGetDeviceInfo(device, CL_DEVICE_VERSION, name.size(), &name[0], &nActualSize);
	if (rc == CL_SUCCESS) {
		version.assign(&name[0], nActualSize);
	}
}


static bool is_platform_amd(cl_platform_id platform_id)
{
	char	name[1024];
	size_t	len = 0;
	int		status;
	status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(name), &name,
		&len);
	if (status != CL_SUCCESS)
		printf("clGetPlatformInfo (%d)\n", status);
	return strncmp(name, "AMD Accelerated Parallel Processing", len) == 0;
}


void ocl_gatelessgate::start(ocl_gatelessgate& device_context) {
	/*TODO*/
	device_context.is_init_success = false;
	device_context.oclc = new OclGGContext;
	auto devices = GetAllDevices(device_context.platform_id);

	printf("pid %i, size %u\n", device_context.platform_id, devices.size());
	auto device = devices[device_context.device_id];

	size_t nActualSize = 0;
	cl_platform_id platform_id = nullptr;
	cl_int rc = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform_id, nullptr);


	device_context.oclc->_dev_id = device;
	device_context.oclc->platform_id = platform_id;

	// context create
	cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)device_context.oclc->platform_id, 0 };
	cl_int error;
	device_context.oclc->_context = clCreateContext(props, 1, &device, 0, 0, &error);
	//OCLR(error, false);
	if (cl_int err = error) {
		printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__);
		return;
	}

	cl_int binstatus;

	device_context.is_amd = is_platform_amd(platform_id);

	char kernelName[64];
	sprintf(kernelName, "gatelessgate_gpu_%u.bin", (unsigned)device_context.device_id);
	if (!clCompileKernel(device_context.oclc->_context,
		device,
		kernelName,
		{ "zcash/gpu/gatelessgate.cl" },
		device_context.is_amd ? OPENCL_BUILD_OPTIONS_AMD : OPENCL_BUILD_OPTIONS,
		&binstatus,
		&device_context.oclc->_program)) {
		return;
	}

	if (binstatus == CL_SUCCESS) {
		if (!device_context.oclc->init(device, device_context.threadsNum, device_context.wokrsize)) {
			printf("Init failed");
			return;
		}
	}
	else {
		printf("GPU %d: failed to load kernel\n", device_context.device_id);
		return;
	}

	device_context.is_init_success = true;
}

#include <iostream>

void ocl_gatelessgate::stop(ocl_gatelessgate& device_context) {
	if (device_context.oclc != nullptr) delete device_context.oclc;
}

void ocl_gatelessgate::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	std::function<bool()> cancelf,
	std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
	std::function<void(void)> hashdonef,
	ocl_gatelessgate& device_context) {

	uint64_t		*nonce_ptr;
	
	unsigned char context[140];
	memset(context, 0, 140);
	memcpy(context, tequihash_header, tequihash_header_len);
	memcpy(context + tequihash_header_len, nonce, nonce_len);

	OclGGContext *miner = device_context.oclc;
	clFlush(miner->queue);

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

	cl_mem buf_blake_st;
	buf_blake_st = check_clCreateBuffer(miner->_context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(blake2b_state_s), &initialCtx);

	cl_uint  compute_units;
	cl_int status = clGetDeviceInfo(miner->_dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
	if (status != CL_SUCCESS)
		printf("clGetDeviceInfo (%d)\n", status);

	miner->local_work_size = LOCAL_WORK_SIZE;

	for (unsigned round = 0; round < PARAM_K; round++)
	{
		init_ht(miner->queue, miner->k_init_ht, miner->buf_ht[round & 1], miner->rowCounters[round & 1]);
		if (!round)
		{
			check_clSetKernelArg(miner->k_rounds[round], 0, &buf_blake_st);
			check_clSetKernelArg(miner->k_rounds[round], 1, &miner->buf_ht[round]);
			check_clSetKernelArg(miner->k_rounds[round], 2, &miner->rowCounters[round % 2]);
			miner->global_ws = select_work_size_blake();
		}
		else
		{
			check_clSetKernelArg(miner->k_rounds[round], 0, &miner->buf_ht[round - 1]);
			check_clSetKernelArg(miner->k_rounds[round], 1, &miner->buf_ht[round]);
			check_clSetKernelArg(miner->k_rounds[round], 2, &miner->rowCounters[(round - 1) % 2]);
			check_clSetKernelArg(miner->k_rounds[round], 3, &miner->rowCounters[round % 2]);
			miner->global_ws = GLOBAL_WORK_SIZE_RATIO * compute_units * LOCAL_WORK_SIZE;
			if (miner->global_ws > NR_ROWS * THREADS_PER_ROW)
				miner->global_ws = NR_ROWS * THREADS_PER_ROW;
		}
		check_clSetKernelArg(miner->k_rounds[round], round == 0 ? 3 : 4, &miner->buf_dbg);
		OCL(clEnqueueNDRangeKernel(miner->queue, miner->k_rounds[round], 1, NULL,
			&miner->global_ws, &miner->local_work_size, 0, NULL, NULL));
		// cancel function
		if (cancelf()) return;
	}

	check_clSetKernelArg(miner->k_sols, 0, &miner->buf_ht[0]);
	check_clSetKernelArg(miner->k_sols, 1, &miner->buf_ht[1]);
	check_clSetKernelArg(miner->k_sols, 2, &miner->buf_ht[2]);
	check_clSetKernelArg(miner->k_sols, 3, &miner->buf_ht[3]);
	check_clSetKernelArg(miner->k_sols, 4, &miner->buf_ht[4]);
	check_clSetKernelArg(miner->k_sols, 5, &miner->buf_ht[5]);
	check_clSetKernelArg(miner->k_sols, 6, &miner->buf_ht[6]);
	check_clSetKernelArg(miner->k_sols, 7, &miner->buf_ht[7]);
	check_clSetKernelArg(miner->k_sols, 8, &miner->buf_ht[8]);
	check_clSetKernelArg(miner->k_sols, 9, &miner->buf_sols);
	check_clSetKernelArg(miner->k_sols, 10, &miner->rowCounters[0]);
	miner->global_ws = GLOBAL_WORK_SIZE_RATIO * compute_units * LOCAL_WORK_SIZE_SOLS;
	if (miner->global_ws > NR_ROWS * THREADS_PER_ROW_SOLS)
		miner->global_ws = NR_ROWS * THREADS_PER_ROW_SOLS;
	miner->local_work_size = LOCAL_WORK_SIZE_SOLS;
	struct timeval start_time;
	gettimeofday(&start_time, NULL);
	OCL(clEnqueueNDRangeKernel(miner->queue, miner->k_sols, 1, NULL,
		&miner->global_ws, &miner->local_work_size, 0, NULL, NULL));
	
	OCL(clEnqueueReadBuffer(miner->queue, miner->buf_sols,
		CL_TRUE,	// cl_bool	blocking_read
		0,		// size_t	offset
		sizeof(*miner->sols),	// size_t	size
		miner->sols,	// void		*ptr
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL));	// cl_event	*event

	if (miner->sols->nr > MAX_SOLS)
		miner->sols->nr = MAX_SOLS;

	clReleaseMemObject(buf_blake_st);

	for (unsigned sol_i = 0; sol_i < miner->sols->nr; sol_i++) {
		verify_sol(miner->sols, sol_i);
	}

	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < miner->sols->nr; i++) {
		if (miner->sols->valid[i]) {
			compress(proof, (uint32_t *)(miner->sols->values[i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}
	}
	hashdonef();
}

// STATICS END

