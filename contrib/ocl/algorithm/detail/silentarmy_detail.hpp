#pragma once
#include <ocl/opencl.hpp>
#include <ocl/sols.hpp>
#include <cstdint>

namespace ocl {
namespace algorithm {
namespace algorithm_detail {

inline void init_ht(cl_command_queue queue, cl_kernel k_init_ht, cl_mem buf_ht, cl_mem rowCounters)
{
	size_t      global_ws = SA_NR_ROWS / SA_ROWS_PER_UINT;
	size_t      local_ws = 256;
	cl_int      status;
#if 0
	uint32_t    pat = -1;
	status = clEnqueueFillBuffer(queue, buf_ht, &pat, sizeof(pat), 0,
		SA_NR_ROWS * SA_NR_SLOTS * SA_SLOT_LEN,
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
	check_clEnqueueNDRangeKernel(queue, k_init_ht,
		1,		// cl_uint	work_dim
		NULL,	// size_t	*global_work_offset
		&global_ws,	// size_t	*global_work_size
		&local_ws,	// size_t	*local_work_size
		0,		// cl_uint	num_events_in_wait_list
		NULL,	// cl_event	*event_wait_list
		NULL);	// cl_event	*event
}


/*
** Sort a pair of binary blobs (a, b) which are consecutive in memory and
** occupy a total of 2*len 32-bit words.
**
** a            points to the pair
** len          number of 32-bit words in each pair
*/
inline void sort_pair(uint32_t *a, uint32_t len)
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

inline uint32_t verify_sol(sols_t *sols, unsigned sol_i)
{
	uint32_t  *inputs = sols->values[sol_i];
	uint32_t  seen_len = (1 << (SA_PREFIX + 1)) / 8;
	uint8_t seen[(1 << (SA_PREFIX + 1)) / 8];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, seen_len);
	for (i = 0; i < (1 << SA_PARAM_K); i++)
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
	for (uint32_t level = 0; level < SA_PARAM_K; level++)
		for (i = 0; i < (1 << SA_PARAM_K); i += (2 << level))
			sort_pair(&inputs[i], 1 << level);
	return 1;
}


inline size_t select_work_size_blake(cl_device_id device_id)
{
	
	size_t work_size =
		64 * /* thread per wavefront */
		SA_BLAKE_WPS * /* wavefront per simd */
		4 * /* simd per compute unit */
		nr_compute_units(device_id);
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	while (SA_NR_INPUTS % work_size)
		work_size += 64;

	return work_size;
}


}
}
}