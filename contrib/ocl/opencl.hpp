#pragma once

#include <CL/cl.h>

#if defined(_MSC_VER)
#pragma comment (lib, "opencl.lib")
#endif

#include <ocl/hex.hpp>
#include <ocl/cl_ext.hpp>
#include <ocl/utility/device_utils.hpp>

#include <cstdio>

namespace ocl {
	
inline cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size, void *host_ptr)
{
	cl_int	status;
	cl_mem	ret;
	ret = clCreateBuffer(ctx, flags, size, host_ptr, &status);
	if (status != CL_SUCCESS || !ret)
		printf("clCreateBuffer (%d)\n", status);
	return ret;
}

inline void check_clSetKernelArg(cl_kernel k, cl_uint a_pos, cl_mem *a)
{
	cl_int	status;
	status = clSetKernelArg(k, a_pos, sizeof(*a), a);
	if (status != CL_SUCCESS)
		printf("clSetKernelArg (%d)\n", status);
}	
	
inline void check_clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel k, cl_uint
	work_dim, const size_t *global_work_offset, const size_t
	*global_work_size, const size_t *local_work_size, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_uint	status;
	status = clEnqueueNDRangeKernel(queue, k, work_dim, global_work_offset,
		global_work_size, local_work_size, num_events_in_wait_list,
		event_wait_list, event);
	if (status != CL_SUCCESS)
		printf("clEnqueueNDRangeKernel (%d)\n", status);
}

inline void check_clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool
	blocking_read, size_t offset, size_t size, void *ptr, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_int	status;
	status = clEnqueueReadBuffer(queue, buffer, blocking_read, offset,
		size, ptr, num_events_in_wait_list, event_wait_list, event);
	if (status != CL_SUCCESS)
		printf("clEnqueueReadBuffer (%d)\n", status);
}


inline unsigned nr_compute_units(cl_device_id device_id)
{
	cl_uint retval;
	cl_int status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &retval, nullptr);
	if (status != CL_SUCCESS)
		printf("nr_compute_units (%d)\n", status);
	return retval;
}

	
}
