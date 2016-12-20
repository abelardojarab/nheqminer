#pragma once
#include "param.h"
#include <CL/cl.h>

struct ocl_gg_context {
	cl_context _context;
	cl_program _program;
	cl_device_id _dev_id;

	cl_platform_id platform_id = 0;

	cl_command_queue queue;

	cl_kernel k_init_ht;
	cl_kernel k_rounds[PARAM_K];
	cl_kernel k_sols;

	cl_mem buf_ht[2], buf_sols, buf_dbg, rowCounters[2];
	size_t global_ws;
	size_t local_work_size = 64;

	sols_t	*sols;

	bool init(cl_device_id dev, unsigned threadsNum, unsigned threadsPerBlock);

	~ocl_gg_context() {
		clReleaseMemObject(buf_dbg);
		clReleaseMemObject(buf_ht[0]);
		clReleaseMemObject(buf_ht[1]);
		clReleaseMemObject(rowCounters[0]);
		clReleaseMemObject(rowCounters[1]);
		free(sols);
	}
};
