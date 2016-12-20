#pragma once
#include <ocl/opencl.hpp>
#include <ocl/sols.hpp>
#include <cstdint>

namespace ocl {
namespace algorithm {
namespace algorithm_detail {



struct silentarmy_context {
	cl_context _context;
	cl_program _program;
	cl_device_id _dev_id;
	cl_platform_id platform_id = 0;
	cl_command_queue queue;


	cl_kernel k_init_ht;
	cl_kernel k_rounds[SA_PARAM_K];
	cl_kernel k_sols;

	cl_mem buf_ht[2], buf_sols, buf_dbg, rowCounters[2];
	size_t global_ws;
	size_t local_work_size = 64;

	sols_t	*sols;

	bool init(cl_device_id dev, unsigned threadsNum, unsigned threadsPerBlock) {
		cl_int error;

		queue = clCreateCommandQueue(_context, dev, 0, &error);

	#ifdef SA_ENABLE_DEBUG
		size_t              dbg_size = SA_NR_ROWS;
	#else
		size_t              dbg_size = 1;
	#endif

		buf_dbg = check_clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, dbg_size, NULL);
		buf_ht[0] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, SA_HT_SIZE, NULL);
		buf_ht[1] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, SA_HT_SIZE, NULL);
		buf_sols = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, sizeof(sols_t), NULL);

		rowCounters[0] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, SA_NR_ROWS, NULL);
		rowCounters[1] = check_clCreateBuffer(_context, CL_MEM_READ_WRITE, SA_NR_ROWS, NULL);



		fprintf(stderr, "Hash tables will use %.1f MB\n", 2.0 * SA_HT_SIZE / 1e6);

		k_init_ht = clCreateKernel(_program, "kernel_init_ht", &error);
		for (unsigned i = 0; i < SA_PARAM_K; i++) {
			char kernelName[128];
			sprintf(kernelName, "kernel_round%d", i);
			k_rounds[i] = clCreateKernel(_program, kernelName, &error);
		}

		sols = (sols_t *)malloc(sizeof(*sols));

		k_sols = clCreateKernel(_program, "kernel_sols", &error);
		return true;		
		
		
	}
	
	~silentarmy_context() {
		clReleaseMemObject(buf_dbg);
		clReleaseMemObject(buf_ht[0]);
		clReleaseMemObject(buf_ht[1]);
		clReleaseMemObject(rowCounters[0]);
		clReleaseMemObject(rowCounters[1]);
		free(sols);
	}
	
	
};
		
}	
}
}