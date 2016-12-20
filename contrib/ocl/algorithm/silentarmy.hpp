#pragma once

#include <ocl/opencl.hpp>
#include <ocl/sols.hpp>
#include <ocl/algorithm/detail/silentarmy_context.hpp>
#include <ocl/algorithm/detail/silentarmy_detail.hpp>
#include <ocl/algorithm/compress.hpp>
#include <ocl/crypto/blake.hpp>
#include <string>
#include <vector>
#include <functional>


#define SA_COLLISION_BIT_LENGTH (SA_PARAM_N / (SA_PARAM_K+1))
#define SA_COLLISION_BYTE_LENGTH ((SA_COLLISION_BIT_LENGTH+7)/8)
#define SA_FINAL_FULL_WIDTH (2*SA_COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (SA_PARAM_K)))

#define SA_NDIGITS   (SA_PARAM_K+1)
#define SA_DIGITBITS (SA_PARAM_N/(SA_NDIGITS))
#define SA_PROOFSIZE (1u<<SA_PARAM_K)
#define SA_COMPRESSED_PROOFSIZE ((SA_COLLISION_BIT_LENGTH+1)*SA_PROOFSIZE*4/(8*sizeof(uint32_t)))

namespace ocl {
namespace algorithm {
	
struct silentarmy {
	
	int blocks;
	int device_id;
	int platform_id;

	algorithm_detail::silentarmy_context* oclc;
	// threads
	unsigned threadsNum; // TMP
	unsigned wokrsize;

	bool is_init_success = false;

	silentarmy(int platf_id, int dev_id) 
	: blocks(0)
	, device_id(dev_id)
	, platform_id(platf_id)
	, oclc(nullptr)
	, threadsNum(8192U)
	, wokrsize(128)
	{
	
	}

	static int getcount() {
		static auto devices = utility::GetAllDevices();
		return devices.size();
	}

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version) {
		static auto devices = utility::GetAllDevices();

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

	static void start(silentarmy& device_context) {
		device_context.is_init_success = false;
		device_context.oclc = new algorithm_detail::silentarmy_context;
		auto devices = utility::GetAllDevices();

		auto& device = devices[device_context.device_id];

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

		char kernelName[64];
		sprintf(kernelName, "silentarmy_gpu_%u.bin", (unsigned)device_context.device_id);
		if (!utility::clCompileKernel(device_context.oclc->_context,
			device,
			kernelName,
			{ "zcash/gpu/silentarmy.cl" },
			"",
			&binstatus,
			&device_context.oclc->_program)) {
			return;
		}

		if (binstatus == CL_SUCCESS) {
			if (!device_context.oclc->init(device, device_context.threadsNum, device_context.wokrsize)) {
				printf("Init failed");
				return;
			}
		} else {
			printf("GPU %d: failed to load kernel\n", device_context.device_id);
			return;
		}

		device_context.is_init_success = true;		
	}

	static void stop(silentarmy& device_context) {
		if (device_context.oclc != nullptr) delete device_context.oclc;
	}

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		silentarmy& device_context) {
			using namespace ocl::crypto;
			using namespace algorithm_detail;
			
			unsigned char context[140];
			memset(context, 0, 140);
			memcpy(context, tequihash_header, tequihash_header_len);
			memcpy(context + tequihash_header_len, nonce, nonce_len);

			auto *miner = device_context.oclc;
			clFlush(miner->queue);

			blake2b_state_t initialCtx;
			zcash_blake2b_init(&initialCtx, SA_ZCASH_HASH_LEN, SA_PARAM_N, SA_PARAM_K);
			zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

			cl_mem buf_blake_st;
			buf_blake_st = check_clCreateBuffer(miner->_context, CL_MEM_READ_ONLY |
				CL_MEM_COPY_HOST_PTR, sizeof(blake2b_state_s), &initialCtx);

			for (unsigned round = 0; round < SA_PARAM_K; round++)
			{
				init_ht(miner->queue, miner->k_init_ht, miner->buf_ht[round & 1], miner->rowCounters[round & 1]);
				if (!round)
				{
					check_clSetKernelArg(miner->k_rounds[round], 0, &buf_blake_st);
					check_clSetKernelArg(miner->k_rounds[round], 1, &miner->buf_ht[round & 1]);
					check_clSetKernelArg(miner->k_rounds[round], 2, &miner->rowCounters[round & 2]);
					miner->global_ws = select_work_size_blake(miner->_dev_id);
				}
				else
				{
					check_clSetKernelArg(miner->k_rounds[round], 0, &miner->buf_ht[(round - 1) & 1]);
					check_clSetKernelArg(miner->k_rounds[round], 1, &miner->buf_ht[round & 1]);
					check_clSetKernelArg(miner->k_rounds[round], 2, &miner->rowCounters[(round - 1) & 1]);
					check_clSetKernelArg(miner->k_rounds[round], 3, &miner->rowCounters[round & 1]);
					miner->global_ws = SA_NR_ROWS;
				}
				check_clSetKernelArg(miner->k_rounds[round], round == 0 ? 3 : 4, &miner->buf_dbg);
				if (round == SA_PARAM_K - 1)
					check_clSetKernelArg(miner->k_rounds[round], 5, &miner->buf_sols);
				check_clEnqueueNDRangeKernel(miner->queue, miner->k_rounds[round], 1, NULL,
					&miner->global_ws, &miner->local_work_size, 0, NULL, NULL);
				// cancel function
				if (cancelf()) return;
			}
			check_clSetKernelArg(miner->k_sols, 0, &miner->buf_ht[0]);
			check_clSetKernelArg(miner->k_sols, 1, &miner->buf_ht[1]);
			check_clSetKernelArg(miner->k_sols, 2, &miner->buf_sols);
			check_clSetKernelArg(miner->k_sols, 3, &miner->rowCounters[0]);
			check_clSetKernelArg(miner->k_sols, 4, &miner->rowCounters[1]);
			miner->global_ws = SA_NR_ROWS;
			check_clEnqueueNDRangeKernel(miner->queue, miner->k_sols, 1, NULL,
				&miner->global_ws, &miner->local_work_size, 0, NULL, NULL);

			check_clEnqueueReadBuffer(miner->queue, miner->buf_sols,
				CL_TRUE,	// cl_bool	blocking_read
				0,		// size_t	offset
				sizeof(*miner->sols),	// size_t	size
				miner->sols,	// void		*ptr
				0,		// cl_uint	num_events_in_wait_list
				NULL,	// cl_event	*event_wait_list
				NULL);	// cl_event	*event

			if (miner->sols->nr > SA_MAX_SOLS)
				miner->sols->nr = SA_MAX_SOLS;

			clReleaseMemObject(buf_blake_st);

			for (unsigned sol_i = 0; sol_i < miner->sols->nr; sol_i++) {
				verify_sol(miner->sols, sol_i);
			}

			uint8_t proof[SA_COMPRESSED_PROOFSIZE * 2];
			for (uint32_t i = 0; i < miner->sols->nr; i++) {
				if (miner->sols->valid[i]) {
					compress<SA_PREFIX>(proof, (uint32_t *)(miner->sols->values[i]), 1 << SA_PARAM_K);
					solutionf(std::vector<uint32_t>(0), 1344, proof);
				}
			}
			hashdonef();
		}

	std::string getname() const { return "OCL_SILENTARMY"; }

	std::string getdevinfo() {
		static auto devices = ocl::utility::GetAllDevices();
		auto device = devices[device_id];
		std::vector<char> name(256, 0);
		size_t nActualSize = 0;
		std::string gpu_name;

		cl_int rc = clGetDeviceInfo(device, CL_DEVICE_NAME, name.size(), &name[0], &nActualSize);

		gpu_name.assign(&name[0], nActualSize);

		return "GPU_ID( " + gpu_name + ")";		
	}
	
private:
	std::string m_gpu_name;
	std::string m_version;
	
};

}
}