#pragma once

#include "ocl_gg_context.hpp"
#include <string>

namespace gg {
namespace impl {

}
	struct ocl_gatelessgate
	{
		//int threadsperblock;
		int blocks;
		int device_id;
		int platform_id;

		ocl_gg_context* oclc;
		// threads
		unsigned threadsNum; // TMP
		unsigned wokrsize;

		bool is_init_success = false;

		ocl_gatelessgate(int platf_id, int dev_id)
			: blocks(0)
			, device_id(dev_id)
			, platform_id(platf_id)
			, oclc(nullptr)
			, threadsNum(8192U)
			, wokrsize(128U)
			, is_init_success(false)
		{
		}

		std::string getdevinfo() {
			static auto devices = GetAllDevices();
			auto device = devices[device_id];
			std::vector<char> name(256, 0);
			size_t nActualSize = 0;
			std::string gpu_name;

			cl_int rc = clGetDeviceInfo(device, CL_DEVICE_NAME, name.size(), &name[0], &nActualSize);

			gpu_name.assign(&name[0], nActualSize);

			return "GPU_ID( " + gpu_name + ")";
		}

		static int getcount();

		static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

		static void start(ocl_silentarmy& device_context);

		static void stop(ocl_silentarmy& device_context);

		static void solve(const char *tequihash_header,
			unsigned int tequihash_header_len,
			const char* nonce,
			unsigned int nonce_len,
			std::function<bool()> cancelf,
			std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
			std::function<void(void)> hashdonef,
			ocl_silentarmy& device_context);

		std::string getname() { return "OCL_SILENTARMY"; }

	private:
		std::string m_gpu_name;
		std::string m_version;
	};

}