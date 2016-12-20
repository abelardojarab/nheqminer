#pragma once

#include <ocl/opencl.hpp>
#include <ocl/algorithm/detail/gatelessgate_context.hpp>
#include <string>
#include <vector>

namespace ocl {
namespace algorithm {

struct gatelessgate {
	
	int blocks;
	int device_id;
	int platform_id;

	algorithm_detail::gatelessgate_context* oclc;
	// threads
	unsigned threadsNum; // TMP
	unsigned wokrsize;

	bool is_init_success = false;

	gatelessgate(int platf_id, int dev_id) 
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

	static void getinfo(int platf_id, int d_id, ::std::string& gpu_name, int& sm_count, ::std::string& version) {
		static auto devices = utility::GetAllDevices();

		if (devices.size() <= d_id) {
			return;
		}
		auto device = devices[d_id];

		::std::vector<char> name(256, 0);
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

	static void start(gatelessgate& device_context) {
		
	}

	static void stop(gatelessgate& device_context) {
		
	}

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		::std::function<bool()> cancelf,
		::std::function<void(const ::std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		::std::function<void(void)> hashdonef,
		gatelessgate& device_context) {
			
		}

	std::string getname() const { return "OCL_GATELESSGATE"; }

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
	::std::string m_gpu_name;
	::std::string m_version;
	
};

	
}
}

