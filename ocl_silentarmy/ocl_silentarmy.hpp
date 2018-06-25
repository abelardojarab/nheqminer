#pragma once
#ifdef _LIB
#define DLL_OCL_SILENTARMY __declspec(dllexport)
#else
#define DLL_OCL_SILENTARMY
#endif

// remove after
#include <cstdio>
#include <string>
#include <functional>
#include <vector>
#include <cstdint>

#pragma warning(disable: 4996)
#include <CL/cl.h>

// extern cl_context gContext;

#define OCL(error)							\
	if(cl_int err = error){						\
		printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__); \
		return;							\
	}								\

#define OCLR(error, ret)						\
	if(cl_int err = error){						\
		printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__); \
		return ret;						\
	}								\

#define OCLE(error)							\
	if(cl_int err = error){						\
		printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__); \
		exit(err);						\
	}								\





template<typename T>
class clBuffer {
public:

	clBuffer() {

		Size = 0;
		HostData = 0;
		DeviceData = 0;

	}

	~clBuffer() {

		if(HostData)
			delete [] HostData;

		if(DeviceData)
			clReleaseMemObject(DeviceData);

	}

	void init(cl_context gContext, int size, cl_mem_flags flags = 0) {

		Size = size;

		if(!(flags & CL_MEM_HOST_NO_ACCESS)){
			HostData = new T[Size];
			memset(HostData, 0, Size*sizeof(T));
		}else
			HostData = 0;

		//printf("clCreateBuffer: size = %d, %d bytes\n", Size, Size*sizeof(T));

		cl_int error;
		if (flags & CL_MEM_HOST_NO_ACCESS)
			flags = CL_MEM_READ_WRITE;
		DeviceData = clCreateBuffer(gContext, flags, Size*sizeof(T), 0, &error);
		OCL(error);

	}

	void copyToDevice(cl_command_queue cq, bool blocking = true) {

		OCL(clEnqueueWriteBuffer(cq, DeviceData, blocking, 0, Size*sizeof(T), HostData, 0, 0, 0));

	}

	void copyToHost(cl_command_queue cq, bool blocking = true, unsigned size = 0) {

		if(size == 0)
			size = Size;

		OCL(clEnqueueReadBuffer(cq, DeviceData, blocking, 0, size*sizeof(T), HostData, 0, 0, 0));

	}

	T& get(int index) {
		return HostData[index];
	}

	T& operator[](int index) {
		return HostData[index];
	}

public:

	int Size;
	T* HostData;
	cl_mem DeviceData;


};


std::vector<cl_device_id> GetAllDevices(int platform_id = -1);

bool clInitialize(int requiredPlatform, std::vector<cl_device_id> &gpus);
bool clCompileKernel(cl_context gContext,
                     cl_device_id gpu,
                     const char *binaryName,
                     const std::vector<const char*> &sources,
                     const char *arguments,
                     cl_int *binstatus,
                     cl_program *gProgram);

struct OclContext;

struct DLL_OCL_SILENTARMY ocl_silentarmy
{
	//int threadsperblock;
	int blocks;
	int device_id;
	int platform_id;

	OclContext* oclc;
	// threads
	unsigned threadsNum; // TMP
	unsigned wokrsize;

	bool is_init_success = false;

	ocl_silentarmy(int platf_id, int dev_id);

	std::string getdevinfo();

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
