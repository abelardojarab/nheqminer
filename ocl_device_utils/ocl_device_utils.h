#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "cl_ext.hpp"
#include <map>
#include <vector>
#include <stdio.h>
#include <string.h>

#include "OpenCLDevice.h"
#include <CL/cl.h>
#pragma warning(disable: 4996)

struct PrintInfo {
	std::string PlatformName;
	int PlatformNum;
	std::vector<OpenCLDevice> Devices;
};

class ocl_device_utils {
public:
	static bool QueryDevices();
	static void PrintDevices();
	static int GetCountForPlatform(int platformID);
	static void print_opencl_devices();

private:
	static std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId);
	static std::vector<cl::Platform> getPlatforms();

	static bool _hasQueried;
	static std::vector<std::string> _platformNames;
	static std::vector<PrintInfo> _devicesPlatformsDevices;
	static std::vector<cl::Device> _AllDevices;

	static std::string StringnNullTerminatorFix(const std::string& str);
};

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
