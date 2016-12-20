#pragma once
#include <vector>
#include <iostream>
#include <ocl/cl_ext.hpp>//reinclude just in case..
#include <fstream>

typedef std::vector<cl_device_id> ocl_devices;

#define OCL(error) \
  if(cl_int err = error){ \
    printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__); \
    return; \
  }

#define OCLR(error, ret) \
  if(cl_int err = error){ \
    printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__); \
    return ret; \
  }

#define OCLE(error) \
  if(cl_int err = error){ \
    printf("OpenCL error: %d at %s:%d\n", err, __FILE__, __LINE__); \
    exit(err); \
  }

namespace ocl {
namespace utility {
	inline ocl_devices GetAllDevices() {
		ocl_devices retval;
		retval.reserve(8);

		cl_platform_id platforms[64];
		cl_uint numPlatforms;
		cl_int rc = clGetPlatformIDs(sizeof(platforms) / sizeof(cl_platform_id), platforms, &numPlatforms);

		for (cl_uint i = 0; i < numPlatforms; i++) {
			cl_uint numDevices = 0;
			cl_device_id devices[64];
			rc = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, sizeof(devices) / sizeof(cl_device_id), devices, &numDevices);
			for (cl_uint n = 0; n < numDevices; n++) {
				retval.push_back(devices[n]);
			}
		}

		return retval;
	}

	inline void PrintDevices() {
		using namespace ::std;

		auto devices = GetAllDevices();
		cout << "Number of OpenCL devices found: " << devices.size() << endl;
		for (unsigned int i = 0; i < devices.size(); ++i) {
			cl::Device device(devices[i]);
			auto& platform = cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>());
			cout << "Device #" << i << " | " << platform.getInfo<CL_PLATFORM_NAME>() << " | " << device.getInfo<CL_DEVICE_NAME>();

			switch (device.getInfo<CL_DEVICE_TYPE>()) {
			case CL_DEVICE_TYPE_CPU:
				cout << " | CPU";
				break;
			case CL_DEVICE_TYPE_GPU:
				cout << " | GPU";
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				cout << " | ACCELERATOR";
				break;
			default:
				cout << " | DEFAULT";
				break;
			}
			cout << " | " << device.getInfo<CL_DEVICE_VERSION>();
			cout << endl;
		}
	}

	inline bool clCompileKernel(cl_context gContext,
		cl_device_id gpu,
		const char *binaryName,
		const std::vector<const char*> &sources,
		const char *arguments,
		cl_int *binstatus,
		cl_program *gProgram)
	{
		std::ifstream testfile(binaryName);

		if (!testfile) {
			printf("<info> compiling ...\n");

			std::string sourceFile;
			for (auto &i : sources) {
				std::ifstream stream;
				stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
				try {
					stream.open(i);
				}
				catch (std::system_error& e) {
					fprintf(stderr, "<error> %s\n", e.code().message().c_str());
					return false;
				}
				std::string str((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
				sourceFile.append(str);
			}

			printf("<info> source: %u bytes\n", (unsigned)sourceFile.size());
			if (sourceFile.size() < 1) {
				fprintf(stderr, "<error> source files not found or empty\n");
				return false;
			}

			cl_int error;
			const char *sources[] = { sourceFile.c_str(), 0 };
			*gProgram = clCreateProgramWithSource(gContext, 1, sources, 0, &error);
			OCLR(error, false);

			if (clBuildProgram(*gProgram, 1, &gpu, arguments, 0, 0) != CL_SUCCESS) {
				size_t logSize;
				clGetProgramBuildInfo(*gProgram, gpu, CL_PROGRAM_BUILD_LOG, 0, 0, &logSize);

				std::unique_ptr<char[]> log(new char[logSize]);
				clGetProgramBuildInfo(*gProgram, gpu, CL_PROGRAM_BUILD_LOG, logSize, log.get(), 0);
				printf("%s\n", log.get());

				return false;
			}

			size_t binsize;
			OCLR(clGetProgramInfo(*gProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binsize, 0), false);
			//     for (size_t i = 0; i < 1; i++) {
			if (!binsize) {
				printf("<error> no binary available!\n");
				return false;
			}
			//     }

			printf("<info> binsize = %u bytes\n", (unsigned)binsize);
			std::unique_ptr<unsigned char[]> binary(new unsigned char[binsize + 1]);

			OCLR(clGetProgramInfo(*gProgram, CL_PROGRAM_BINARIES, sizeof(void*), &binary, 0), false);

			{
				std::ofstream bin(binaryName, std::ofstream::binary | std::ofstream::trunc);
				bin.write((const char*)binary.get(), binsize);
				bin.close();
			}

			OCLR(clReleaseProgram(*gProgram), false);
		}

		std::ifstream bfile(binaryName, std::ifstream::binary);
		if (!bfile) {
			printf("<error> %s not found\n", binaryName);
			return false;
		}

		bfile.seekg(0, bfile.end);
		size_t binsize = bfile.tellg();
		bfile.seekg(0, bfile.beg);
		if (!binsize) {
			printf("<error> %s empty\n", binaryName);
			return false;
		}

		std::vector<char> binary(binsize + 1);
		bfile.read(&binary[0], binsize);
		bfile.close();

		cl_int error;
		//   binstatus.resize(gpus.size(), 0);
		//   std::vector<size_t> binsizes(gpus.size(), binsize);
		//   std::vector<const unsigned char*> binaries(gpus.size(), (const unsigned char*)&binary[0]);
		const unsigned char *binaryPtr = (const unsigned char*)&binary[0];

		*gProgram = clCreateProgramWithBinary(gContext, 1, &gpu, &binsize, &binaryPtr, binstatus, &error);
		OCLR(error, false);
		OCLR(clBuildProgram(*gProgram, 1, &gpu, 0, 0, 0), false);
		return true;
	}
}
}