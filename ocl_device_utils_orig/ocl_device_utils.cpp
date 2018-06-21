#include "ocl_device_utils.h"
#include "opencl.h"
#include <iostream>
#include <stdexcept>
#include <utility>
#include <algorithm>

using namespace std;
using namespace cl;


bool ocl_device_utils::_hasQueried = false;
std::vector<std::string> ocl_device_utils::_platformNames;
std::vector<PrintInfo> ocl_device_utils::_devicesPlatformsDevices;
std::vector<cl::Device> ocl_device_utils::_AllDevices;


vector<Platform> ocl_device_utils::getPlatforms() {
	vector<Platform> platforms;
	try {
		Platform::get(&platforms);
	}
	catch (Error const& err) {
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
		if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
			cout << "No OpenCL platforms found" << endl;
		else
#endif
			throw err;
	}
	return platforms;
}

void ocl_device_utils::print_opencl_devices() {
	ocl_device_utils::QueryDevices();
	ocl_device_utils::PrintDevices();
}

vector<Device> ocl_device_utils::getDevices(vector<Platform> const& _platforms, unsigned _platformId) {
	vector<Device> devices;
	try {
		auto cl_devices = GetAllDevices();
		for (auto& cl_device : cl_devices) {
			devices.push_back({ cl_device });
		}
	}
	catch (Error const& err) {
		// if simply no devices found return empty vector
		if (err.err() != CL_DEVICE_NOT_FOUND)
			throw err;
	}
	return devices;
}

string ocl_device_utils::StringnNullTerminatorFix(const string& str) {
	return string(str.c_str(), strlen(str.c_str()));
}

bool ocl_device_utils::QueryDevices() {
	if (!_hasQueried) {
		_hasQueried = true;
		try {
			auto devices = GetAllDevices();

			unsigned int device_num = 0;
			for (auto& cldevice : devices) {
				cl::Device device(cldevice);
				cl::Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
				_AllDevices.emplace_back(device);
				PrintInfo current;
				current.PlatformName = StringnNullTerminatorFix(platform.getInfo<CL_PLATFORM_NAME>());
				current.PlatformNum = 0;
				OpenCLDevice curDevice;
				curDevice.DeviceID = device_num++;
				curDevice._CL_DEVICE_NAME = StringnNullTerminatorFix(device.getInfo<CL_DEVICE_NAME>());

				switch (device.getInfo<CL_DEVICE_TYPE>()) {
					case CL_DEVICE_TYPE_CPU:
						curDevice._CL_DEVICE_TYPE = "CPU";
						break;
					case CL_DEVICE_TYPE_GPU:
						curDevice._CL_DEVICE_TYPE = "GPU";
						break;
					case CL_DEVICE_TYPE_ACCELERATOR:
						curDevice._CL_DEVICE_TYPE = "ACCELERATOR";
						break;
					default:
						curDevice._CL_DEVICE_TYPE = "DEFAULT";
						break;
				}

				curDevice._CL_DEVICE_GLOBAL_MEM_SIZE = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
				curDevice._CL_DEVICE_VENDOR = StringnNullTerminatorFix(device.getInfo<CL_DEVICE_VENDOR>());
				curDevice._CL_DEVICE_VERSION = StringnNullTerminatorFix(device.getInfo<CL_DEVICE_VERSION>());
				curDevice._CL_DRIVER_VERSION = StringnNullTerminatorFix(device.getInfo<CL_DRIVER_VERSION>());
				current.Devices.push_back(curDevice);
				_devicesPlatformsDevices.push_back(current);
			}
		}
		catch (exception &ex) {
			// TODO
			cout << "ocl_device_utils::QueryDevices() exception: " << ex.what() << endl;
			return false;
		}
		return true;
	}
	
	return false;
}

int ocl_device_utils::GetCountForPlatform(int platformID) {
	for (const auto &platInfo : _devicesPlatformsDevices)
	{
		if (platformID == platInfo.PlatformNum) {
			return (int)platInfo.Devices.size();
		}
	}
	return 0;
}

void ocl_device_utils::PrintDevices() {
	cout << "Number of OpenCL devices found: " << _AllDevices.size() << endl;
	for (unsigned int i = 0; i < _AllDevices.size(); ++i) {
		auto& item = _AllDevices[i];
		auto& platform = cl::Platform(item.getInfo<CL_DEVICE_PLATFORM>());
		cout << "Device #" << i << " | " << platform.getInfo<CL_PLATFORM_NAME>() << " | " << item.getInfo<CL_DEVICE_NAME>();

		switch (item.getInfo<CL_DEVICE_TYPE>()) {
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
		cout << " | " << item.getInfo<CL_DEVICE_VERSION>();
		cout << endl;
	}
}