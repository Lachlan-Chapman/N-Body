#include <iostream>

#include <cuda_gl_interop.h>
#include "graphics/Cuda.hpp"

namespace Cuda {
	bool createGPU() {
		unsigned int gpu_count;
		int device_ids[4];
		cudaError_t err = cudaGLGetDevices(
			&gpu_count,
			device_ids,
			4, //how many devices im asking about, my system has 1 4080
			cudaGLDeviceListAll //any and all types of devices tell me about
		);

		if(err != cudaSuccess) {
			std::cerr << "cudaGLGetDevices() Failed: " << cudaGetErrorString(err) << "\n";
			return false;
		}

		if(gpu_count == 0) {
			std::cerr << "no CUDA devices compatible with the current OpenGL context\n";
			return false;
		}

		cudaSetDevice(device_ids[0]); //set the first gpu found as the gpu to use for our context
		return true;
	}

	void* malloc(size_t p_size) {
		void *_ptr;
		cudaError_t _err = cudaMalloc(&_ptr, p_size);
		if(_err != cudaSuccess) {
			std::cerr << "ERROR: Cuda::malloc() | " << cudaGetErrorString(_err) << std::endl;
			return nullptr;
		}
		return _ptr;
	}

	void* unifiedMalloc(size_t p_size) {
		void *_ptr;
		cudaError_t _err = cudaMallocManaged(&_ptr, p_size);
		if(_err != cudaSuccess) {
			std::cerr << "ERROR: Cuda::malloc() | " << cudaGetErrorString(_err) << std::endl;
			return nullptr;
		}
		return _ptr;
	}

	

	void free(void *p_ptr) {
		cudaFree(p_ptr);
	}

}