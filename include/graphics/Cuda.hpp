#pragma once

#ifdef __CUDACC__
	#define host_device __host__ __device__ //cpp files will see this as if cuda doesnt exist at all
	#define host __host__ //cpp files will see this as if cuda doesnt exist at all
	#define device __device__
#else
	#define host_device
	#define host
	#define device
#endif

namespace Cuda {
	bool createGPU();
	void* malloc(size_t p_size); //size in bytes
	void* unifiedMalloc(size_t p_size); //creates a shared memory space that auto manages what memory the data belongs into either host or gpu
	void free(void *p_ptr);
}