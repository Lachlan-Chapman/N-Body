#pragma once
#pragma once

#ifdef __CUDACC__
	#define HD __host__ __device__ //cpp files will see this as if cuda doesnt exist at all
#else
	#define HD
#endif

namespace Cuda {
	bool createGPU();
}