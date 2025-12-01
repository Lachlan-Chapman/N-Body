#include <cstdio>

__global__ void testKernel() {
	printf("Hello from kernel thread %d\n", threadIdx.x);
}

extern "C" void runTestKernel() { //ensures no name mangling when compiling
	testKernel<<<1, 8>>>();
	cudaDeviceSynchroniz();
}