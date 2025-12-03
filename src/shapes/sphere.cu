#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "shapes/sphere.hpp"

__device__ constexpr float GOLDEN_ANGLE() { return 2.39996323; }

__global__ void buildFibonnaciSphere(float *p_positions, int p_vertexCount) {
	float N_INVERSE = 1.0f / (p_vertexCount - 1.0f);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= p_vertexCount) { return; }

	float N = (float)(p_vertexCount - 1);
	float y = 1.0f - (2.0f * idx / N);
	float radius = sqrt(1.0f - y*y); //sqrt(1-y^2) manipulated to use the fused multiply add instr
	float phi = idx * GOLDEN_ANGLE();
	
	p_positions[idx * 3 + 0] = radius * cosf(phi); //x
	p_positions[idx * 3 + 1] = y;
	p_positions[idx * 3 + 2] = radius * sinf(phi); //z

	// p_positions[idx * 3 + 0] = (float)idx;
	// p_positions[idx * 3 + 1] = (float)idx;
	// p_positions[idx * 3 + 2] = (float)idx;
	//printf("Thread %d: hello\n%f, %f, %f\n\n", idx, p_positions[idx * 3 + 0], p_positions[idx * 3 + 1], p_positions[idx * 3 + 2]);
}

void fibonacciSphere(float *p_positions, int p_vertexCount) {
	int block_size = 256;
	int grid_size = (p_vertexCount + block_size - 1) / block_size;
	buildFibonnaciSphere<<<grid_size, block_size>>>(p_positions, p_vertexCount);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << "\n";
	}
}