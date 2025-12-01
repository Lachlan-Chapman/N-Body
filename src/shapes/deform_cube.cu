#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

__global__ void deformCube(float *data, float time, int vertexCount) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= vertexCount) {
		return; //dont run thread beyond the vertex count else itll read and write illegal memory
	}

	// if(idx == 0) {
	// 	printf("v0: %f, v1: %f, v2: %f\n", data[0], data[1], data[2]);
	// }
	int posOffset = idx * 6; //each thread gets 6 floats the vec3 pos and vec3 col

	//load vertex data
	float x = data[posOffset + 0];
	float y = data[posOffset + 1];
	float z = data[posOffset + 2];

	float scale = sinf(time * 3.0f) * 1.0f + 0.0f;

	//write back the changes data
	data[posOffset + 0] = x * scale;
	data[posOffset + 1] = y * scale;
	data[posOffset + 2] = z * scale;
}

void runDeformCube(float *data, float time, size_t bufferSize) {
	int totalFloats = bufferSize / sizeof(float);
	int vertices = totalFloats / 6;

	int threads = 64;
	int blocks = (vertices + threads - 1) / threads;

	deformCube<<<blocks, threads>>>(data, time, vertices);
}