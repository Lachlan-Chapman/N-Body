#include <curand_kernel.h>
#include "graphics/Cuda.hpp"
#include "simulation/octree.hpp"

GPUoctree::GPUoctree(universe const &p_universe) :
m_universe(p_universe) {
	unsigned int particle_count = m_universe.m_particles->m_particleCount;
	uint32_t *m_keyBuffer = (uint32_t*)Cuda::malloc(particle_count * sizeof(uint32_t));
	for(int arr_id = 0; arr_id < 4; arr_id++) {
		m_positionCopies[arr_id] = (float*)Cuda::malloc(sizeof(float) * particle_count); //arr of floats for each axis
	}
}

GPUoctree::~GPUoctree() {
	Cuda::free(m_keyBuffer);
	for(int arr_id = 0; arr_id < 4; arr_id++) {
		Cuda::free(m_positionCopies[arr_id]); //arr of floats for each axis
	}
}

__global__ void calculateMinMax(float *p_values, float *p_out, int p_count) {
	int start_index = blockDim.x * blockIdx.x;
	int block_size = min(blockDim.x, p_count - start_index);
	int thread_id = threadIdx.x;

	for (int stride = 1; stride < block_size; stride <<= 1) {

		unsigned int local_base = thread_id * 2 * stride;

		if (local_base + stride < block_size) {
			int lhs = start_index + local_base;
			int rhs = lhs + stride;

			float min_val = fminf(p_values[lhs], p_values[rhs]);
			float max_val = fmaxf(p_values[lhs], p_values[rhs]);

			p_values[lhs] = min_val;
			p_values[rhs] = max_val;
		}

		__syncthreads(); //MUST be unconditional
	}

	//write packed min/max for this block
	int out_index = blockIdx.x * 2;
	p_out[out_index] = p_values[start_index];

	if (block_size > 1) {
		p_out[out_index + 1] = p_values[start_index + block_size - 1];
	}
}

void GPUoctree::runTest() {
	int count = 8;
	float *gpu_arr = (float*)Cuda::unifiedMalloc(sizeof(float) * count);
	float *gpu_out = (float*)Cuda::unifiedMalloc(sizeof(float) * count);
	for (int i = 0; i < count; i++) { gpu_arr[i] = -20; }
	gpu_arr[2] = 935;
	gpu_arr[4] = -115;
	int block_count = 1;
	calculateMinMax<<<block_count, count>>>(
		gpu_arr,
		gpu_out,
		count
	);
	cudaDeviceSynchronize();
}

void GPUoctree::AABB() {
	int particle_count = m_universe.m_particles->m_particleCount;
	size_t arr_byte_size = sizeof(float) * particle_count;
	Cuda::memcpy(
		m_positionCopies[0],
		m_universe.m_particles->m_posX,
		arr_byte_size
	);
	Cuda::memcpy(
		m_positionCopies[1],
		m_universe.m_particles->m_posY,
		arr_byte_size
	);
	Cuda::memcpy(
		m_positionCopies[2],
		m_universe.m_particles->m_posZ,
		arr_byte_size
	);

	int thread_count = 256;
	vec3f min, max;
	float *in, *out;
	for(int dimension = 0; dimension < 3; dimension++) {
		float* temp = in;
		in = m_positionCopies[dimension];
		out = m_positionCopies[3];
		int arr_size = particle_count;
		while(arr_size > 2) {
			int block_count = (arr_size + (thread_count-1)) / thread_count;
			calculateMinMax<<<block_count, thread_count>>>(in, out, particle_count);
			cudaDeviceSynchronize();
			in = out;
			out = temp;
			arr_size = block_count * 2; //each block produces a minmax pair
		}
		min[dimension] = in[0];
		max[dimension] = in[1];
	}
	std::cout << min << "\n" << max << "\n";
}

__global__ void createMortonKeys(
	particles *p_particles,
	vec3f p_minPosition,
	vec3f p_maxPosition,
	unsigned int p_axisResolution,
	uint32_t *p_keyBuffer
) {
	int pid = blockDim.x * blockIdx.x + threadIdx.x; //how wide is the block * how many blocks over am i + the thread i am in the block
	if(pid >= p_particles->m_particleCount) { return; }
	//scale positions
	vec3f universe_extent = p_maxPosition - p_minPosition;
	uint32_t scale = 1u << p_axisResolution; //10 bit length produces 1024 say
	float x_float = (p_particles->m_posX[pid] - p_minPosition.x) / universe_extent.x; //we now have the x pos as a % of how far across the universe on the x axis so 0.5 would mean it sits in teh middle of the universe on the x axis
	x_float = fminf(fmaxf(x_float, 0.0f), 1.0f); //first ensure its >= 0 then make sure its <= 1
	uint32_t x_integer = min(
		static_cast<uint32_t>(
			x_float * scale
		),
		scale - 1
	); //ensure floating point inprecision hasnt allowed us to exceed the scale. if the x_float was 1.0001 then x_i would be > 2^bitLength being in the bitLength+1th bit

	float y_float = (p_particles->m_posY[pid] - p_minPosition.y) / universe_extent.y;
	y_float = fminf(fmaxf(y_float, 0.0f), 1.0f);
	uint32_t y_integer = min(
		static_cast<uint32_t>(
			y_float * scale
		),
		scale - 1
	);

	float z_float = (p_particles->m_posZ[pid] - p_minPosition.z) / universe_extent.z;
	z_float = fminf(fmaxf(z_float, 0.0f), 1.0f);
	uint32_t z_integer = min(
		static_cast<uint32_t>(
			z_float * scale
		),
		scale - 1
	);

	//fragment xyz positions to go from largest value to smallest in triplets to build our octree traversal structure
	uint32_t morton = 0; //fresh block of 32 0s
	for(int bit_id = p_axisResolution-1; bit_id >= 0; --bit_id) {
		//get the bit_idth bit
		morton <<= 3; //the msb of the triplet need to shift up so we can add the now LSB triplet
		morton |= ((x_integer >> bit_id) & 1u) << 2;
		morton |= ((y_integer >> bit_id) & 1u) << 1;
		morton |= ((z_integer >> bit_id) & 1u);
	}
	p_keyBuffer[pid] = morton;
}

void GPUoctree::build(unsigned int p_axisBitCount) {

	//uint32_t for each particle

	unsigned int thread_count = 256;
	unsigned int block_count = (m_universe.m_particles->m_particleCount + thread_count - 1) / thread_count; //giving it will be an integer divide, how many lots of threads fits into the particle count
	
	AABB();
}