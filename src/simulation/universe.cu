#include <curand_kernel.h>

#include "math/vec.hpp"
#include "graphics/Cuda.hpp"
#include "simulation/particle.hpp"
#include "simulation/universe.hpp"

__global__ void initUniverse(particles *p_particles, float p_radius) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= p_particles->m_particleCount) return;

	curandState rng;
	curand_init(935, idx, 0, &rng);

	float u = curand_uniform(&rng) * 2.0f - 1.0f;
	float phi = curand_uniform(&rng) * 6.283185f;
	float sqrt_term = sqrtf(1.0f - (u * u));

	vec3f dir(
		sqrt_term * cosf(phi),
		sqrt_term * sinf(phi),
		u
	);

	float mag = cbrtf(curand_uniform(&rng) * p_radius);

	p_particles->m_accX[idx] = 0;
	p_particles->m_accY[idx] = 0;
	p_particles->m_accZ[idx] = 0;

	p_particles->m_velX[idx] = 0;
	p_particles->m_velY[idx] = 0;
	p_particles->m_velZ[idx] = 0;

	p_particles->m_posX[idx] = dir.x * mag;
	p_particles->m_posY[idx] = dir.y * mag;
	p_particles->m_posZ[idx] = dir.z * mag;

	p_particles->m_mass[idx] = 0.01;
}



universe::universe(size_t p_particleCount, unsigned int p_frequency, float p_radius) { //struct of arrays init
	m_frequency = p_frequency;
	m_particles = new particles;
	m_particles->m_particleCount = p_particleCount;

	m_particles->m_accX = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_accY = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float)); 
	m_particles->m_accZ = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_velX = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_velY = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_velZ = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_posX = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_posY = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_posZ = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));
	m_particles->m_mass = (float*)Cuda::unifiedMalloc(p_particleCount * sizeof(float));

	int thread_count = 256;
	int block_count = (m_particles->m_particleCount + thread_count - 1) / thread_count; //ensure more than enough blocks of 256 are dispatched
	initUniverse<<<block_count, thread_count>>>(m_particles, p_radius); //dispatch kernel
	cudaDeviceSynchronize(); //wait for universe to init
}

__constant__ __device__ float epsilon_squared = 0.05;
//__constant__ __device__ float G = 6.674e-11f;
__constant__ __device__ float G = 0.1; //1 for speed
//note p_particles is soa
__global__ void stepSimulation(particles *p_particles, vec3f *p_positionVBO, float p_dt, int p_stepCount) { //combined kernel that does accel, force, pos, copy data to pen gl buff
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= p_particles->m_particleCount) { return; }

	for(int step_id = 0; step_id < p_stepCount; step_id++) {
		vec3f my_pos(
			p_particles->m_posX[idx],
			p_particles->m_posY[idx],
			p_particles->m_posZ[idx]
		);


		//calculate acceleration
		vec3f acceleration(0.0f);
		for(int particle_id = 0; particle_id < p_particles->m_particleCount; particle_id++) {
			if(idx == particle_id) { continue; }
			//force due to gravity
			vec3f other_pos(
				p_particles->m_posX[particle_id],
				p_particles->m_posY[particle_id],
				p_particles->m_posZ[particle_id]
			);

			vec3f direction = other_pos - my_pos;
			float dist_sqaured = direction.sqauredMagnitude();
			float inv_dist = rsqrt(dist_sqaured);
			inv_dist += epsilon_squared;
			float inv_dist_cubed = inv_dist * inv_dist * inv_dist;

			float other_mass = p_particles->m_mass[particle_id];
			acceleration += direction * (G * other_mass * inv_dist_cubed);
		}
		
		p_particles->m_accX[idx] = acceleration[0];
		p_particles->m_accY[idx] = acceleration[1];
		p_particles->m_accZ[idx] = acceleration[2];
		
		//calculate velocity
		p_particles->m_velX[idx] += p_particles->m_accX[idx] * p_dt;
		p_particles->m_velY[idx] += p_particles->m_accY[idx] * p_dt;
		p_particles->m_velZ[idx] += p_particles->m_accZ[idx] * p_dt;

		//calculate position
		p_particles->m_posX[idx] += p_particles->m_velX[idx] * p_dt;
		p_particles->m_posY[idx] += p_particles->m_velY[idx] * p_dt;
		p_particles->m_posZ[idx] += p_particles->m_velZ[idx] * p_dt;
	}
	//copy final pos to opengl buffer
	p_positionVBO[idx] = vec3f(
		p_particles->m_posX[idx],
		p_particles->m_posY[idx],
		p_particles->m_posZ[idx]
	);

}


void universe::step(vec3f *p_positionVBO, int p_stepCount) {
	//using combined calculation kernel
	int thread_count = 256;
	int block_count = (m_particles->m_particleCount + thread_count - 1) / thread_count;
	stepSimulation<<<block_count, thread_count>>>(
		m_particles,
		p_positionVBO,
		1.0f / m_frequency,
		p_stepCount
	);
}
