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

	p_particles->m_posX[idx] = dir.x * mag;
	p_particles->m_posY[idx] = dir.y * mag;
	p_particles->m_posZ[idx] = dir.z * mag;

	p_particles->m_velX[idx] = 0;
	p_particles->m_velY[idx] = 0;
	p_particles->m_velZ[idx] = 0;

	p_particles->m_mass[idx] = 0.01;
}

universe::universe(size_t p_particleCount, unsigned int p_frequency, float p_radius) { //struct of arrays init
	m_frequency = p_frequency;
	m_particles = new particles;
	m_particles->m_particleCount = p_particleCount;
	m_particles->m_accX = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_accY = (float*)Cuda::malloc(p_particleCount * sizeof(float)); 
	m_particles->m_accZ = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_velX = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_velY = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_velZ = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_posX = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_posY = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_posZ = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	m_particles->m_mass = (float*)Cuda::malloc(p_particleCount * sizeof(float));
	int thread_count = 256;
	int block_count = (m_particles->m_particleCount + thread_count - 1) / thread_count; //ensure more than enough blocks of 256 are dispatched
	initUniverse<<<block_count, thread_count>>>(m_particles, p_radius);
}



__constant__ __device__ float epsilon_squared = 0.05;
//__constant__ __device__ float G = 6.674e-11f;
__constant__ __device__ float G = 0.1; //1 for speed
__global__ void calculateForce(particles *p_particles) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //particle ID
	unsigned int particle_count = p_particles->m_particleCount;
	if(idx >= p_particles->m_particleCount) { return; }
	
	vec3f my_pos(
		p_particles->m_posX[idx],
		p_particles->m_posY[idx],
		p_particles->m_posZ[idx]
	);

	vec3f accel(0.0f); //accel is re set each frame its the bedrock layer of our euler intergration


	for(unsigned int pid = 0; pid < particle_count; pid++) {

		if(pid == idx) { continue; } //avoid self interaction
		
		//get distance
		vec3f other_pos(
			p_particles->m_posX[pid],
			p_particles->m_posY[pid],
			p_particles->m_posZ[pid]
		);
		
		vec3f dir = other_pos - my_pos;
		float dist_square = dir.squared_magnitude(); //gpu has fast inverse (need to benchmark again fast_inverse from the vec class)
		dist_square += epsilon_squared; //softening the distance to stop 0 distance calculations
		
		float inv_dist = rsqrtf(dist_square);
		float inv_dist_cube = inv_dist * inv_dist * inv_dist;
		
		//calculate accel mag, appliy to direction, append to this particle accel
		float other_mass = p_particles->m_mass[pid];
		accel += dir * (G * other_mass * inv_dist_cube); //force applied to me only depends on the mass of the other object since solving for acceleration based on the f = ma you dividide by your own mass
		
		
	}
	p_particles->m_accX[idx] = accel[0];
	p_particles->m_accY[idx] = accel[1];
	p_particles->m_accZ[idx] = accel[2];
}

void universe::calculateAcceleration() {
	int thread_count = 256;
	int block_count = (m_particles->m_particleCount + thread_count - 1) / thread_count; //ensure more than enough blocks of 256 are dispatched
	calculateForce<<<block_count, thread_count>>>(m_particles);
}

__global__ void integrateAcceleration(particles *p_particles, float p_dt) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //particle ID
	unsigned int particle_count = p_particles->m_particleCount;
	if(idx >= p_particles->m_particleCount) { return; }

	p_particles->m_velX[idx] += p_particles->m_accX[idx] * p_dt;
	p_particles->m_velY[idx] += p_particles->m_accY[idx] * p_dt;
	p_particles->m_velZ[idx] += p_particles->m_accZ[idx] * p_dt;

	p_particles->m_posX[idx] += p_particles->m_velX[idx] * p_dt;
	p_particles->m_posY[idx] += p_particles->m_velY[idx] * p_dt;
	p_particles->m_posZ[idx] += p_particles->m_velZ[idx] * p_dt;
}

void universe::integrate() {
	int thread_count = 256;
	int block_count = (m_particles->m_particleCount + thread_count - 1) / thread_count; //ensure more than enough blocks of 256 are dispatched
	integrateAcceleration<<<block_count, thread_count>>>(m_particles, 1.0f / m_frequency);
}

void universe::step() {
	calculateAcceleration();
	integrate();
}
