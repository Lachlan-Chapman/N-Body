#include <curand_kernel.h>

#include "math/vec.hpp"
#include "graphics/Cuda.hpp"
#include "simulation/particle.hpp"
#include "simulation/universe.hpp"
#include "simulation/octree.hpp"

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


//big enough for about > 9.8 billion points universe using 12 depth and 8 nodes
#define MAX_STACK 92
//distance threshold whether to approximate or not
#define THETA 10

__global__ void stepSimulation(octree *p_octree, particles *p_particles, vec3f *p_positionVBO, float p_dt, int p_stepCount) {
	int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if(pid >= p_particles->m_particleCount) { return; }
	
	int stack[MAX_STACK]; 
	float particle_mass = p_particles->m_mass[pid];
	
	for(int step_id = 0; step_id < p_stepCount; step_id++) {
		int stack_ptr = 0;
		stack[stack_ptr++] = 0;
	
		vec3f particle_position(
			p_particles->m_posX[pid],
			p_particles->m_posY[pid],
			p_particles->m_posZ[pid]
		);

		vec3f acceleration(0.0f);
	
		while(stack_ptr) { //we got nodes to search
			int node_id = stack[--stack_ptr];
			octreeNode &node = p_octree->m_nodes[node_id];
			if(node.d_state == nodeState::EMPTY) { continue; }
			if(node.d_state == nodeState::LEAF && node.d_particleIndex == pid) { continue; } //self intersection guard
			
			float current_mass = node.d_mass; //only changes if bedrock is self containing
			vec3f current_com = node.d_massCenter; //INTERNAL, LEAF, BEDROCK (unaltered) will use these values
			
			if(node.d_state == nodeState::BEDROCK) { //adjust only if self containing
				for(int child_id = 0; child_id < node.d_particleCount; child_id++) { //check if i am a child of that node
					if(node.d_particleBucket[child_id] == pid) {
						current_mass -= particle_mass;
						if(current_mass <= 0) { continue; }
						current_com = ((node.d_massCenter * node.d_mass) - (particle_position * particle_mass)) / current_mass;
						break;
					}
				}
			}

			vec3f direction = current_com - particle_position; //common for all types, needed before internal check to see if we approximate or add to stack
			float dist_squared = direction.sqauredMagnitude();

			if(node.d_state == nodeState::INTERNAL) {
				float node_size = node.d_dimensions.max(); //largest length of the AABB
				if(dist_squared < (THETA*THETA * node_size*node_size)) { //checking the regular d / s < theta | theta is the limit on how many node_sizes away we are. if we are too close, then push to the stack
					for(int child_id = 0; child_id < 8; child_id++) { //check if i am a child of that node
						if(stack_ptr < MAX_STACK) {
							stack[stack_ptr++] = node.d_childStart + child_id; //add node ids to now re check
						}
					}
					continue;
				}
			}

			//get accleartion, LEAF & INTERNAL nodes have values directly set, BEDROCK nodes if self containing will adjust the set values
			float inv_dist = rsqrt(dist_squared + epsilon_squared);
			float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
			acceleration += direction * (G * current_mass * inv_dist_cubed);	
		}
		p_particles->m_accX[pid] = acceleration[0];
		p_particles->m_accY[pid] = acceleration[1];
		p_particles->m_accZ[pid] = acceleration[2];
		
		//calculate velocity
		p_particles->m_velX[pid] += p_particles->m_accX[pid] * p_dt;
		p_particles->m_velY[pid] += p_particles->m_accY[pid] * p_dt;
		p_particles->m_velZ[pid] += p_particles->m_accZ[pid] * p_dt;

		//calculate position
		p_particles->m_posX[pid] += p_particles->m_velX[pid] * p_dt;
		p_particles->m_posY[pid] += p_particles->m_velY[pid] * p_dt;
		p_particles->m_posZ[pid] += p_particles->m_velZ[pid] * p_dt;
	}

	//copy pos to vbo
	p_positionVBO[pid] = vec3f(
		p_particles->m_posX[pid],
		p_particles->m_posY[pid],
		p_particles->m_posZ[pid]
	);

}

void universe::step(octree *p_octree, vec3f *p_positionVBO, int p_stepCount) {
	
	//using combined calculation kernel
	int thread_count = 256;
	int block_count = (m_particles->m_particleCount + thread_count - 1) / thread_count;
	stepSimulation<<<block_count, thread_count>>>(
		p_octree,
		m_particles,
		p_positionVBO,
		1.0f / m_frequency,
		p_stepCount
	);
}
