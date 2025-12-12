#include <limits>
#include <cstring> //memcpy
#include <cmath>

#include "graphics/Cuda.hpp"
#include "simulation/universe.hpp"
#include "simulation/octree.hpp"

//assumes parents has malloc space in our buffer for children
void octree::createChildNodes(int p_childStart, vec3f p_parentCenter, vec3f p_parentHalfDimension) {
	vec3f parent_quarter_dimension = p_parentHalfDimension * 0.5;
	for(int child_id = 0; child_id < 8; child_id++) {
		octreeNode &child = m_nodes[p_childStart + child_id];

		child.d_state = nodeState::EMPTY;

		child.d_dimensions = p_parentHalfDimension;
		child.d_halfDimensions = parent_quarter_dimension;

		child.d_center = p_parentCenter + 
		vec3f(
			parent_quarter_dimension.x * ((child_id & 0b001) ? -1 : 1), //if the bit is non zero then we say its a negative dir. ie child_centers 0 (0b000) = (+x, +y, +z) and child_centers 3 (0b010) = (+x, -y, +z)
			parent_quarter_dimension.y * ((child_id & 0b010) ? -1 : 1),
			parent_quarter_dimension.z * ((child_id & 0b100) ? -1 : 1)
		);
	}
}

int octree::leafToInternal(int p_nodeTarget) {
	//get and check node
	octreeNode &node = m_nodes[p_nodeTarget];
	if(node.d_state != nodeState::LEAF) { return -1; } //double check it is a proper leaf node
	node.d_state = nodeState::INTERNAL;
	
	//init children
	node.d_childStart = m_nodeArrayPtr; //assign the current free memory as the start
	createChildNodes(node.d_childStart, node.d_center, node.d_halfDimensions);
	node.d_childCount = 8; //now has 8 children a brand new internal node | childStart is already set when node was built so that remains
	m_nodeArrayPtr += node.d_childCount; //we have kind of "malloc" into our own block so we progress the ptr to the next free location

	//evict current particle
	int old_particle = node.d_particleIndex;
	node.d_particleIndex = -1; //not needed since it would never be read since its state is internal BUT if it is, should cause corrupt memory and crash which would tell me a bug exists

	return old_particle; //return the id of particle we evicted to be reinserted (unoptimal but cpu build is merely a test over optimal build. its concpetually neat)
}

void octree::leafToBedrock(int p_nodeTarget) {
	octreeNode &node = m_nodes[p_nodeTarget];
	if(node.d_state != nodeState::LEAF) { return; }
	node.d_state = nodeState::BEDROCK;

	node.d_particleBucketSize = 8; //init size being 8 | double each expansion
	node.d_particleCount = 0;
	node.d_particleBucket = (int*)Cuda::unifiedMalloc(sizeof(int) * node.d_particleBucketSize);
	node.d_particleBucket[node.d_particleCount++] = node.d_particleIndex; //move the singular particle into the new bucket
	node.d_particleIndex = -1;
}

bool octree::insert(universe const &p_universe, int p_particleIndex, int p_rootIndex, int p_depth) {
	int node_index = p_rootIndex; //using custom root allows for fast insertion if we know the parent a particle belongs too
	int current_depth = p_depth;
	while(true) {
		octreeNode &node_data = m_nodes[node_index]; //acts as a read only data store of the current node
		//have we found a valid leaf node to insert our particle into
		bool success = true;
		switch(node_data.d_state) {
			case nodeState::EMPTY:
				m_nodes[node_index].d_particleIndex = p_particleIndex; //simply store our particle here
				m_nodes[node_index].d_state = nodeState::LEAF; //mark as now storing a particle
				return true;
				break;
			case nodeState::LEAF: //we have found a node that we want to insert into but a particle is already here so we must now convert this into an internal node
				if(current_depth < m_maxDepth) { //can safely make an internal node
					int evicted_particle = leafToInternal(node_index);
					insert(p_universe, evicted_particle, node_index, current_depth); //fast insert right into the new node
				} else { //cant safely make internal node so we must make a bedrock node
					leafToBedrock(node_index); //makes node[node_index]  a bedrock bucket
				}
				success = success && insert(p_universe, p_particleIndex, node_index, current_depth); //re insert out new node fast to either new interval or new bedrock node
				return success;
				break;
			case nodeState::BEDROCK: //handle bucket insertions and occasionaly expansion of the bucket
				if(node_data.d_particleCount >= node_data.d_particleBucketSize) { //then our bucket is full and we must expand
					int *old_bucket = node_data.d_particleBucket;
					node_data.d_particleBucketSize *= 2; //double space
					node_data.d_particleBucket = (int*)Cuda::unifiedMalloc(sizeof(int) * node_data.d_particleBucketSize);

					//copy old bucket over
					std::memcpy(
						node_data.d_particleBucket,
						old_bucket,
						sizeof(int) * node_data.d_particleCount
					);
					Cuda::free(old_bucket);
				}
				node_data.d_particleBucket[node_data.d_particleCount++] = p_particleIndex; //stored particle into bucket
				return true;
				break;
			case nodeState::INTERNAL:
				vec3f particle_pos(
					p_universe.m_particles->m_posX[p_particleIndex],
					p_universe.m_particles->m_posY[p_particleIndex],
					p_universe.m_particles->m_posZ[p_particleIndex]
				);
				vec3f direction = particle_pos - node_data.d_center; //go from our center to the position | same vector used when setting the centers
				int child_key =
					((direction.x < 0) << 0) |
					((direction.y < 0) << 1) |
					((direction.z < 0) << 2);
				node_index = m_nodes[node_index].d_childStart + child_key; //increment node
				current_depth++; //we are now one layer deeper
				break;
		}
	}
}

//the beauty of the build order AND using a 1d array is that for any node i, its certain that its children live > i in the buffer
//this way if we simply loop backward from the ptr and computer the masses
//if leaf, assign particle mass to node mass and assign center of mass to particle mass
//if internal loop children and weighted average driven by mass of children center of mass
//we know that for any node its children have been computed since for node i, its children are > i
void octree::compute(universe const &p_universe) {
	//runs [0, m_nodeArrayPtr)
	int node_ptr = m_nodeArrayPtr;
	while(node_ptr) {
		octreeNode &node = m_nodes[--node_ptr];
		float mass = 0.0f;
		vec3f com(0.0f);
		switch(node.d_state) {
			case nodeState::LEAF:
				//node essentially is particle
				node.d_massCenter = vec3f(
					p_universe.m_particles->m_posX[node.d_particleIndex],
					p_universe.m_particles->m_posY[node.d_particleIndex],
					p_universe.m_particles->m_posZ[node.d_particleIndex]
				);
				node.d_mass = p_universe.m_particles->m_mass[node.d_particleIndex];
				break;
			case nodeState::BEDROCK: //sum bucket
				for(int child_id = 0; child_id < node.d_particleCount; child_id++) {
					int particle_id = node.d_particleBucket[child_id];
					mass += p_universe.m_particles->m_mass[particle_id];
					com +=
						p_universe.m_particles->m_mass[particle_id] *
						vec3f(
							p_universe.m_particles->m_posX[particle_id],
							p_universe.m_particles->m_posY[particle_id],
							p_universe.m_particles->m_posZ[particle_id]
						);
				}
				com /= mass;
				node.d_mass = mass;
				node.d_massCenter = com;
				break;
			case nodeState::INTERNAL:
				for(int child_id = 0; child_id < 8; child_id++) {
					octreeNode &child_node = m_nodes[node.d_childStart + child_id];
					mass += child_node.d_mass;
					com += child_node.d_mass * child_node.d_massCenter;
				}
				com /= mass;
				node.d_mass = mass;
				node.d_massCenter = com;
				break;
		}
	}
	//std::cout << "Universe Mass " << m_nodes[0].d_mass << "\n";
	
}

void octree::reset(universe const &p_universe) {
	if(m_nodes) {
		int node_ptr = m_nodeArrayPtr;
		while(node_ptr > 0) { //delete all bedrock bucket allocations
			--node_ptr;
			if(m_nodes[node_ptr].d_particleBucket) {
				Cuda::free(m_nodes[node_ptr].d_particleBucket);
			}
		}
		Cuda::free(m_nodes); //delete all nodes
	}

	//(8^(x-1) - 1) / 7 = node_count
	//we need node_count > particle_count
	//(8^(x-1) - 1) / 7 = particle_count
	//ceil((log(6.66 * particle_count + 1) * 0.48)) + 1

	m_maxDepth = static_cast<unsigned int>(
		std::ceil(
			std::log((6.66 * p_universe.m_particles->m_particleCount) + 1) *
			0.4808 //.4808 = 1/ln(8)
		) + 2
	); 
	unsigned int full_tree_count = (std::pow(8, m_maxDepth - 1) - 1.0) * 0.15; //~= (8^(maxDepth - 1) - 1) / 7
	//std::cout << "allowing " << full_tree_count << " Nodes\n";
	m_nodes = (octreeNode*)Cuda::unifiedMalloc(sizeof(octreeNode) * full_tree_count);
	m_nodeArrayPtr = 0; //reset ptr
}

void octree::build(universe const &p_universe) {
	//reset m_nodes
	reset(p_universe);

	//build root node
	vec3f min_coords(std::numeric_limits<float>::infinity());
	vec3f max_coords(-std::numeric_limits<float>::infinity());
	for(int particle_id = 0; particle_id < p_universe.m_particles->m_particleCount; particle_id++) { //create aabb
		if(p_universe.m_particles->m_posX[particle_id] < min_coords.x) { min_coords.x = p_universe.m_particles->m_posX[particle_id]; }
		if(p_universe.m_particles->m_posY[particle_id] < min_coords.y) { min_coords.y = p_universe.m_particles->m_posY[particle_id]; }
		if(p_universe.m_particles->m_posZ[particle_id] < min_coords.z) { min_coords.z = p_universe.m_particles->m_posZ[particle_id]; }

		if(p_universe.m_particles->m_posX[particle_id] > max_coords.x) { max_coords.x = p_universe.m_particles->m_posX[particle_id]; }
		if(p_universe.m_particles->m_posY[particle_id] > max_coords.y) { max_coords.y = p_universe.m_particles->m_posY[particle_id]; }
		if(p_universe.m_particles->m_posZ[particle_id] > max_coords.z) { max_coords.z = p_universe.m_particles->m_posZ[particle_id]; }
	}
	octreeNode &root = m_nodes[m_nodeArrayPtr++];
	root.d_state = nodeState::EMPTY;
	root.d_dimensions = max_coords - min_coords;
	root.d_halfDimensions = 0.5 * root.d_dimensions;
	root.d_center = min_coords + (root.d_halfDimensions);



	//insert all particles
	for(int particle_id = 0; particle_id < p_universe.m_particles->m_particleCount; particle_id++) {
		insert(p_universe, particle_id); //insert from the regular root
	}

	compute(p_universe);
}