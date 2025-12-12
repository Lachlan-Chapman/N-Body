#include <limits>
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

	//init child nodes to make their centers etc

	return old_particle; //return the id of particle we evicted to be reinserted (unoptimal but cpu build is merely a test over optimal build. its concpetually neat)
}

bool octree::insert(universe const &p_universe, int p_particleIndex, int p_rootIndex) {
	int node_index = p_rootIndex; //using custom root allows for fast insertion if we know the parent a particle belongs too

	while(true) {
		octreeNode &node_data = m_nodes[node_index]; //acts as a read only data store of the current node
		//have we found a valid leaf node to insert our particle into
		if(node_data.d_state == nodeState::EMPTY) {
			m_nodes[node_index].d_particleIndex = p_particleIndex; //simply store our particle here
			m_nodes[node_index].d_state = nodeState::LEAF; //mark as now storing a particle
			return true;
		}

		
		if(node_data.d_state == nodeState::LEAF) { //leaf node that isnt holding anything
			//we have found a node that we want to insert into but a particle is already here so we must now convert this into an internal node
			int evicted_particle = leafToInternal(node_index);
			//re insert the particle into this node which would now be internal
			insert(p_universe, evicted_particle, node_index);
			//re insert our particle
			return insert(p_universe, p_particleIndex, node_index); //begins recursion unfortunately
		}

		//parent of other nodes
		if(node_data.d_state == nodeState::INTERNAL) {
			//found an internal node, simply find the quadrant this particle belongs to and set that and loop
			vec3f particle_pos(
				p_universe.m_particles->m_posX[p_particleIndex],
				p_universe.m_particles->m_posY[p_particleIndex],
				p_universe.m_particles->m_posZ[p_particleIndex]
			);
			vec3f direction = particle_pos - node_data.d_center; //go from our center to the position | same vector used when setting the centers
			int child_key =
				(direction.x < 0 ? 0b001 : 0b000) |
				(direction.y < 0 ? 0b010 : 0b000) |
				(direction.z < 0 ? 0b100 : 0b000);
			node_index = m_nodes[node_index].d_childStart + child_key; //increment node
		}
	}
}

void octree::build(universe const &p_universe) {
	//reset m_nodes
	if(m_nodes) { delete[] m_nodes; } //if nodes exist, remove them dont bother trying to overwrite and reset, just let the OS gives us fresh space to use with all default struct values
	m_nodes = new octreeNode[p_universe.m_particles->m_particleCount * 8];
	m_nodeArrayPtr = 0; //reset ptr essentially having no malloced any nodes to our buffer

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
}