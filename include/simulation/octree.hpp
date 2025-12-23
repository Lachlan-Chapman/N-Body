#pragma once
#include "math/vec.hpp"
#include "simulation/universe.hpp"

enum class nodeState : int {
	UNINITIALIZED = 0, //is not yet been malloced
	EMPTY = 1, //malloced by not used
	LEAF = 2, //a bucket that stores up to 8 particles before becoming an interval
	BEDROCK = 3, //a bucket that stores N particles, allocating space as needed | safe guard against insertion of > 8 "identical" (consider float inaccuracy) points
	INTERNAL = 4
};


struct octreeNode {
	//set for each node on creation
	nodeState d_state = nodeState::UNINITIALIZED;
	int d_childStart = -1;
	int d_childCount = -1; //index into global node arr where children begin and total children
	
	int d_particleIndex; //index into global particle arr which

	int d_particleBucketSize; //total particles allowed before a remalloc is needed
	int d_particleCount; //number of particles stored in this node
	int *d_particleBucket = nullptr; //bucket of variable size for bedrock nodes which remain bedrock permanetly

	//root has this set, but all other nodes have this built after the tree is made
	vec3f d_center, d_dimensions, d_halfDimensions; //need half widths for easy bound calcs later

	//calculated after tree construction
	vec3f d_massCenter;
	float d_mass; //total combined mass of the node and half the node width IE "radius"

};

class octree {
public:
	void build(universe const &p_universe);
	int m_maxDepth;
	octreeNode *m_nodes = nullptr; //contigous 1d arr to store octree nodes
private:	
	void reset(universe const &p_universe);
	void createChildNodes(int p_childStart, vec3f p_parentCenter, vec3f p_parentHalfWidth); //allocates space in the m_nodes arr | sets position, width data
	int leafToInternal(int p_nodeTarget); //takes a leaf, allocated 8 children, sets pos data and dimensions for each child, returns the particle which was in that leaf
	void leafToBedrock(int p_nodeTarget);
	bool insert(universe const &p_universe, int p_particleIndex, int p_rootIndex = 0, int p_depth = 0); //default starts from root, traverses top down to find an empty leaf to store particle data | will expand tree as needed
	void compute(universe const &p_universe);
	int m_nodeArrayPtr; //ptr to the next free location in the global node arr
};

class GPUoctree {
public:
	GPUoctree() = delete;
	GPUoctree(universe const &p_universe);
	~GPUoctree();
	void runTest();
	void build(unsigned int p_axisBitCount);
	void AABB();
private:
	uint32_t *m_keyBuffer = nullptr;
	float *m_positionCopies[4];
	universe const &m_universe;
};