#pragma once
#include <cmath>
#include "math/vec.hpp"
#include "shapes/primitive.hpp"


class triangle {
public:
	static unsigned int calculateVertexCount(unsigned int p_subdivisonCount);
	static unsigned int calculateEdgeCount(unsigned int p_subdivisonCount);
	static unsigned int calculateFaceCount(unsigned int p_subdivisonCount);
	
	triangle();

	triangle(
		vec3f p_alpha,
		vec3f p_beta,
		vec3f p_epsilon
	);

	triangle(
		vec3f *p_vertices,
		unsigned int p_vertexCount
	);

	void subdivide(
		unsigned int p_subdivisionCount,
		vec3f *&p_newVertices,
		unsigned int &p_vertexCount
	);
private:
	vec3f m_alpha, m_beta, m_epsilon; //used for basic primitive triangle
	
	vec3f *m_vertices; //used when storing a "complex" triangle
	unsigned int m_vertexCount;
};



class octahedron {
public:
	octahedron();
	octahedron(unsigned int p_subdivisonCount);

private:

	vec3f* mallocNewVertices();
	void subdivideEdges();

	mesh m_baseMesh;
	mesh m_mesh;


	unsigned int m_subdivisionCount;
};