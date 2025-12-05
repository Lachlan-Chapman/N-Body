#include "shapes/octahedron.hpp"

inline unsigned int triangle::calculateVertexCount(unsigned int p_subdivisonCount) { //v(sd) = (sd^2 + 5sd + 6) / 2
	return ((p_subdivisonCount * p_subdivisonCount) + (5 * p_subdivisonCount) + (6)) / 2;
}

inline unsigned int triangle::calculateEdgeCount(unsigned int p_subdivisonCount) {
	return p_subdivisonCount;
}

inline unsigned int triangle::calculateFaceCount(unsigned int p_subdivisonCount) { //f(sd) = (sd+1)^2 
	unsigned int face_count = p_subdivisonCount + 1;
	return face_count * face_count;
}

triangle::triangle() : 
	triangle(
		Primitives::triangle_mesh.d_vertices[0],
		Primitives::triangle_mesh.d_vertices[1],
		Primitives::triangle_mesh.d_vertices[2]
	)
{}

triangle::triangle(
	vec3f p_alpha,
	vec3f p_beta,
	vec3f p_epsilon
) :
	m_alpha(p_alpha),
	m_beta(p_beta),
	m_epsilon(p_epsilon)
{}

triangle::triangle(
	vec3f *p_vertices,
	unsigned int p_vertexCount
) :
	m_vertices(p_vertices),
	m_vertexCount(p_vertexCount)
{}


/* 
memory layout for easy refacing for 2 subdivisionCount
	| m_alpha, mid_point, mid_point, mid_point, center_point, mid_point, m_beta, mid_point, mid_point, m_epsilon|
which can be viewed as such
m_alpha
mid_point,	mid_point
mid_point,	center_point,	mid_point,
m_beta,		mid_point,		mid_point,	m_epsilon

*/
void triangle::subdivide(
	unsigned int p_subdivisionCount,
	vec3f *&p_newVertices,
	unsigned int &p_vertexCount
) {
	p_vertexCount = calculateVertexCount(p_subdivisionCount);
	p_newVertices = new vec3f[p_vertexCount];

	float denominator = 1.0f / (1.0f + p_subdivisionCount); //to get equal step size based on how many new mid points are there
	vec3f down = (m_epsilon - m_alpha).unit() * denominator;
	vec3f right = (m_beta - m_epsilon).unit() * denominator;
	
	vec3f current_position = m_alpha - down; //start off the triangle so our first step puts us on m_alpha
	unsigned int current_vertex = 0;
	
	unsigned int layer_count = 2 + p_subdivisionCount; //total vertical layers
	for(unsigned int layer_id = 0; layer_id < layer_count; layer_id++) {
		vec3f edge_position = m_alpha + (layer_id * down);
		unsigned int layer_vertex_count = layer_id + 1;

		for(unsigned int vertex_id = 0; vertex_id < layer_vertex_count; vertex_id++) {
			vec3f mid_position = edge_position + (vertex_id * right); //only ever move across to the right
			p_newVertices[current_vertex++] = mid_position;
			std::cout << "Vertex " << vertex_id << " " << mid_position << "\n";
		}
	}	
}

//read left to right. so ya gonna go 1, 2, 3, 4 mid points per layer in arithemtic sum
//if it stime to start new layer then go down to the left from row start else go across by mid point distance