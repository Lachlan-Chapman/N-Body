#include "shapes/octahedron.hpp"

unsigned int triangle::calculateVertexCount(unsigned int p_subdivisonCount) { //v(sd) = (sd^2 + 5sd + 6) / 2
	//return ((p_subdivisonCount * p_subdivisonCount) + (5 * p_subdivisonCount) + (6)) / 2;
	return static_cast<unsigned int>(((p_subdivisonCount * p_subdivisonCount) + (5 * p_subdivisonCount) + (6)) * 0.5f); //should floor it
}

unsigned int triangle::calculateEdgeCount(unsigned int p_subdivisonCount) {
	return p_subdivisonCount;
}

unsigned int triangle::calculateFaceCount(unsigned int p_subdivisonCount) { //f(sd) = (sd+1)^2 
	unsigned int face_count = p_subdivisonCount + 1;
	return face_count * face_count;
}

unsigned int triangle::calculateLayerCount(unsigned int p_subdivisionCount) {
	return p_subdivisionCount + 1; //top and bottom vertex + midpoints from subdividing but minus the top vertex since layers begin from the 1th vertex
}

void triangle::faceTriangle(vec<3, unsigned int> *&p_faceIndicies, vec3f *p_vertices, unsigned int p_subdivisonCount) {

	unsigned int layer_count = calculateLayerCount(p_subdivisonCount);
	unsigned int down_count = 0; //track total down faces made
	
	

	vec<3, unsigned int> down_layer_end(down_count++, 2, 1); //init as the first triangle (identical for all tris) //the final triangle from the layer above, always a down triangle
	vec<3, unsigned int> across_layer_end(1, 2, 4);
	unsigned int face_counter = 0;
	p_faceIndicies[face_counter++] = down_layer_end; //0th triangle is hard coded
	p_faceIndicies[face_counter++] = across_layer_end; //0th across triangle is hard coded
	for(int layer_id = 1; layer_id < layer_count; layer_id++) { //0th layer is hard coded
		vec<3, unsigned int> face_order(0);
		unsigned int total_downs = (layer_id + 1) % (layer_count + 1);
		for(unsigned int down_id = 0; down_id < total_downs; down_id++) {
			face_order[0] = down_count++; //alpha
			face_order[1] = down_layer_end[1] + 2 + down_id; //beta
			face_order[2] = face_order[1] - 1; //epsilon
			p_faceIndicies[face_counter++] = face_order;
			//std::cout << "Down Face Order ID " << layer_id << ", " << down_id << " " << face_order << "\n";
		}
		down_layer_end = face_order; //by here we are at the final tri of this layer
		//std::cout << "stored prev layer order ID " << down_layer_end << "\n";
		
		unsigned int total_across = layer_id;
		unsigned int across_count = 0;
		if(layer_id == 1) { continue; } //the 0th layer has no across but we dont do layer 0, layer 1 is hard coded to skip it and everyone is then lined up
		for(unsigned int across_id = 0; across_id < total_across; across_id++) {
			face_order[0] = across_id + 2 + across_count + across_layer_end[0]; //alpha
			face_order[1] = across_id + 2 + across_count + across_layer_end[1]; //beta
			face_order[2] = across_id + 3 + across_count + across_layer_end[2]; //epsilon
			p_faceIndicies[face_counter++] = face_order;
			std::cout << "Across Face Order ID " << layer_id << ", " << across_id << " " << face_order << "\n";
		}
		across_layer_end = face_order;
		//std::cout << "stored prev layer order ID " << across_layer_end << "\n";
	}
	//std::cout << "created " << face_counter << " faces | shoulda made " << triangle::calculateFaceCount(p_subdivisonCount) << "\n";
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
		}
	}	
}

//read left to right. so ya gonna go 1, 2, 3, 4 mid points per layer in arithemtic sum
//if it stime to start new layer then go down to the left from row start else go across by mid point distance