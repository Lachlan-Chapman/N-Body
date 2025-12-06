#pragma once
#include "graphics/Cuda.hpp"
#include "math/vec.hpp"
struct mesh {
	vec3f const *d_vertices;
	unsigned int const *d_edgeIds;
	unsigned int const *d_faceIds;
	unsigned int d_vertexCount, d_edgeCount, d_faceCount;
};

constexpr float TAU = 6.2831855f;
constexpr float third_tau = TAU / 3.0f;

namespace Primitives {
	static vec3f octahedron_vertices[6] = {
		vec3f( 0.0f,  0.5f,  0.0f), //top
	
		vec3f( 0.5f,  0.0f,  0.5f),  //middle square
		vec3f(-0.5f,  0.0f,  0.5f),
		vec3f(-0.5f,  0.0f, -0.5f),
		vec3f( 0.5f,  0.0f, -0.5f),
	
		vec3f( 0.0f, -0.5f,  0.0f) //bottom
	};
	
	static constexpr unsigned int octahedron_edges[24] = {
		0,1, 0,2, 0,3, 0,4, //top pyramid
		1,2, 2,3, 3,4, 4,1, //middle sqaure
		1,5, 2,5, 3,5, 4,5  //bottom pyramid
	};

	static vec<3, unsigned int> octahedron_faces[8] = {
		vec<3, unsigned int>(0, 1, 2),
		vec<3, unsigned int>(0, 2, 3),
		vec<3, unsigned int>(0, 3, 4),
		vec<3, unsigned int>(0, 4, 1),
		vec<3, unsigned int>(5, 2, 1),
		vec<3, unsigned int>(5, 3, 2),
		vec<3, unsigned int>(5, 4, 3),
		vec<3, unsigned int>(5, 1, 4)
	};

	static mesh const octahedron_mesh = {
		octahedron_vertices,
		octahedron_edges,
		nullptr,
		6, 12, 8
	};

	static vec3f triangle_vertices[3] = {
		vec3f(cosf(0 * third_tau + 1.6f), sinf(0 * third_tau + 1.6f), 0.0f), //top
		vec3f(cosf(2 * third_tau + 1.6f), sinf(2 * third_tau + 1.6f), 0.0f),
		vec3f(cosf(1 * third_tau + 1.6f), sinf(1 * third_tau + 1.6f), 0.0f),  //middle square
	};
	
	static constexpr unsigned int triangle_edges[6] = {
		0,1, 1,2, 2,0
	};

	static constexpr unsigned int triangle_faces[3] = {
		0, 1, 2
	};

	static mesh const triangle_mesh = {
		triangle_vertices,
		triangle_edges,
		triangle_faces,
		3, 3, 1
	};

	static const float plane_vertices[] = {
		-5.0f, -10.0f, -5.0f,
		 5.0f, -10.0f, -5.0f,
		 5.0f, -10.0f,  5.0f,
		-5.0f, -10.0f,  5.0f
	};

	static const unsigned int plane_indices[] = {
		0, 1, 2,
		2, 3, 0
	};
}