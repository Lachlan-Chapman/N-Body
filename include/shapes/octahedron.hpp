#pragma once
namespace Primitive {
	float octahedron[] = {
		 0.0f,  0.5f,  0.0f, //top
	
		 0.5f,  0.0f,  0.5f,  //middle square
		-0.5f,  0.0f,  0.5f,
		-0.5f,  0.0f, -0.5f,
		 0.5f,  0.0f, -0.5f,
	
		 0.0f, -0.5f,  0.0f //bottom
	};

	unsigned int octahedronEdge[] = {
		0,1, 0,2, 0,3, 0,4, //top pyramid
		1,2, 2,3, 3,4, 4,1, //middle sqaure
		1,5, 2,5, 3,5, 4,5  //bottom pyramid
	};
}