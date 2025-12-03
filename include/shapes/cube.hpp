#pragma once
namespace Primitive {
	//8 points top and bottom
	float cube[] = {
		// positions			/ colors
		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		-0.5f,  0.5f, -0.5f,
		
		-0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
	};
}

//EBO element buffer object store indicies to tell the gpu where the positions are found in the object data
//VBO vertex buffer object is a chunk of GPU memory to store attributes about out vertices like position and color
//VAO vertex array object doesnt store anything to do with the vertecies instead is store meta data about how to read the VBO
//vertex shader runs for each vertex, it reads attributes in the VBO and can change them and pass on new data to the rasteriser
//rasteriser is a fixed function hardware stage that converts the triangles into pixels or fragments. so it interpolates per vertex attribute across the triangle
//fragment shader runs for each of these frags and outputs a final color

const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos; //taken from the VBO via VAO
layout(location = 1) in vec3 aColor;

out vec3 vColor; //passed onto the frag shader

uniform mat4 uMVP; //the model view projection matrix in one

void main() {
	gl_Position = uMVP * vec4(aPos, 1.0);
	vColor = aColor;
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
	FragColor = vec4(vColor, 1.0);
}
)";