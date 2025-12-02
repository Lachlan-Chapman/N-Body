#include <iostream>
#include <string>

#include "graphics/open_gl_helper.hpp"
#include "graphics/cuda_helper.hpp"


#include "shapes/sphere.hpp"


void runDeformCube(float *data, float time, size_t bufferSize);

int main(int argc, char** argv) {
	
	GLFWwindow *window = createContext(
		800,
		600,
		"Context"
	);
	if(!window) { return 1; }

	
	if(!createGPU()) {
		std::cerr << "Unable to set gpu\n";
		return 1;
	}

	int vertex_count = 128;
	int vbo_size = sizeof(float) * 3 * vertex_count;
	GLuint sphere_vbo_handle = mallocVBO(vbo_size, GL_STATIC_DRAW); //a read only mesh
	cudaGraphicsResource *sphere_cuda_handle = bindVBO(sphere_vbo_handle); //cuda is allowed to write to this buffer

	lockVBO(sphere_cuda_handle); //cuda is about to modify this buffer so lock it
	
	size_t sphere_data_size;
	float *sphere_data = (float*) getVBO(&sphere_data_size, sphere_cuda_handle); //we now have a ptr right to the buffer that cuda is allowed to safely use
	fibonacciSphere(sphere_data, vertex_count);
	
	unlockVBO(sphere_cuda_handle); //give it back to OpenGL | also waits for all work on this resource to finish
	
	//vao setup
	GLuint sphere_vao_handle;
	glGenVertexArrays(1, &sphere_vao_handle);
	glBindVertexArray(sphere_vao_handle);

	//vbo setup
	glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo_handle);
	glEnableVertexAttribArray(0); //location 0 of the vShader will take in what im about to point it to
	glVertexAttribPointer(
		0, //pointing to location 0 of the shader
		3, //3 items from the vbo
		GL_FLOAT, //data type of the array
		GL_FALSE, //dont normalise just read as is
		sizeof(float) * 3, //stride is in steps of 3 floats as im making a vec3
		(void*)0 //no offset
	);

	std::string vert_shader = loadShader("shaders/fibonacci_sphere.vert");
	std::string frag_shader = loadShader("shaders/fibonacci_sphere.frag");
	if(vert_shader.empty() || frag_shader.empty()) { return 1; }
	
	GLuint vert_handler, frag_handler;
	if(!(vert_handler = compileShader(GL_VERTEX_SHADER, vert_shader.c_str()))) { return -1; }
	if(!(frag_handler = compileShader(GL_FRAGMENT_SHADER, frag_shader.c_str()))) { return -1; }

	GLuint prog_handler;
	if(!(prog_handler = linkProgram(vert_handler, frag_handler))) { return -1; }

	while(!glfwWindowShouldClose(window)) {
		if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, true);
		}

		glClear(
			GL_COLOR_BUFFER_BIT | //clear the color data
			GL_DEPTH_BUFFER_BIT //also clear the depth data since it wont run the frag shader if its depth is >= to optimise. and a static point is at the exact depth
		);
		glUseProgram(prog_handler);
		glBindVertexArray(sphere_vao_handle);

		glDrawArrays(GL_POINTS, 0, vertex_count);

		GLenum err;
		while ((err = glGetError()) != GL_NO_ERROR) {
			std::cerr << "GL ERROR: " << err << "\n";
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	destroyContext(window);
	glfwTerminate();
	return 0;
}