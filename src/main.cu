#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

#include "graphics/OpenGL.hpp"
#include "graphics/Cuda.hpp"
#include "graphics/OpenCuda.hpp"

#include "observer/cameraFlight.hpp"
#include "observer/cameraFPS.hpp"

#include "shapes/octahedron.hpp"

#include "math/vec.hpp"

#include "simulation/universe.hpp"


__global__ void mapPositions(vec3f *p_vbo, particles *p_particles) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= p_particles->m_particleCount) { return; }

	p_vbo[idx] = vec3f(
		p_particles->m_posX[idx],
		p_particles->m_posY[idx],
		p_particles->m_posZ[idx]
	);
}

int main(int argc, char** argv) {
	
	GLFWwindow *window = OpenGL::createContext(
		1920,
		1080,
		"Context"
	);
	if(!window) { return 1; }

	
	if(!Cuda::createGPU()) {
		std::cerr << "Unable to set gpu\n";
		return 1;
	}

	

	std::string vert_shader = OpenGL::loadShader("shaders/instance.vert");
	std::string frag_shader = OpenGL::loadShader("shaders/white.frag");
	if(vert_shader.empty() || frag_shader.empty()) { return 1; }
	
	GLuint vert_handler, frag_handler;
	if(!(vert_handler = OpenGL::compileShader(GL_VERTEX_SHADER, vert_shader.c_str()))) { return -1; }
	if(!(frag_handler = OpenGL::compileShader(GL_FRAGMENT_SHADER, frag_shader.c_str()))) { return -1; }

	GLuint prog_handler;
	if(!(prog_handler = OpenGL::linkProgram(vert_handler, frag_handler))) { return -1; }
	

	camera *_cam = new cameraFPS(
		glm::vec3(0.0, 0.0, 3.0),
		glm::radians(60.0f),
		16.0f/9.0f,
		0.1f,
		100.0f
	);

	//particle sim
	universe omega(2, 1); //omega is just name i used for all test objects
	int thread_count = 256;
	int block_count = (omega.m_particles->m_particleCount + thread_count - 1) / thread_count; //ensure more than enough blocks of 256 are dispatched

	//vao handler
	GLuint octahedron_vao = OpenGL::createVAO();
	OpenGL::bindVAO(octahedron_vao);
	
	//instance vbo
	GLuint octahedron_vbo = OpenGL::createVBO();
	OpenGL::bindVBO(octahedron_vbo);
	OpenGL::setVBO(
		sizeof(vec3f) * Primitives::octahedron_mesh.d_vertexCount,
		Primitives::octahedron_vertices
	);

	glEnableVertexAttribArray(0); //make it a mesh vertex
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(vec3f),
		(void*)0
	);

	//instance ebo
	GLuint octahedron_ebo = OpenGL::createEBO();
	OpenGL::bindEBO(octahedron_ebo);
	OpenGL::setEBO(
		sizeof(vec<3, unsigned int>) * Primitives::octahedron_mesh.d_faceCount,
		Primitives::octahedron_faces
	);


	//positions vbo
	GLuint position_vbo = OpenGL::createVBO();
	OpenGL::bindVBO(position_vbo);
	
	//manual instance testing
	vec3f test_pos[2] = {vec3f(0.0f), vec3f(2.0)};
	OpenGL::setVBO(
		sizeof(vec3f) * 2,
		&test_pos
	);

	//set instancing based on positions
	glEnableVertexAttribArray(1); //now this vbo which is the positions are data per instance
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(vec3f),
		(void*)0
	);
	glVertexAttribDivisor(1, 1); //1 to 1 mapping of pos vbo data to mesh vbo instance
	glBindVertexArray(0); //unbind vao marking the end of the setup

	cudaGraphicsResource *cuda_positions = OpenCuda::bindVBO(position_vbo);


	float last_time = glfwGetTime();
	double last_mouseX, last_mouseY;
	glfwGetCursorPos(window, &last_mouseX, &last_mouseY);
	
	while(!glfwWindowShouldClose(window)) {
		float current_time = glfwGetTime();
		float delta_time = current_time - last_time;
		last_time = current_time;
		if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, true);
		}
		
		_cam->move(
			glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS,
			delta_time
		);



		double current_mouseX, current_mouseY;
		glfwGetCursorPos(window, &current_mouseX, &current_mouseY);
				
		double delta_mouseX = current_mouseX - last_mouseX;
		double delta_mouseY = current_mouseY - last_mouseY;
		last_mouseX = current_mouseX;
		last_mouseY = current_mouseY;

		_cam->rotate(
			delta_mouseX,
			delta_mouseY,
			glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS,
			delta_time
		);

		_cam->update();

		// omega.step();
		// OpenCuda::lockVBO(cuda_positions); //we lock since we about to change it
		// size_t position_size;
		// vec3f* positions = (vec3f*)OpenCuda::getVBO(&position_size, cuda_positions); //this is the ptr to the graphics subsystem vbo with out positions
		// mapPositions<<<block_count, thread_count>>>(positions, omega.m_particles); //the vbo after this contains the positions as vec3
		// OpenCuda::unlockVBO(cuda_positions);



		//rendering things
		OpenGL::clearScreen();
		
		
		glUseProgram(prog_handler);
		
		//set the camera transform for 3d based on cameras state
		glUniformMatrix4fv(
			glGetUniformLocation(prog_handler, "u_projection"),
			1,
			GL_FALSE,
			glm::value_ptr(_cam->m_projection)
		);
		glUniformMatrix4fv(
			glGetUniformLocation(prog_handler, "u_view"),
			1,
			GL_FALSE,
			glm::value_ptr(_cam->m_view)
		);
		glUniformMatrix4fv(
			glGetUniformLocation(prog_handler, "u_model"),
			1,
			GL_FALSE,
			glm::value_ptr(glm::mat4(1.0f))
		);
		glUniform3fv(
			glGetUniformLocation(prog_handler, "u_cameraPosition"),
			1,
			glm::value_ptr(_cam->m_position)
		);
		
		glBindVertexArray(octahedron_vao);
		glDrawElementsInstanced(
			GL_TRIANGLES,
			Primitives::octahedron_mesh.d_faceCount * 3,
			GL_UNSIGNED_INT,
			(void*)0,
			2 //omega.m_particles->m_particleCount
		);



		GLenum err;
		while ((err = glGetError()) != GL_NO_ERROR) {
			std::cerr << "GL ERROR: " << err << "\n";
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	OpenGL::destroyContext(window);
	glfwTerminate();
	return 0;
}