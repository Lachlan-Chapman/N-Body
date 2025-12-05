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



void runDeformCube(float *data, float time, size_t bufferSize);

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

	

	std::string vert_shader = OpenGL::loadShader("shaders/vertex.vert");
	std::string frag_shader = OpenGL::loadShader("shaders/white.frag");
	if(vert_shader.empty() || frag_shader.empty()) { return 1; }
	
	GLuint vert_handler, frag_handler;
	if(!(vert_handler = OpenGL::compileShader(GL_VERTEX_SHADER, vert_shader.c_str()))) { return -1; }
	if(!(frag_handler = OpenGL::compileShader(GL_FRAGMENT_SHADER, frag_shader.c_str()))) { return -1; }

	GLuint prog_handler;
	if(!(prog_handler = OpenGL::linkProgram(vert_handler, frag_handler))) { return -1; }

	triangle _tri;
	vec3f *new_vertices;
	unsigned int new_count = 0;
	_tri.subdivide(3, new_vertices, new_count);

	GLuint tri_vao = OpenGL::mallocVAO();
	glBindVertexArray(tri_vao);
	GLuint tri_vbo = OpenGL::mallocVBO(sizeof(vec3f) * 3);
	glBindBuffer(GL_ARRAY_BUFFER, tri_vbo);
	glBufferData(
		GL_ARRAY_BUFFER,
		sizeof(vec3f) * new_count,
		new_vertices,
		GL_STATIC_DRAW
	);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(vec3f),
		(void*)0
	);


	camera *_cam = new cameraFPS(
		glm::vec3(0.0, 0.0, 3.0),
		glm::radians(60.0f),
		16.0f/9.0f,
		0.1f,
		100.0f
	);

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
		
		glBindVertexArray(tri_vao);
		glDrawArrays(GL_POINTS, 0, new_count); //draw points
		//glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);

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