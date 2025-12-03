#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

#include "graphics/OpenGL.hpp"
#include "graphics/Cuda.hpp"
#include "graphics/OpenCuda.hpp"

#include "observer/camera.hpp"

#include "shapes/octahedron.hpp"



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

	int vertex_count = 6;
	int vbo_size = sizeof(float) * 3 * vertex_count;
	GLuint vbo_handle = OpenGL::mallocVBO(vbo_size, GL_STATIC_DRAW); //read only for vertex data
	
	//object data setup
	GLuint vao_handle = OpenGL::mallocVAO(1);
	glBindVertexArray(vao_handle); //bind to this vao that are we going to define the setup in the vbo setup
	
	glBindBuffer(GL_ARRAY_BUFFER, vbo_handle); //attach vbo to the bounded vao ^
	glBufferData(GL_ARRAY_BUFFER, vbo_size, Primitive::octahedron, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0); //location 0 of the vShader will take in what im about to point it to
	glVertexAttribPointer(
		0, //pointing to location 0 of the shader
		3, //3 items from the vbo
		GL_FLOAT, //data type of the array
		GL_FALSE, //dont normalise just read as is
		sizeof(float) * 3, //stride is in steps of 3 floats as im making a vec3
		(void*)0 //no offset
	);

	//create edges
	GLuint ebo_handle = OpenGL::mallocEBO(1);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_handle);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Primitive::octahedronEdge), Primitive::octahedronEdge, GL_STATIC_DRAW);

	std::string vert_shader = OpenGL::loadShader("shaders/vertex.vert");
	std::string frag_shader = OpenGL::loadShader("shaders/white.frag");
	if(vert_shader.empty() || frag_shader.empty()) { return 1; }
	
	GLuint vert_handler, frag_handler;
	if(!(vert_handler = OpenGL::compileShader(GL_VERTEX_SHADER, vert_shader.c_str()))) { return -1; }
	if(!(frag_handler = OpenGL::compileShader(GL_FRAGMENT_SHADER, frag_shader.c_str()))) { return -1; }

	GLuint prog_handler;
	if(!(prog_handler = OpenGL::linkProgram(vert_handler, frag_handler))) { return -1; }


	camera _cam(
		glm::vec3(0.0, 0.0, 3.0),
		glm::radians(90.0f),
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
		
		_cam.moveFlight(
			glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS,
			delta_time
		);

		// _cam.moveFPS(
		// 	glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS,
		// 	glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS,
		// 	glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS,
		// 	glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS,
		// 	glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS,
		// 	glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS,
		// 	delta_time
		// );



		double current_mouseX, current_mouseY;
		glfwGetCursorPos(window, &current_mouseX, &current_mouseY);
				
		double delta_mouseX = current_mouseX - last_mouseX;
		double delta_mouseY = current_mouseY - last_mouseY;
		last_mouseX = current_mouseX;
		last_mouseY = current_mouseY;

		_cam.rotate(
			delta_mouseX,
			delta_mouseY,
			glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS,
			glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS,
			delta_time
		);

		_cam.update();

		OpenGL::clearScreen();
		
		
		glUseProgram(prog_handler);
		
		//set the camera transform for 3d based on cameras state
		glUniformMatrix4fv(
			glGetUniformLocation(prog_handler, "u_projection"),
			1,
			GL_FALSE,
			glm::value_ptr(_cam.m_projection)
		);
		glUniformMatrix4fv(
			glGetUniformLocation(prog_handler, "u_view"),
			1,
			GL_FALSE,
			glm::value_ptr(_cam.m_view)
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
			glm::value_ptr(_cam.m_position)
		);
		
		glBindVertexArray(vao_handle);
		//glDrawArrays(GL_POINTS, 0, vertex_count); //draw points
		glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);

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