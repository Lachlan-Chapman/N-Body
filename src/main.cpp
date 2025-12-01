#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <iostream>


#include "cube.hpp"

void runDeformCube(float *data, float time, size_t bufferSize);

int main(int argc, char** argv) {
	if(!glfwInit()) {
		std::cerr << "Failed to init GLFW\n";
		return -1;
	}

	GLFWwindow *window = glfwCreateWindow(800, 600, "OpenGL Context", nullptr, nullptr);
	if(!window) {
		std::cerr << "Failed to create window\n";
		glfwTerminate();
		return -1;
	}
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

	glfwMakeContextCurrent(window);
	if(glewInit() != GLEW_OK) {
		std::cerr << "Failed to init GLEW\n";
		return -1;
	}

	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";

	/* cube rendering section START */
	
		/* building the shaders on the host */
		GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vShader, 1, &vertexShaderSrc, nullptr);
		glCompileShader(vShader); //we have set and asked to compile our vertex shader

		GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fShader, 1, &fragmentShaderSrc, nullptr);
		glCompileShader(fShader); //the shader that runs per fragment from the rasterisation step

		GLuint shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vShader);
		glAttachShader(shaderProgram, fShader);
		glLinkProgram(shaderProgram);

		glDeleteShader(vShader);
		glDeleteShader(fShader);

		/* creating the VAO, VBO, EBO */
		GLuint VAO, VBO, EBO;
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);

		//vbo
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

		cudaGraphicsResource *cuda_vbo_resource;
		cudaGraphicsGLRegisterBuffer( //we say to cuda that this buffer belongs to OpenGL but you can write to it
			&cuda_vbo_resource,
			VBO,
			cudaGraphicsMapFlagsWriteDiscard
		);

		//EBO
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices, GL_STATIC_DRAW);

		//VAO - so pass in 6 floats but say hey youll find location 0 in the first 3
		//location = 0 being the first 3 floats
		glVertexAttribPointer(
			0, //location 0 in the shader
			3, //read in 3 floats
			GL_FLOAT, //type of data
			GL_FALSE, //dont normalise so read the floats exactly as is | if the values were ints 0-255 then normalise these back to floats [-1, 1]
			6 * sizeof(float), //one vertex is 6 floats of pos and col
			(void*)0 //start at offset 0
		);
		glEnableVertexAttribArray(0);

		//loation = 1 being the color the next 3 floats in the cube vertices arr
		glVertexAttribPointer(
			1, //location 1
			3,
			GL_FLOAT,
			GL_FALSE,
			6 * sizeof(float),
			(void*)(3 * sizeof(float)) //start at offset 3. so from [3]
		);
		glEnableVertexAttribArray(1); //tell that this item is to be read as color

		glBindVertexArray(0);
		glEnable(GL_DEPTH_TEST);


		//rendering loop
		while(!glfwWindowShouldClose(window)) {
			if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				glfwSetWindowShouldClose(window, true);
			}

			float time = glfwGetTime();

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clears the color and depth buffer

			//view matrix to allow 3d perspective
			glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0.5, 1.0f, 0.2f));
			glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -3));
			glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.f/600.f, 0.1f, 100.f);
			glm::mat4 mvp = proj * view * model;

			//actually use the shaders
			glUseProgram(shaderProgram);
			GLuint mvpLocation = glGetUniformLocation(shaderProgram, "uMVP"); //finds the mat4 declared in the vertex shader
			glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, &mvp[0][0]);

			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(cubeVertices), cubeVertices); //reset data as the kernel is destructive overwritting the buffer data leading to a collpase of values to 0

			size_t bufferSize;
			float *dptr;
			cudaGraphicsMapResources(1, &cuda_vbo_resource);
			cudaGraphicsResourceGetMappedPointer(
				(void**)&dptr,
				&bufferSize,
				cuda_vbo_resource
			);

			runDeformCube(dptr, time, bufferSize);
			cudaDeviceSynchronize();
			cudaError_t err = cudaGetLastError();
			if(err != cudaSuccess) {
				std::cerr << "*** Cuda Error: " << cudaGetErrorString(err) << "***\n";
			}

			cudaGraphicsUnmapResources(1, &cuda_vbo_resource); //give it back to OpenGL


			//draw the cube
			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	/* cube rendering section END */

	//clean up after exiting
	cudaGraphicsUnregisterResource(cuda_vbo_resource);
	glDeleteVertexArrays(1, &VAO);
	glDeleteVertexArrays(1, &VBO);
	glDeleteVertexArrays(1, &EBO);

	glDeleteProgram(shaderProgram);
	glfwDestroyWindow(window);
	glfwTerminate();
	
	cudaDeviceReset();

	return 0;
}