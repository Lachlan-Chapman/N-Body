#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main(int argc, char** argv) {
	if(!glfwInit()) {
		std::cerr << "Failed to init GLFW\n";
		return -1;
	}

	GLFWwindow *window = glfwCreateWindow(800, 600, "OpenGL Context", nullptr, nullptr);\
	if(!window) {
		std::cerr << "Failed to create window\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	if(glewInit() != GLEW_OK) {
		std::cerr << "Failed to init GLEW\n";
		return -1;
	}

	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";

	while(!glfwWindowShouldClose(window)) {
		glClearColor(0.2f, 0.0f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window); //swaps the double buffer
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}