#pragma once
//opengl the graphics library that will talk to the rendering sub system of our gpu
//glfw handling the input, window and context of OpenGL
//glew OpenGL function loader for older style OpenGL since OpenGL functions are not exposed by the os, they are loaded at runtime

#include <GLAD/glad.h>
#include <GLFW/glfw3.h>

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>

//why namespaces?
//i either make them free functions which dont have the same amount readability as a class
//i dont want to make an instance for the context
//i dont want to make static functions of a class as there is never some global state to track so the semantics are messy
//name spaces are the viable option here, i believe. Trying to keep as functional as we can for the more hardware related items
namespace OpenGL {
	std::string loadShader(const std::string &p_path);
	
	GLuint compileShader(
		GLenum p_type, //GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER etc
		const char* p_src
	);
	
	GLuint linkProgram( //creates a pipeline obj that performs a pipeline of processes
		GLuint *p_shaderHandle,
		unsigned int p_size
	);
	
	GLFWwindow* createContext(
		int p_width,
		int p_height,
		const char* p_name,
		GLFWmonitor *p_monitor = nullptr,
		GLFWwindow *p_shared = nullptr
	);
	
	void destroyContext(GLFWwindow *p_window);

	void clearScreen();
	
	GLuint createVAO();
	void bindVAO(GLuint p_hanlde);
	
	GLuint createVBO();
	void bindVBO(GLuint p_hanlde);
	//ensure you bind what you wanna set prior, leaving that to the caller as thats how open gl does it
	void setVBO(size_t p_size, void const *p_data, GLenum p_usage = GL_STATIC_DRAW);

	GLuint createEBO();
	void bindEBO(GLuint p_hanlde);
	//ensure you bind what you wanna set prior, leaving that to the caller as thats how open gl does it
	void setEBO(size_t p_size, void const *p_data, GLenum p_usage = GL_STATIC_DRAW);

	GLuint loadPNG(const std::string &p_path, bool p_alpha);
}
