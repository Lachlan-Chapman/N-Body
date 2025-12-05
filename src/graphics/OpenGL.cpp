#include <iostream>

#include "graphics/OpenGL.hpp"
namespace OpenGL {
	std::string loadShader(const std::string &p_path) {
		std::ifstream file(p_path);
		if(!file.is_open()) {
			std::cerr << "Failed to open " << p_path << "\n";
			std::clog << "Working Dir: " << std::filesystem::current_path() << "\n";
			return "";
		}
		std::stringstream ss;
		ss << file.rdbuf();
		return ss.str(); //convert to string ALWAYS
	}

	GLuint compileShader(GLenum p_type, const char* p_src) {
		GLuint shader = glCreateShader(p_type); //compile either frag or vert shader
		glShaderSource(
			shader, //use this handle and attach the shader to it
			1, //we are giving one long string
			&p_src, //ptr to the string of source code
			nullptr //let GL figure out the string length (internally uses strlen)
		);
		glCompileShader(shader);

		GLint success = 0;
		glGetShaderiv( //check for errors
			shader,
			GL_COMPILE_STATUS,
			&success
		);

		if(!success) {
			GLint log_length = 0;
			glGetShaderiv(
				shader,
				GL_INFO_LOG_LENGTH,  //ask for log size for that shader err
				&log_length
			);

			std::string err(log_length, 0);
			glGetShaderInfoLog(
				shader,
				log_length, //max char to write
				nullptr, //dont need the actual length here
				err.data() //the buffer to write too
			);

			std::cerr << "!!Shader Failed To Compile!!\n" << err << "\n";
			return 0; //failure signal
		}
		std::clog << "~Shader Successfully Compiled~\n";
		return shader; //return the non zero shader handle as success
	}

	GLuint linkProgram(GLuint p_vertex, GLuint p_frag) {
		GLuint program = glCreateProgram();
		glAttachShader(
			program,
			p_vertex
		);

		glAttachShader(
			program,
			p_frag
		);

		glLinkProgram(program);

		GLint success = 0;
		glGetProgramiv(
			program,
			GL_LINK_STATUS,
			&success
		);
		if(!success) {
			GLint log_length = 0;
			glGetProgramiv(
				program,
				GL_INFO_LOG_LENGTH,
				&log_length
			);

			std::string err(log_length, 0);
			glGetProgramInfoLog(
				program,
				log_length, //max char to write
				nullptr, //dont need the actual length here
				err.data() //the buffer to write too
			);

			std::cerr << "!!Program Failed To Compile!!\n" << err << "\n";
			return 0; //failure signal
		}

		//if link worked, then the "object" files for the shaders are to be removed
		glDetachShader(program, p_vertex);
		glDetachShader(program, p_frag);
		glDeleteShader(p_vertex);
		glDeleteShader(p_frag);
		return program; //non zero program handler is a winner
	}

	GLFWwindow* createContext(
		int p_width,
		int p_height,
		const char* p_name,
		GLFWmonitor *p_monitor,
		GLFWwindow *p_shared
	) {
		if(!glfwInit()) {
			std::cerr << "Failed to init GLFW\n";
			return nullptr;
		}

		GLFWwindow *window = glfwCreateWindow(
			p_width,
			p_height,
			p_name,
			p_monitor,
			p_shared
		);
		
		if(!window) {
			std::cerr << "Failed to create window\n";
			return nullptr;
		}

		glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE); //allow keys
		//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); //fix cursor
		glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE); //allow mouse delta input

		glfwMakeContextCurrent(window); //set the context bound to this window

		if(!gladLoadGL()) {
			std::cerr << "Failed to init GLEW\n";
			return nullptr;
		}

		std::cout << "GLFW+GLEW | OpenGL Version: " << glGetString(GL_VERSION) << "\n";

		glEnable(GL_DEPTH_TEST); //enable 3d depth testing

		int buff_width, buff_height;
		glfwGetFramebufferSize(window, &buff_width, &buff_height);
		glViewport(0, 0, buff_width, buff_height);
		glEnable(GL_PROGRAM_POINT_SIZE);

		return window;
	}

	void destroyContext(GLFWwindow *p_window) {
		glfwDestroyWindow(p_window);
	}

	void clearScreen() { //wipe all data from previous frame both color and depth
		glClear(
			GL_COLOR_BUFFER_BIT | //clear the color data
			GL_DEPTH_BUFFER_BIT //also clear the depth data since it wont run the frag shader if its depth is >= to optimise. and a static point is at the exact depth
		);
	}

	GLuint mallocVBO(int p_size, GLenum p_usage) {
		GLuint vbo_handle;
		glGenBuffers(1, &vbo_handle);
		return vbo_handle;
	}

	GLuint mallocVAO(unsigned int p_count) {
		GLuint vao_handle;
		glGenVertexArrays(p_count, &vao_handle);
		return vao_handle;
	}

	GLuint mallocEBO(unsigned int p_count) {
		GLuint ebo_handle;
		glGenBuffers(p_count, &ebo_handle);
		return ebo_handle;
	}
}