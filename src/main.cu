#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

#include "graphics/OpenGL.hpp"
#include "graphics/Cuda.hpp"
#include "graphics/OpenCuda.hpp"

#include "math/vec.hpp"
#include "benchmark/profiler.hpp"

#include "observer/cameraFlight.hpp"

#include "shapes/primitive.hpp"

#include "simulation/universe.hpp"
#include "simulation/octree.hpp"

#include "text/glyph.hpp"


#ifndef GIT_HASH
#define GIT_HASH "default"
#endif


int main(int argc, char** argv) {
	
	GLFWwindow *window = OpenGL::createContext(
		3840,
		2160,
		"Context"
	);
	if(!window) { return 1; }

	
	if(!Cuda::createGPU()) {
		std::cerr << "Unable to set gpu\n";
		return 1;
	}

	scopeTimer::selfTest();


	//particle sim
	universe omega(16384, 128, 256); //omega is just name i used for all test objects | bechmark obj
	octree *alpha = (octree*)Cuda::unifiedMalloc(sizeof(octree));

	int thread_count = 256;
	int block_count = (omega.m_particles->m_particleCount + thread_count - 1) / thread_count; //ensure more than enough blocks of 256 are dispatched

	//vao handler
	GLuint particle_vao = OpenGL::createVAO();
	OpenGL::bindVAO(particle_vao);

	//instance vbo, ebo, attribptr
	GLuint quad_vbo = OpenGL::createVBO();
	OpenGL::bindVBO(quad_vbo);
	OpenGL::setVBO(
		sizeof(vec3f) * Primitives::quad_mesh.d_vertexCount,
		Primitives::quad_vertices
	); //malloc and set qaud vertex data
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(vec3f),
		(void*)0
	);
	GLuint quad_ebo = OpenGL::createEBO();
	OpenGL::bindEBO(quad_ebo);
	OpenGL::setEBO(
		sizeof(Primitives::quad_faces), //3 vertices per face
		Primitives::quad_faces,
		GL_STATIC_DRAW
	);
	
	//positions vbo
	GLuint position_vbo = OpenGL::createVBO();
	OpenGL::bindVBO(position_vbo);
	OpenGL::setVBO(
		sizeof(vec3f) * omega.m_particles->m_particleCount,
		nullptr
	); //allocate raw size

	//set instancing based on positions
	glEnableVertexAttribArray(1); //now this vbo which is the positions are data per quad instance
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(vec3f),
		(void*)0
	);
	glVertexAttribDivisor(1, 1); //1:1 mapping from pos data to quad instance
	//shader creation
	std::string shader_path[2] = {
		"shaders/billboard_instance.vert",
		"shaders/white.frag"
	};

	std::string shader_src[2];
	for(int shader_id = 0; shader_id < 2; shader_id++) {
		shader_src[shader_id] = OpenGL::loadShader(shader_path[shader_id]);
		if(shader_src[shader_id].empty()) { return 1; }
	}

	GLuint shader_handles[2];
	if(!(shader_handles[0] = OpenGL::compileShader(GL_VERTEX_SHADER, shader_src[0].c_str()))) { return -1; }
	if(!(shader_handles[1] = OpenGL::compileShader(GL_FRAGMENT_SHADER, shader_src[1].c_str()))) { return -1; }

	GLuint billboard_handler;
	if(!(billboard_handler = OpenGL::linkProgram(shader_handles, 2))) { return -1; }

	OpenGL::bindVAO(0);


	
	//set init conditions
	cudaGraphicsResource *cuda_positions = OpenCuda::bindVBO(position_vbo);
	size_t position_size;
	vec3f* positions = (vec3f*)OpenCuda::getVBO(&position_size, cuda_positions); //this is the ptr to the graphics subsystem vbo with out positions


	//text
	if(!Text::buildGlyphMap("res/cascadiaCodeMono.fnt", vec2i(512, 512))) {
		return 1;
	}


	std::string text_shader_path[2] = {
		"shaders/text.vert",
		"shaders/text.frag"
	};
	std::string text_shader_src[2];
	for(int shader_id = 0; shader_id < 2; shader_id++) {
		text_shader_src[shader_id] = OpenGL::loadShader(text_shader_path[shader_id]);
		if(text_shader_src[shader_id].empty()) { return 1; }
	}
	GLuint txt_shader_handles[2];
	if(!(txt_shader_handles[0] = OpenGL::compileShader(GL_VERTEX_SHADER, text_shader_src[0].c_str()))) { return -1; }
	if(!(txt_shader_handles[1] = OpenGL::compileShader(GL_FRAGMENT_SHADER, text_shader_src[1].c_str()))) { return -1; }
	GLuint txt_handler;
	if(!(txt_handler = OpenGL::linkProgram(txt_shader_handles, 2))) { return -1; }
	glUseProgram(txt_handler);
	glUniform2f(glGetUniformLocation(txt_handler, "u_screenSize"), 3840.0f, 2160.0f);
	glUniform3f(glGetUniformLocation(txt_handler, "u_color"), 1.0f, 1.0f, 1.0f);
	glUniform1i(glGetUniformLocation(txt_handler, "u_atlas"), 0); //texture unit 0
	GLuint text_atlas_handle = OpenGL::loadPNG("res/cascadiaCodeMono.png", true);



	GLuint text_vao = OpenGL::createVAO();
	OpenGL::bindVAO(text_vao);
	GLuint text_vbo = OpenGL::createVBO();
	OpenGL::bindVBO(text_vbo);
	OpenGL::setVBO(
		sizeof(vec4f) * 6 * 128,
		nullptr,
		GL_DYNAMIC_DRAW
	); //allocate enough room for (x,y,u,v) * 6 vertices (2 tris) * 128 chars
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec4f), (void*)0); //location 0 will be xy
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vec4f), (void*)(sizeof(float)*2)); //location 1 will be uv
	
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //needed for alpha fonts




	//camera
	camera *_cam = new cameraFlight(
		glm::vec3(0.0, 0.0, 64.0),
		glm::radians(60.0f),
		16.0f/9.0f,
		0.1f,
		384.0f
	);



	float last_time = glfwGetTime();
	double last_mouseX, last_mouseY;
	glfwGetCursorPos(window, &last_mouseX, &last_mouseY);
	

	float universe_time = 0.0f;
	float universe_frequency = 1.0f / omega.m_frequency;
	int frame_counter = 0; //render 10 frames, we use frames 4th -> 7th
	while(!glfwWindowShouldClose(window)) {
		if(frame_counter == 55) {break;}
		std::clog << "\nFrame ID: " << frame_counter++ << "\n";
		scopeTimer frameTime("Frame Timer", std::clog);
		float current_time = glfwGetTime();
		float delta_time = current_time - last_time;
		last_time = current_time;
		
		{
			scopeTimer inputTimer("Input Timer", std::clog);
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
		}


		{
			scopeTimer treeBuild("Build Octree Timer", std::clog);
			alpha->build(omega);
		}
		
		{
			scopeTimer dispatchTimer("Dispatch Simulation Timer", std::clog);
			int step_count = 0;
			universe_time += delta_time; //this ticks up
			while(universe_time >= universe_frequency) { //when we are over the frequency it means we are due for a update
				universe_time -= universe_frequency; //reduce the time by however long a "step" takes if its 1 step per frame, then universe time will be less than the frequency
				++step_count;
			}
			if(step_count) {
				OpenCuda::lockVBO(cuda_positions); //we lock since we about to change it
				size_t position_size;
				vec3f* positions = (vec3f*)OpenCuda::getVBO(&position_size, cuda_positions); //this is the ptr to the graphics subsystem vbo with out positions
				float step_time;
				{
					scopeTimer stepTimer("All Simulation Step Time", std::clog);
					omega.step(
						alpha,
						positions,
						step_count
					);
					cudaDeviceSynchronize(); //wait for kernel to return
					step_time = stepTimer.microseconds();
				}
				OpenCuda::unlockVBO(cuda_positions);
				std::cout << "Single Simulation Step Time: " << step_time / step_count << "us\n";
			}
		}

		{
			scopeTimer renderTimer("Render Timer", std::clog);
			//rendering things
			OpenGL::clearScreen();
			
			
			glUseProgram(billboard_handler);

			//set the camera transform for 3d based on cameras state
			glUniformMatrix4fv(
				glGetUniformLocation(billboard_handler, "u_projection"),
				1,
				GL_FALSE,
				glm::value_ptr(_cam->m_projection)
			);
			glUniformMatrix4fv(
				glGetUniformLocation(billboard_handler, "u_view"),
				1,
				GL_FALSE,
				glm::value_ptr(_cam->m_view)
			);
			glUniformMatrix4fv(
				glGetUniformLocation(billboard_handler, "u_model"),
				1,
				GL_FALSE,
				glm::value_ptr(glm::mat4(1.0f))
			);
			glUniform3fv(
				glGetUniformLocation(billboard_handler, "u_cameraPosition"),
				1,
				glm::value_ptr(_cam->m_position)
			);
			
			float scale = 0.025f;
			glUniform1f(glGetUniformLocation(billboard_handler, "u_scale"), scale);
			
			OpenGL::bindVAO(particle_vao);
			glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, omega.m_particles->m_particleCount);
	
			//text drawing
			glDisable(GL_DEPTH_TEST); //not needed for on screen text
			glUseProgram(txt_handler);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, text_atlas_handle);
			
			vec4f txt_verts[6*128];
			int fps = static_cast<float>(1) / delta_time;
			std::string fps_str = "FPS| " + std::to_string(fps);
			int vertex_count = Text::buildTextVertices(fps_str, vec3f(0.0f, 0.0f, 1.0f), txt_verts);
			OpenGL::bindVBO(text_vbo);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec4f) * vertex_count, txt_verts);
	
			OpenGL::bindVAO(text_vao);
			glDrawArrays(GL_TRIANGLES, 0, vertex_count);
			glEnable(GL_DEPTH_TEST);
	
	
			// GLenum err;
			// while ((err = glGetError()) != GL_NO_ERROR) {
			// 	std::cerr << "GL ERROR: " << err << "\n";
			// }
	
			glfwSwapBuffers(window);
		}
		glfwPollEvents();
	}


	std::cout << "GIT HASH " << GIT_HASH << "\n";

	OpenGL::destroyContext(window);
	glfwTerminate();
	return 0;
}