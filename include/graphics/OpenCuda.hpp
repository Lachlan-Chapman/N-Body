#pragma once
#include <cuda_gl_interop.h>
#include <GLAD/types.hpp>

//all items relating to the cross between rendering and computer subsystems
namespace OpenCuda {
	cudaGraphicsResource* bindVBO(GLuint p_handle); //tells cuda "hey btw this opengl buffer you can map then write to it if you want"
	
	void lockVBO(cudaGraphicsResource *p_cudaHandle); //this says to opengl "hey btw cuda is about to use this buffer, so you cant change it"
	
	void* getVBO(size_t *p_size, cudaGraphicsResource *p_cudaHandle); //p_data will now point to memory in the graphics subsystem
	
	void unlockVBO(cudaGraphicsResource *p_cudaHandle); //tells OpenGL cuda wont be trynna write so sync it across to OpenGL

}

/*
 * steps to create vetex data that cuda can write to
 * create a vbo handle with OpenGL::mallocVBO() with enough floats for n vertices of vec3 so (sizeof(float) * 3 * vertex_count)
 * then bind this VBO handle to cuda telling OpenGL that cuda will write to this buffer with OpenCuda::bindVBO()
 * then lock the VBO before writing giving cuda control over the data with OpenCuda::lockVBO()
 * get the ptr to the data that the kernel needs to know where the memory is to operate on with OpenCuda::getVBO()
 * then run the kernel manipulating the VBO data
 * then unlock the VBO telling OpenGL that cuda is done changing it with OpenCuda::unlockVBO() 
*/