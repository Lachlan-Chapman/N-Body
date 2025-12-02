#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "GLAD/types.hpp"

bool createGPU();

cudaGraphicsResource* bindVBO(GLuint p_handle); //tells cuda "hey btw this opengl buffer you can map then write to it if you want"

void lockVBO(cudaGraphicsResource *p_cudaHandle); //this says to opengl "hey btw cuda is about to use this buffer, so you cant change it"

void* getVBO(size_t *p_size, cudaGraphicsResource *p_cudaHandle); //p_data will now point to memory in the graphics subsystem

void unlockVBO(cudaGraphicsResource *p_cudaHandle); //tells OpenGL cuda wont be trynna write so sync it across to OpenGL