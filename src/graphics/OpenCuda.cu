#include "graphics/OpenCuda.hpp"
namespace OpenCuda {
	cudaGraphicsResource* bindVBO(GLuint p_handle) { //tells cuda "hey btw this opengl buffer you can map then write to it if you want"
		cudaGraphicsResource *cuda_handle;
		cudaGraphicsGLRegisterBuffer(
			&cuda_handle,
			p_handle,
			cudaGraphicsMapFlagsWriteDiscard
		);
		return cuda_handle;
	}

	void lockVBO(cudaGraphicsResource *p_cudaHandle) { //this says to opengl "hey btw cuda is about to use this buffer, so you cant change it"
		cudaGraphicsMapResources(
			1,
			&p_cudaHandle,
			0
		);
	}

	void* getVBO(size_t *p_size, cudaGraphicsResource *p_cudaHandle) { //p_data will now point to memory in the graphics subsystem
		void* data_ptr = nullptr;
		size_t set_size = 0;
		cudaGraphicsResourceGetMappedPointer(
			&data_ptr,
			&set_size,
			p_cudaHandle
		);
		if(p_size) { *p_size = set_size; }
		return data_ptr;
	}

	void unlockVBO(cudaGraphicsResource *p_cudaHandle) { //this says to opengl "hey btw cuda is about to use this buffer, so you cant change it"
		cudaGraphicsUnmapResources(
			1,
			&p_cudaHandle,
			0
		);
	}
}