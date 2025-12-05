#include "shapes/octahedron.hpp"

#include "graphics/Cuda.hpp"
#include "graphics/OpenGL.hpp"
#include "graphics/OpenCuda.hpp"



octahedron::octahedron() :
	octahedron(0)
{}

octahedron::octahedron(unsigned int p_subdivisionCount) :
	m_subdivisionCount(p_subdivisionCount)
{
	m_baseMesh = Primitives::octahedron_mesh; //we have out simple data for the base mesh
}

void octahedron::subdivideEdges() {

}
