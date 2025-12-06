#pragma once
#include "graphics/Cuda.hpp"
#include "simulation/particle.hpp"

class universe {
public:
	universe() = delete;
	universe(size_t p_particleCount, unsigned int p_frequency, float p_radius);
	void calculateAcceleration(); //dispatches kernel
	void integrate(); //dispatches kernel, host called
	void step();
	particles *m_particles;
	float m_frequency;
};