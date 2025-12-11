#pragma once
#include "graphics/Cuda.hpp"
#include "simulation/particle.hpp"

class universe {
public:
	universe() = delete;
	universe(size_t p_particleCount, unsigned int p_frequency, float p_radius);
	void step(vec3f *p_positionVBO, int stepCount = 1);
	particles *m_particles;
	float m_frequency;
};