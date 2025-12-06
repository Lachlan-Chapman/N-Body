#pragma once
#include "graphics/Cuda.hpp"

struct particles {
	float *m_accX, *m_accY, *m_accZ;
	float *m_velX, *m_velY, *m_velZ;
	float *m_posX, *m_posY, *m_posZ;
	float *m_mass;
	size_t m_particleCount;
};