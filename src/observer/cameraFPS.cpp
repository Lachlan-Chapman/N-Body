#include "observer/cameraFPS.hpp"
void cameraFPS::move(
	bool p_w,
	bool p_a,
	bool p_s,
	bool p_d,
	bool p_space,
	bool p_alt,
	float p_deltaTime
) {
	float distance = this->m_speed * p_deltaTime;

	//remove the body y to move on the xz plane only
	glm::vec3 flat_forward = localForward();
	flat_forward.y = 0.0f;
	flat_forward = glm::normalize(flat_forward);

	glm::vec3 flat_right= localRight();
	flat_right.y = 0.0f;
	flat_right = glm::normalize(flat_right);
	
	if(p_w) { this->m_position += flat_forward * distance; }
	if(p_a) { this->m_position -= flat_right * distance; }
	if(p_s) { this->m_position -= flat_forward * distance; }
	if(p_d) { this->m_position += flat_right * distance; }

	if(p_space) { this->m_position += worldUp() * distance; }
	if(p_alt) { this->m_position -= worldUp() * distance; }
}

glm::quat cameraFPS::yaw(float p_radians) const {
	return glm::angleAxis(p_radians, glm::normalize(localUp()));
}

void cameraFPS::rotate(
	float p_mouseX,
	float p_mouseY,
	bool p_q,
	bool p_e,
	float p_deltaTime
) {}