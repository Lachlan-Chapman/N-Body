#include "observer/cameraFlight.hpp"
void cameraFlight::move(
	bool p_w,
	bool p_a,
	bool p_s,
	bool p_d,
	bool p_space,
	bool p_alt,
	float p_deltaTime
) {
	float distance = m_speed * p_deltaTime;

	if(p_w) { m_position += localForward() * distance; }
	if(p_a) { m_position -= localRight() * distance; }
	if(p_s) { m_position -= localForward() * distance; }
	if(p_d) { m_position += localRight() * distance; }

	if(p_space) { m_position += localUp() * distance; }
	if(p_alt) { m_position -= localUp() * distance; }
}

glm::quat cameraFlight::yaw(float p_radians) const {
	return glm::angleAxis(p_radians, glm::normalize(localUp()));
}

void cameraFlight::rotate(
	float p_mouseX,
	float p_mouseY,
	bool p_q,
	bool p_e,
	float p_deltaTime
) {
	float _pitch = -p_mouseY * m_sensitivity;
	float _yaw = -p_mouseX * m_sensitivity;

	glm::quat delta_pitch = pitch(_pitch);
	glm::quat delta_yaw = yaw(_yaw); //child class changes this to determine FPS or flight control

	float _roll = 0.0f;
	static const float scale = 100.0f; //adhoc scale for now to handle keyboard based rotation
	if (p_q) { _roll -= m_sensitivity * scale * p_deltaTime; }
	if (p_e) { _roll += m_sensitivity * scale * p_deltaTime; }
	glm::quat delta_roll = roll(_roll);

	//apply all rotations at once
	m_orientation = glm::normalize(delta_roll * delta_pitch * delta_yaw * m_orientation);
}