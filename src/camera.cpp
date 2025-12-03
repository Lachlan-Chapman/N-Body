#include "observer/camera.hpp"

camera::camera(
	glm::vec3 p_position,
	float p_fov,
	float p_aspectRatio,
	float p_nearPlane,
	float p_farPlane
) :
	m_position(p_position),
	m_fov(p_fov),
	m_aspectRatio(p_aspectRatio),
	m_nearPlane(p_nearPlane),
	m_farPlane(p_farPlane),
	m_orientation(glm::quat(1, 0, 0, 0))
{}


/* ROTATION */
void camera::pitch(float p_radians) {
	glm::quat _quaternion = glm::angleAxis(p_radians, glm::normalize(localRight()));
	m_orientation = glm::normalize(_quaternion * m_orientation);
}

void camera::yaw(float p_radians) {
	glm::quat _quaternion = glm::angleAxis(p_radians, glm::normalize(localUp()));
	m_orientation = glm::normalize(_quaternion * m_orientation);
}

void camera::roll(float p_radians) {
	glm::quat _quaternion = glm::angleAxis(p_radians, glm::normalize(localForward()));
	m_orientation = glm::normalize(_quaternion * m_orientation);
}

void camera::rotate(
	float p_mouseX,
	float p_mouseY,
	bool p_q,
	bool p_e,
	float p_deltaTime
) {
	yaw(-p_mouseX * m_sensitivity);
	pitch(-p_mouseY * m_sensitivity);
	float delta_roll = 0.0f;
	float scale = 100.0f;
	if(p_q) { delta_roll -= m_sensitivity * p_deltaTime * scale; }
	if(p_e) { delta_roll += m_sensitivity * p_deltaTime * scale; }
	if(delta_roll != 0.0f) { roll(delta_roll); }
}

/* MOVEMENT */
glm::vec3 camera::localForward() const {
	return m_orientation * glm::vec3(0, 0, -1);
}
glm::vec3 camera::localUp() const {
	return m_orientation * glm::vec3(0, 1, 0);
}
glm::vec3 camera::localRight() const {
	return m_orientation * glm::vec3(1, 0, 0);
}

inline glm::vec3 camera::globalForward() const {
	return glm::vec3(0, 0, -1);
}
inline glm::vec3 camera::globalUp() const {
	return glm::vec3(0, 1, 0);
}
inline glm::vec3 camera::globalRight() const {
	return glm::vec3(1, 0, 0);
}

void camera::moveFlight(
	bool p_w,
	bool p_a,
	bool p_s,
	bool p_d,
	float p_deltaTime
) {
	float distance = m_speed * p_deltaTime;

	//flight using local vectors
	if(p_w) { m_position += localForward() * distance; }
	if(p_a) { m_position -= localRight() * distance; }
	if(p_s) { m_position -= localForward() * distance; }
	if(p_d) { m_position += localRight() * distance; }
}

void camera::moveFPS(
	bool p_w,
	bool p_a,
	bool p_s,
	bool p_d,
	bool p_space,
	bool p_alt,
	float p_deltaTime
) {
	float distance = m_speed * p_deltaTime;

	glm::vec3 flat_forward = localForward();
	flat_forward.y = 0.0f;
	flat_forward = glm::normalize(flat_forward);

	glm::vec3 flat_right= localRight();
	flat_right.y = 0.0f;
	flat_right = glm::normalize(flat_right);
	//flight using local vectors
	if(p_w) { m_position += flat_forward * distance; }
	if(p_a) { m_position -= flat_right * distance; }
	if(p_s) { m_position -= flat_forward * distance; }
	if(p_d) { m_position += flat_right * distance; }
	if(p_space) { m_position += globalUp() * distance; }
	if(p_alt) { m_position -= globalUp() * distance; }
}

/* PROJECTION */
glm::mat4 camera::projection() const {
	return glm::perspective(
		m_fov,
		m_aspectRatio,
		m_nearPlane,
		m_farPlane
	);
}
glm::mat4 camera::view() const {
	return
		glm::mat4_cast(glm::conjugate(m_orientation)) *
		glm::translate(glm::mat4(1.0f), -m_position);

}
glm::mat4 camera::model() const {
	return glm::mat4(1.0f); //identity since the camera isnt rendered as an object
	// return
	// 	glm::translate(glm::mat4(1.0f), m_position) *
	// 	glm::mat4_cast(m_orientation) *
	// 	glm::scale(glm::mat4(1.0), glm::vec3(1.0f));
}

/* SETTER */
void camera::update() {
	m_projection = projection();
	m_view = view();
	m_model = model();
	m_mvp = m_projection * m_view * m_model;
}
