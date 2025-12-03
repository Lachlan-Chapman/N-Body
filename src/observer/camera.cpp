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


glm::quat camera::pitch(float p_radians) const {
	return glm::angleAxis(p_radians, glm::normalize(localRight()));
}
glm::quat camera::roll(float p_radians) const {
	return glm::angleAxis(p_radians, glm::normalize(localForward()));
}

inline glm::vec3 camera::worldForward() const {
	return glm::vec3(0, 0, -1);
}
inline glm::vec3 camera::worldUp() const {
	return glm::vec3(0, 1, 0);
}
inline glm::vec3 camera::worldRight() const {
	return glm::vec3(1, 0, 0);
}

glm::vec3 camera::localForward() const {
	return m_orientation * worldForward();
}
glm::vec3 camera::localUp() const {
	return m_orientation * worldUp();
}
glm::vec3 camera::localRight() const {
	return m_orientation * worldRight();
}

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
inline glm::mat4 camera::model() const {
	return glm::mat4(1.0f); //identity since the camera isnt rendered as an object
}

void camera::update() {
	m_projection = projection();
	m_view = view();
	m_model = model();
	m_mvp = m_projection * m_view * m_model;
}
