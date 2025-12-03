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
	m_target(glm::vec3(0.0f, 0.0f, 0.0f)),
	m_up(glm::vec3(0.0f, 1.0f, 0.0f))
{
	m_model = glm::mat4(1.0); //simple no object transforms by the camera
	m_view = glm::lookAt(
		m_position,
		m_target,
		m_up
	);
	m_projection = glm::perspective(
		m_fov,
		m_aspectRatio,
		m_nearPlane,
		m_farPlane
	);

}

void camera::lookAt( //set a global look at target
	glm::vec3 p_target,
	glm::vec3 p_up
) {
	m_target = p_target;
	m_up = p_up;
	m_view = m_view = glm::lookAt(
		m_position,
		m_target,
		m_up
	);
}

glm::mat4 camera::mvp() const { //unoptimal right now having to do the math each call
	return m_projection * m_view * m_model;
}