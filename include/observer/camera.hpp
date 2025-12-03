#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
class camera {
public:
	camera() = delete; //must define the cameras parameters on any creation
	camera(
		glm::vec3 p_position,
		float p_fov,
		float p_aspectRatio,
		float p_nearPlane,
		float p_farPlane
	);

	void lookAt(
		glm::vec3 p_target,
		glm::vec3 p_up = glm::vec3(0.0f, 1.0f, 0.0f)
	);

	glm::mat4 mvp() const;

	glm::vec3 m_position;
	glm::mat4 m_view, m_model, m_projection, m_mvp;
private:
	glm::vec3 m_target, m_up;
	float m_fov, m_aspectRatio, m_nearPlane, m_farPlane;
};