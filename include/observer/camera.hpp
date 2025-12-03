#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

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

	void moveFlight(
		bool p_w,
		bool p_a,
		bool p_s,
		bool p_d,
		float p_deltaTime
	);

	void moveFPS(
		bool p_w,
		bool p_a,
		bool p_s,
		bool p_d,
		bool p_space,
		bool p_alt,
		float p_deltaTime
	);

	void rotate(
		float p_mouseX,
		float p_mouseY,
		bool p_q,
		bool p_e,
		float p_deltaTime
	);

	void update();

	glm::vec3 m_position;
	glm::mat4 m_view, m_model, m_projection, m_mvp;
private:
	void pitch(float p_radians);
	void yaw(float p_radians);
	void roll(float p_radians);

	glm::vec3 localForward() const;
	glm::vec3 localUp() const;
	glm::vec3 localRight() const;
	glm::vec3 globalForward() const;
	glm::vec3 globalUp() const;
	glm::vec3 globalRight() const;

	glm::mat4 projection() const;
	glm::mat4 view() const;
	glm::mat4 model() const;
	
	
	glm::quat m_orientation;
	
	float m_speed = 10.0f, m_sensitivity = 0.015f;
	float m_fov, m_aspectRatio, m_nearPlane, m_farPlane;
};