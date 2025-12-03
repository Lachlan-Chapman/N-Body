#pragma once
#include "observer/camera.hpp"

class cameraFlight : public camera {
public:
	cameraFlight() = delete;
	cameraFlight(
		glm::vec3 p_position,
		float p_fov,
		float p_aspectRatio,
		float p_nearPlane,
		float p_farPlane
	);

	void move(
		bool p_w,
		bool p_a,
		bool p_s,
		bool p_d,
		bool p_space,
		bool p_leftAlt,
		float p_deltaTime
	) override;

	void rotate(
		float p_mouseX,
		float p_mouseY,
		bool p_q,
		bool p_e,
		float p_deltaTime
	) override;

protected:
	glm::quat yaw(float p_radians) const override;
};