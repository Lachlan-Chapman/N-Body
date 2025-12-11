#version 460 core

layout(location = 0) in vec3 a_quadVertex;
layout(location = 1) in vec3 a_instancePos;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_cameraPosition;
uniform float u_scale;

void main() {
	vec3 direction = a_instancePos - u_cameraPosition; //vec to the point from the cam
	direction.y = 0.0; //dont use the y direction since the billboard faces up in world space
	direction = normalize(direction);

	vec3 up = vec3(0.0, 1.0, 0.0);
	vec3 right = normalize(cross(up, direction)); //so perpendicular to our vec to the point with the world up gives us the right

	vec3 worldPos = vec3(
		a_instancePos +
		(right * a_quadVertex.x * u_scale) +
		(up * a_quadVertex.z * u_scale)
	);
	gl_Position = u_projection * u_view * vec4(worldPos, 1);
}