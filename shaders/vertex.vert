//useful for simple point rendering in 3d
#version 460 core

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform vec3 u_cameraPosition;
layout(location = 0) in vec3 d_position;

void main() {

	//world projection
	vec4 world_position = u_model * vec4(d_position, 1.0);
	gl_Position = u_mvp * world_position;

	//world space distance from camera to vertex
	float distance = length(world_position.xyz - u_cameraPosition);
	gl_PointSize = 40.0 / distance;
}