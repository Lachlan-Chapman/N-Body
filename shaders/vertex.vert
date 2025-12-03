#version 460 core

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;

uniform vec3 u_cameraPosition;

layout(location = 0) in vec3 d_position;

void main()
{
	// Transform into world space
	vec4 world_position = u_model * vec4(d_position, 1.0);

	// Final transform to clip space
	gl_Position = u_projection * u_view * world_position;

	// Scale point size based on world-space distance from camera
	float distance = length(world_position.xyz - u_cameraPosition);
	gl_PointSize = 40.0 / distance;
}
