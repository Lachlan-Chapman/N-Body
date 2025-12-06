#version 460 core

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;

uniform vec3 u_cameraPosition;

uniform float u_scale;

layout(location = 0) in vec3 d_vertexPosition;
layout(location = 1) in vec3 d_instancePosition;


void main()
{
	//adjust by the instance pos
	vec3 vert_pos = d_vertexPosition * u_scale;
	vec3 world_position = vert_pos + d_instancePosition;

	//make sure to adjust so our camera can see it
	gl_Position = u_projection * u_view * vec4(world_position, 1.0);
}
