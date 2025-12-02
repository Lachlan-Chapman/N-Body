#version 460 core

layout(location = 0) in vec3 d_position;

void main() {
	gl_Position = vec4(d_position, 1.0);
	gl_PointSize = 10.0; //big as hell so you can’t miss it
}