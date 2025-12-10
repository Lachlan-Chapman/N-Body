#version 460 core

layout(location = 0) in vec3 d_worldPosition;

out VS_OUT {
	vec3 d_worldPosition;
} vs_out; //passing out a world position

void main() {
	vs_out.d_worldPosition = d_worldPosition;
}