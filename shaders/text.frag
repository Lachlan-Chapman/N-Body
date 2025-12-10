#version 460 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_atlas;

void main() {
	float a = texture(u_atlas, v_uv).a; //sample alpha
	fragColor = vec4(a, a, a, 1.0); //visualize alpha
}
