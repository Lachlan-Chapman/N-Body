#version 460 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in VS_OUT {
	vec3 d_worldPosition;
} gs_in[];

out GS_OUT {
	vec2 dummy; // placeholder until you add UV etc
} gs_out;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_cameraPosition;
uniform float u_scale;


void main() {
	vec3 point = gs_in[0].d_worldPosition;
	vec3 camera = u_cameraPosition;

	vec3 dir = point - camera; //vec to the point from the cam

	dir.y = 0.0; //dont use the y dir since the billboard faces up in world space
	dir = normalize(dir);

	float scale = u_scale;
	vec3 up = vec3(0.0, 1.0, 0.0) * scale;
	vec3 right = normalize(cross(up, dir)) * scale; //so perpendicular to our vec to the point with the world up gives us the right
	
	vec3 top_left = point - right + up; //simply go left and up
	vec3 top_right = point + right + up;
	vec3 bottom_left = point - right - up;
	vec3 bottom_right = point + right - up;

	//as a triangle strip being TL, TR, BL, BR

	//convert world to clip space
	gl_Position = u_projection * u_view * vec4(top_left, 1.0);
	EmitVertex(); //tells the gpu im done creating a vertex and to send it on throught to the pipeline (into our triangle strip obj)
	gl_Position = u_projection * u_view * vec4(top_right, 1.0);
	EmitVertex();
	gl_Position = u_projection * u_view * vec4(bottom_left, 1.0);
	EmitVertex();
	gl_Position = u_projection * u_view * vec4(bottom_right, 1.0);
	EmitVertex();

	EndPrimitive(); //this triangle strip is done
}