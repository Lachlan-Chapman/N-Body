#version 460 core

layout(location = 0) in vec2 a_pos; //x, y in screen space (starting btm left)
layout(location = 1) in vec2 a_uv; //texture UVs

out vec2 v_uv;

uniform vec2 u_screenSize; //resolution, eg (3840, 2160)

void main()
{
	v_uv = a_uv;

	//Convert screen pixels → NDC (see "shaders/coordinateSpacesExplained.txt" for more info)
	float ndc_x = (a_pos.x / u_screenSize.x) * 2.0 - 1.0;
	float ndc_y = 1.0 - (a_pos.y / u_screenSize.y) * 2.0 - 0.1;

	gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
}
