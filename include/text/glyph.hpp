#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "math/vec.hpp"
namespace Text{
	struct glyph {
		float u0, v0, u1, v1;
		int d_width, d_height;
		int d_xOffset, d_yOffset;
		int d_xAdvance;
	};
	int font_base;
	std::unordered_map<char, Text::glyph> font_map; //easy O(1) lookup from char to rendering meta data
	bool buildGlyphMap(const std::string &p_path, vec2i p_atlasDimension) {
		std::ifstream atlas(p_path);
		if(!atlas.is_open()) {
			std::cerr << "Failed to open font file " << p_path << "\n";
			return false;
		}

		std::string line;
		while(std::getline(atlas, line)) {
			if(line.rfind("char ", 0) == 0) {
				int id, x, y, width, height, xoffset, yoffset, xadvance, page, channel;
				int matched = std::sscanf( //load vars from string as ints
					line.c_str(),
					"char id=%d x=%d y=%d width=%d height=%d xoffset=%d yoffset=%d xadvance=%d page=%d chnl=%d",
					&id, &x, &y, &width, &height, &xoffset, &yoffset, &xadvance, &page, &channel
				);
				if(matched == 10) { //all values were found
					Text::glyph g;
					g.d_width = width;
					g.d_height = height;
					g.d_xOffset = xoffset;
					g.d_yOffset = yoffset;
					g.d_xAdvance = xadvance;

					g.u0 = static_cast<float>(x) / p_atlasDimension.x;
					g.v0 = static_cast<float>(y) / p_atlasDimension.y;
					g.u1 = static_cast<float>(x + width) / p_atlasDimension.x;
					g.v1 = static_cast<float>(y + height) / p_atlasDimension.y;

					Text::font_map[static_cast<char>(id)] = g; //store into map using the ascii char as key
				}
			}
		}
		return true;
	}

	//p_position is xy coord with z being scale
	int buildTextVertices(const std::string &p_text, vec3f p_position, vec4f *p_buffer) {
		float cursor = 0.0f;
		int vertex_count = 0;
		int text_length = p_text.length();
		for(int char_id = 0; char_id < text_length; char_id++) {
			char c = p_text[char_id];
			if(Text::font_map.find(c) == Text::font_map.end()) { continue; } //character doesnt exist in our text atlas
			const Text::glyph &g = Text::font_map[c];
			float x_pos = p_position.x + (g.d_xOffset * p_position.z) + cursor;
			float y_pos = p_position.y - (g.d_yOffset * p_position.z);

			float w = g.d_width * p_position.z;
			float h = g.d_height * p_position.z;

			p_buffer[vertex_count++] = vec4f( //tl
				x_pos, y_pos,
				g.u0, g.v0
			);
			p_buffer[vertex_count++] = vec4f( //tr
				x_pos + w, y_pos,
				g.u1, g.v0
			);
			p_buffer[vertex_count++] = vec4f( //bl
				x_pos, y_pos + h,
				g.u0, g.v1
			);


			p_buffer[vertex_count++] = vec4f( //tr
				x_pos + w, y_pos,
				g.u1, g.v0
			);
			p_buffer[vertex_count++] = vec4f( //bl
				x_pos, y_pos + h,
				g.u0, g.v1
			);
			p_buffer[vertex_count++] = vec4f( //br
				x_pos + w, y_pos + h,
				g.u1, g.v1
			);

			cursor += g.d_xAdvance * p_position.z;
		}
		return vertex_count;
	}
}