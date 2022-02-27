#pragma once
#include "vec3.hpp"
//vertex struct.
struct vertex {
	//vertex position
	vec3 pos;
	//vertex normal
	vec3 norm;
	//tex coord. ignore z for now
	vec3 tex;
};
//boudning box struct
struct boundingbox {
	vec3 max;
	vec3 min;
};

//tiangle class
class tri {
public:
	boundingbox box;
	vertex verts[3];
	boundingbox getboundingbox() {
		//return minimum and maximumvalues from traingle for bvh construction
		boundingbox out;
		out.min = vec3(fmin(verts[0].pos[0], fmin(verts[1].pos[0], verts[2].pos[0])), fmin(verts[0].pos[1], fmin(verts[1].pos[1], verts[2].pos[1])), fmin(verts[0].pos[2], fmin(verts[1].pos[2], verts[2].pos[2])));
		out.max = vec3(fmax(verts[0].pos[0], fmax(verts[1].pos[0], verts[2].pos[0])), fmax(verts[0].pos[1], fmax(verts[1].pos[1], verts[2].pos[1])), fmax(verts[0].pos[2], fmax(verts[1].pos[2], verts[2].pos[2])));
		//save box for later
		box = out;
		return out;
	}
	//todo add intersection 


};