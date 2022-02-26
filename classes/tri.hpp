#pragma once
#include "vec3.hpp"
//vertex class
struct vertex {
	//vertex position
	vec3 pos;
	//vertex normal
	vec3 norm;
	//tex coord. ignore z for now
	vec3 tex;
};
//tiangle class
class tri {
public:

	vertex verts[3];

	//todo add intersection and bounding box functions


};