#pragma once
#include "vec3.hpp"
class ray {
public:
	vec3 origin;
	vec3 dir;

	__host__ __device__ ray(vec3 o, vec3 d) {
		dir = d;
		origin = o;
	}
	vec3 at(float time) {
		return origin + vec3(time) * dir;
	}
	

};
