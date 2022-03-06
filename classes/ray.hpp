#pragma once
#include "vec3.hpp"
//ray class
class ray {
public:
	vec3 origin;
	vec3 dir;
	vec3 attenuation;
	__host__ __device__ ray(vec3 o, vec3 d) {
		dir = d;
		origin = o;
	}
	//get ray postion distance
	__device__ vec3 at(float distance) {
		return origin + (vec3(distance) * dir);
	}
	

};
