#pragma once
#include "vec3.hpp"
//ray class
class Ray {
public:
	//attributes
	vec3 origin;
	vec3 dir;
	vec3 attenuation;

	__host__ __device__ Ray(vec3 o, vec3 d) {
		dir = d;
		origin = o;
	}

	//get ray postion from distance
	__device__ vec3 at(float distance) {
		return origin + (vec3(distance) * dir);
	}	
};
