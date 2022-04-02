#pragma once
#include "Vec3.hpp"
class Ray {
public:
	//attributes
	Vec3 origin;
	Vec3 dir;
	Vec3 attenuation;

	__host__ __device__ Ray(Vec3 o, Vec3 d) {
		dir = d;
		origin = o;
	}

	//get ray position from distance
	__device__ Vec3 at(float distance) {
		return origin + (dir * distance);
	}	
};
