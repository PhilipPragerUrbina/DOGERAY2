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
	__device__ vec3 raycolor() {
		vec3 unit_direction = dir.normalized();
		float t = 0.5 * (unit_direction[1] + 1.0);
		return vec3(1.0 - t) * vec3(1.0, 1.0, 1.0) + vec3(t) * vec3(0.5, 0.7, 1.0);
	}

};