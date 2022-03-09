#pragma once
#include "config.hpp"
//gpu random sphere vec3 function for matirals
//moved to other class later
__device__ vec3 randomvec3insphere(curandState* seed) {
    while (true) {
        vec3 p = vec3((curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f);
        if (pow(p.length(), 2.0f) >= 1) continue;
        return p;
    }
}

//matirial base class
//have differnt matirials using polymorphism
class Mat {
public:
    //mat properties
    vec3 color = vec3(0.5,0.5,0.5);
    float attribute1 = 0.5;

    __device__ virtual void interact(Ray* ray, vec3 hitpoint, vec3 normal, curandState* seed) = 0;
};

//diffuse amt
class Diffuse : public Mat {
    __device__ virtual void interact(Ray* ray, vec3 hitpoint, vec3 normal,curandState* seed) {
        //calc direction
        vec3 target = hitpoint + normal + randomvec3insphere(seed);
        ray->dir = (target - hitpoint).normalized();
        //update ray
        ray->origin = hitpoint;
        ray->attenuation = ray->attenuation * color;
    }
};

//reflective mat
//atribute 1 is rougnesss
class Metal : public Mat {
    __device__ virtual void interact(Ray* ray, vec3 hitpoint, vec3 normal, curandState* seed) {
        //calculate direction
        vec3 reflected = ray->dir.normalized().reflected(normal);
        ray->dir = reflected + vec3(attribute1) * randomvec3insphere(seed);
        //update ray
        ray->origin = hitpoint;
        ray->attenuation = ray->attenuation * color;
    }
};