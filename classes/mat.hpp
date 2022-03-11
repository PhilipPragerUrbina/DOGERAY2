#pragma once
#include "config.hpp"
//gpu random sphere vec3 function for matirals
//moved to other class later
/*
//try to remove while loop
__device__ vec3 randomvec3insphere(curandState* seed) {
    while (true) {
        vec3 p = vec3((curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f);
        if (pow(p.length(), 2.0f) >= 1) continue;
        return p;
    }
}*/
__device__ vec3 randomvec3insphere(curandState* seed) {
    vec3 p = vec3(curand_normal_double(seed), curand_normal_double(seed), curand_normal_double(seed));
    return p / p.length();
}
__device__ vec3 checker(vec3 uv, vec3 col1, vec3 col2) {
    float u2 = floor(uv[0] * 10);
    float v2 = floor(uv[1] * 10);
    float yes = u2 + v2;
    if (fmod(yes, (float)2) == 0)
        return col1;
    else
        return col2;
}
//I orignally wanted to use runtime polymorphism or function pointers. This proved to be very diffucult to copy from host to device without significant complexity.
//matirial  class
//decided to combine prev shaders into unversal PBR material to work better with GLTF and to avoid polymorphism problems
class Mat {
public:
    //mat properties. Uses PBR inputs. 
    vec3 colorfactor = vec3(0.5,0.5,0.5);
    float metalfactor = 0.5;
    float roughfactor = 0.5;
    //id for material creation in host(prevent duplicates)
    int id = 0;
    Mat(int ident, vec3 col, float metal, float rough) {
        id = ident;
        colorfactor = col;
        metalfactor = metal;
        roughfactor = rough;
    }

    __device__  void interact(Ray* ray, vec3 texcoords , vec3 hitpoint, vec3 normal, curandState* seed) {
        //calc direction
        vec3 target = hitpoint + normal + randomvec3insphere(seed);
        ray->dir = (target - hitpoint).normalized();
        //update ray
        ray->origin = hitpoint;
        ray->attenuation = ray->attenuation * checker(texcoords, vec3(0.8,0.5, 0.5), vec3(0.5, 0.8, 0.5));
    };
};


/*
//diffuse amt
class Diffuse {
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
       else if (b[g].mat == 5) {
                //glossy mat
                float rand = randy(state);

                if (rand > 0.8) {
                    float3 reflected = reflect(getNormalizedVec(raydir), N);
                    cur_attenuation = cur_attenuation * ocolor;
                    rayo = hitpoint;

                    raydir = reflected + make3(rough) * random_in_unit_sphere(state);

                }
                else {
                    float3 target = hitpoint + N;

                    target = target + random_in_unit_sphere(state);


                    cur_attenuation = cur_attenuation * ocolor;

                    rayo = hitpoint;
                    raydir = getNormalizedVec(target - hitpoint);

                }




            }
};*/