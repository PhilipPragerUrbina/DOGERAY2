#pragma once
#include"texture.hpp"
#include <curand_kernel.h>
//class for handling randomness on device
class Noise {
public:
    __device__ Noise(int x, int y) {
        //set up curand withs seed
        curand_init((unsigned long long)clock() + (x + y * blockDim.x * gridDim.x), 0, 0, &seed);
    }
    __device__ Vec3 unitsphere() {
        //random in unit sphere
        Vec3 p = Vec3(curand_normal_double(&seed), curand_normal_double(&seed), curand_normal_double(&seed));
        return p / p.length();
    }
    __device__ Vec3 unitdisk() {
            auto p = Vec3(curand_normal_double(&seed)*2-1, curand_normal_double(&seed)*2-1, 0);
            return p / p.length();
    }
    //random float
    __device__ float rand() {
        return curand_uniform_double(&seed);
    }
    /*  while (true) {
        Vec3 p = Vec3((curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f);
        if (pow(p.length(), 2.0f) >= 1) continue;
        return p;
    }*/
private:
    //store state
    curandState seed;
};









//I orignally wanted to use runtime polymorphism or function pointers. This proved to be very diffucult to copy from host to device without significant complexity.
//matirial  class
//decided to combine prev shaders into unversal PBR material to work better with GLTF and to avoid polymorphism problems
class Mat {
public:
    //mat properties. Uses PBR inputs. 
    Vec3 colorfactor = Vec3(0.5,0.5,0.5);
    Texture colortexture;

    float metalfactor = 0.5;

    float roughfactor = 0.5;
    Texture roughtexture;

    Texture normaltexture;

    //id for material creation in host(prevent duplicates)
    int id = 0;
    Mat(int ident, Vec3 col, float metal, float rough) {
        id = ident;
        colorfactor = col;
        metalfactor = metal;
        roughfactor = rough;
    }


    __device__  void interact(Ray* ray, Vec3 texcoords , Vec3 hitpoint, Vec3 normal, Noise noise) {
     
        if (normaltexture.exists) {
          //  normal = normal * (normaltexture.get(texcoords) * 2.0 - 1.0).normalized();
        }

        float rough = roughfactor;
        if (roughtexture.exists) {
            rough = rough * roughtexture.get(texcoords)[1];
        }

        //calculate direction
        Vec3 reflected = ray->dir.normalized().reflected(normal);
        ray->dir = reflected + Vec3(rough) * noise.unitsphere();

        Vec3 color = colorfactor;
        if (colortexture.exists) {
            color = color * colortexture.get(texcoords);
        }
       
        //update ray
        ray->origin = hitpoint;
        ray->attenuation = ray->attenuation * color;

    };



};


/*
//diffuse amt
class Diffuse {
    __device__ virtual void interact(Ray* ray, Vec3 hitpoint, Vec3 normal,curandState* seed) {
        //calc direction
        Vec3 target = hitpoint + normal + randomvec3insphere(seed);
        ray->dir = (target - hitpoint).normalized();
        //update ray
        ray->origin = hitpoint;
        ray->attenuation = ray->attenuation * color;
    }
};

//reflective mat
//atribute 1 is rougnesss
class Metal : public Mat {
    __device__ virtual void interact(Ray* ray, Vec3 hitpoint, Vec3 normal, curandState* seed) {
        //calculate direction
        Vec3 reflected = ray->dir.normalized().reflected(normal);
        ray->dir = reflected + Vec3(attribute1) * randomvec3insphere(seed);
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

