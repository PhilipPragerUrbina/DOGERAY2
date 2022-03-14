#pragma once
#include"texture.hpp"
//gpu random sphere Vec3 function for matirals
//moved to other class later
/*
//try to remove while loop
__device__ Vec3 randomvec3insphere(curandState* seed) {
    while (true) {
        Vec3 p = Vec3((curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f, (curand_uniform_double(seed) * 2.0f) - 1.0f);
        if (pow(p.length(), 2.0f) >= 1) continue;
        return p;
    }
}*/
__device__ Vec3 randomvec3insphere(curandState* seed) {
    Vec3 p = Vec3(curand_normal_double(seed), curand_normal_double(seed), curand_normal_double(seed));
    return p / p.length();
}
__device__ Vec3 checker(Vec3 uv, Vec3 col1, Vec3 col2) {
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


    __device__  void interact(Ray* ray, Vec3 texcoords , Vec3 hitpoint, Vec3 normal, curandState* seed) {
     
        if (normaltexture.exists) {
            normal = normal + (normaltexture.get(texcoords) * 2.0f - 1.0f);
        }

        float rough = roughfactor;
        if (roughtexture.exists) {
            rough = rough * roughtexture.get(texcoords)[1];
        }

        //calculate direction
        Vec3 reflected = ray->dir.normalized().reflected(normal);
        ray->dir = reflected + Vec3(rough) * randomvec3insphere(seed);

       
        //ray->attenuation = ray->attenuation * checker(texcoords, Vec3(0.8,0.5, 0.5), Vec3(0.5, 0.8, 0.5));

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


