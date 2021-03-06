#pragma once
#include"config.hpp"
#include <curand_kernel.h>

//class for handling randomness on device
class Noise {
public:
    __device__ Noise(int x, int y) {
        //set up curand with seed
        curand_init((unsigned long long)clock() + (x + y * blockDim.x * gridDim.x), 0, 0, &seed);
    }
    //random in unit sphere
    __device__ Vec3 unitsphere() {
        Vec3 p = Vec3(curand_normal_double(&seed), curand_normal_double(&seed), curand_normal_double(&seed));
        return p / p.length();
    }
    //random in unit disk
    __device__ Vec3 unitdisk() {
            auto p = Vec3(curand_normal_double(&seed)*2-1, curand_normal_double(&seed)*2-1, 0);
            return p / p.length();
    }
    //random float
    __device__ float rand() {
        return curand_uniform_double(&seed);
    }
private:
    //store state
    curandState seed;
};



//I originally wanted to use runtime polymorphism or function pointers. This proved to be very difficult to copy from host to device without significant complexity.
class Mat {
public:
    //mat properties. Uses PBR inputs. 
    Vec3 colorfactor{0.5,0.5,0.5};
    Texture colortexture;

    float metalfactor = 0.5;
    float roughfactor = 0.5;
    Texture roughtexture;

    Vec3 emmisivefactor{0,0,0};
    Texture emmisiontexture;
    bool doesemit = false;
  //  Texture normaltexture;

    //id for material creation in host(prevent duplicates)
    int id = 0;

    Mat(int ident, Vec3 col, float metal, float rough, Vec3 emit) {
        id = ident;
        colorfactor = col;
        metalfactor = metal;
        roughfactor = rough;
        emmisivefactor = emit;
        //check if emmisive
        if (emmisivefactor[0] != 0 || emmisivefactor[1] != 0 || emmisivefactor[2] != 0) {
            doesemit = true;
        }
    }
    //clean up textures
    void destroy() {
          colortexture.destroy();
        //normaltexture.destroy();
       roughtexture.destroy();
       emmisiontexture.destroy();
    }
    //matriarchal hit function
    __device__  bool interact(Ray* ray, Vec3 texcoords , Vec3 hitpoint, Vec3 normal, Noise noise) {
     //update normals
      //  if (normaltexture.exists) {
          //  normal = normal * (normaltexture.get(texcoords) * 2.0 - 1.0).normalized();
      //  }
        //get color
        Vec3 color = colorfactor;
        if (colortexture.exists) {
            color = color * colortexture.get(texcoords);
        }

        //get emmisive
        if (doesemit) {
            Vec3 emmision = emmisivefactor;
            if (emmisiontexture.exists) {
                emmision = emmision * emmisiontexture.get(texcoords);
            }
            //if this specific part of triangle is emisive
            if (emmision[0] > 0.1 || emmision[1] > 0.1 || emmision[2] > 0.1) {
                //return light color
                ray->attenuation = ray->attenuation * emmision;
                //return true if emmisive to tell world class to stop bouncing
                return true;
            }
           
        }

        //get metallic roughness
        float rough = roughfactor;
        float metal = metalfactor;
        if (roughtexture.exists) {
            rough = rough * roughtexture.get(texcoords)[1];
           metal = metal * roughtexture.get(texcoords)[2];

        }
     
        //calculate direction
        //metal (No mixing since most of the time metal is metal)
        if (metal > 0.5) {
            ray->dir = ray->dir.normalized().reflected(normal) + Vec3(rough) * noise.unitsphere();
            ray->attenuation = ray->attenuation * color;

        }
        //diffuse
        else  if (rough > 0.7) {
            Vec3 target = hitpoint + normal + noise.unitsphere();
            ray->dir = (target - hitpoint).normalized();
            ray->attenuation = ray->attenuation * color;
        }
        //dialectic(plastic not glass)
        else {
            const float ior = 2.5;
            float cosine = min((-(ray->dir.normalized())).dot(normal), 1.0f);
             //Fresnel
            if (reflectance(cosine, 1.0f / ior) > noise.rand()) {
                //specular(metal without color)
                ray->dir = ray->dir.normalized().reflected(normal) + Vec3(rough) * noise.unitsphere();
            }
            else {
                //diffuse
                Vec3 target = hitpoint + normal + noise.unitsphere();
                ray->dir = (target - hitpoint).normalized();
                ray->attenuation = ray->attenuation * color;
            }
        }     
        //update ray origin
        ray->origin = hitpoint;
        return false;
    };
    private:
        //Schick approximation from ray tracing in one weekend
        __device__ float reflectance(float cosine, float ref_idx) {
            float r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }
};
