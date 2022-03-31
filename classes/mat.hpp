#pragma once
#include"config.hpp"
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
        if (emmisivefactor[0] != 0 || emmisivefactor[1] != 0 || emmisivefactor[2] != 0) {
            doesemit = true;
        }
    }
    void destroy() {
        //TODO add rest of textures here
          colortexture.destroy();
        //materialstoclean[i].normaltexture.destroy();
       roughtexture.destroy();
       emmisiontexture.destroy();
    }


    __device__  bool interact(Ray* ray, Vec3 texcoords , Vec3 hitpoint, Vec3 normal, Noise noise) {
     //update normals
      //  if (normaltexture.exists) {
          //  normal = normal * (normaltexture.get(texcoords) * 2.0 - 1.0).normalized();
      //  }
        Vec3 color = colorfactor;
        if (colortexture.exists) {
            color = color * colortexture.get(texcoords);
        }

        //has emmsive propertyies
        if (doesemit) {
            //get emmisive values
            Vec3 emmision = emmisivefactor;
            if (emmisiontexture.exists) {
                emmision = emmision * emmisiontexture.get(texcoords);
            }
            //if this specific part of triangle is emmsive
            if (emmision[0] > 0.1 || emmision[1] > 0.1 || emmision[2] > 0.1) {
                //retrun light color
                ray->attenuation = ray->attenuation * emmision;
                return true;
            }
           
        }
        //get textures
        float rough = roughfactor;
        float metal = metalfactor;
        if (roughtexture.exists) {
            rough = rough * roughtexture.get(texcoords)[1];
           metal = metal * roughtexture.get(texcoords)[2];

        }
     
        //calulate direction
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
        //dialectric(plastic not glass)
        else {
            //TODO debug and adjust ior for plastic mixing
            const float ior = 2.5;
            float cosine = min((-(ray->dir.normalized())).dot(normal), 1.0f);
             //fresnel
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
         

        //update ray
        ray->origin = hitpoint;
   
        
        return false;
    };
    private:
        //schlick approximation frm ray tracing ij one weekend
        __device__ float reflectance(float cosine, float ref_idx) {
            float r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }


};
