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
    //0 menas no texture
    vec3 colorfactor = vec3(0.5,0.5,0.5);
    cudaTextureObject_t colortexture = 0;

    float metalfactor = 0.5;

    float roughfactor = 0.5;
    cudaTextureObject_t roughtexture = 0;

    cudaTextureObject_t normaltexture = 0;

    //id for material creation in host(prevent duplicates)
    int id = 0;
    Mat(int ident, vec3 col, float metal, float rough) {
        id = ident;
        colorfactor = col;
        metalfactor = metal;
        roughfactor = rough;
    }

    __device__  void interact(Ray* ray, vec3 texcoords , vec3 hitpoint, vec3 normal, curandState* seed) {
     
        if (normaltexture != 0) {
            uchar4 tex = tex2D<uchar4>(normaltexture, texcoords[0], texcoords[1]);
            normal = normal + vec3(float(tex.x) / 127.5 - 1, float(tex.y) / 127.5 - 1, float(tex.z) / 127.5 - 1);
        }

        float rough = roughfactor;
        if (roughtexture != 0) {
            uchar4 tex = tex2D<uchar4>(roughtexture, texcoords[0], texcoords[1]);
            rough = rough * vec3(float(tex.x) / 255, float(tex.y) / 255, float(tex.z) / 255)[1];
        }
        //calculate direction
        vec3 reflected = ray->dir.normalized().reflected(normal);
        ray->dir = reflected + vec3(rough) * randomvec3insphere(seed);


        //update ray
        ray->origin = hitpoint;
        //ray->attenuation = ray->attenuation * checker(texcoords, vec3(0.8,0.5, 0.5), vec3(0.5, 0.8, 0.5));
        vec3 color = colorfactor;
        if (colortexture != 0) {
            uchar4 tex = tex2D<uchar4>(colortexture, texcoords[0], texcoords[1]);
            color = color * vec3(float(tex.x) / 255, float(tex.y) / 255, float(tex.z) / 255);
        }
       
      
        ray->attenuation = ray->attenuation * color;

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


//helpr function for allocating image textrue. Not in class since it sues cuda texture objevt
//TODO move soemwhere else. Add freeing of texture
void loadtexture(unsigned char* data, int width, int height, int channels, int bits, cudaTextureObject_t* texture) {

    cudaError_t status;
    size_t s = bits/8 * channels;
    size_t  sizee = width * height * s;


    // Allocate array and copy image data

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(bits, bits, bits, bits, cudaChannelFormatKindUnsigned);

    if (channels == 1) {
        channelDesc =
            cudaCreateChannelDesc(bits, 0, 0, 0, cudaChannelFormatKindUnsigned);
    }
    else if (channels == 2) {
        channelDesc =
            cudaCreateChannelDesc(bits, bits, 0, 0, cudaChannelFormatKindUnsigned);
    }
    else if (channels == 3) {
        channelDesc =
            cudaCreateChannelDesc(bits, bits, bits, 0, cudaChannelFormatKindUnsigned);
    }

    cudaArray* cuArray;
    status = cudaMallocArray(&cuArray,
        &channelDesc,
        width,
        height);
    if (status != cudaSuccess) { std::cerr << "error allocating textures on device \n"; return; }
    size_t spitch = width * s;

    status = cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * s,
        height, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { std::cerr << "error copying textures on device \n"; return; }
    
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;

    //  texDescr.readMode = cudaReadModeElementType;

    status =cudaCreateTextureObject(texture, &texRes, &texDescr, NULL);
    if (status != cudaSuccess) { std::cerr << "error creating textures on device \n"; return; }
    std::cout << bits << " bit " << channels <<" channel texture loaded succesfully \n";



}