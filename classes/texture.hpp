#pragma once
#include"vec3.hpp"
//class for keeping track of textures
class Texture {
public:
    //error checking
    cudaError_t status;

    //gpu data
    cudaArray* device_data;

    //actual texture object
    cudaTextureObject_t texture;

    //Is this texture not empty?
    bool exists = false;

    //get Vec3 from texture
    __device__ Vec3 get(Vec3 uv) {
        //TODO if other textures appear. Make sure to check what type it should access. Uchar4, Uchar3, Short4, short3, etc
        uchar4 texdata = tex2D<uchar4>(texture, uv[0], uv[1]);
        return Vec3(float(texdata.x) / 255.0f, float(texdata.y) / 255.0f, float(texdata.z) / 255.0f);
    }

    //you may be wondering why this isnt a constructor and destructor the reason is that this is allocating memory on the GPU so it only should be called once during the lifetime of the program
    void create(unsigned char* data, int width, int height, int numchannels, int bits) {
        //calc sizes
        size_t typesize = bits / 8 * numchannels;
        size_t pitch = width * typesize; //widthof data
        //create channel description
        //most have 4 channels
        cudaChannelFormatDesc channels = cudaCreateChannelDesc(bits, bits, bits, bits, cudaChannelFormatKindUnsigned);

        if (numchannels == 1) {
            //one channel
            channels = cudaCreateChannelDesc(bits, 0, 0, 0, cudaChannelFormatKindUnsigned);
        }
        else if (numchannels == 2) {
            //two channels
            channels = cudaCreateChannelDesc(bits, bits, 0, 0, cudaChannelFormatKindUnsigned);
        }
        else if (numchannels == 3) {
            //three channels
            channels = cudaCreateChannelDesc(bits, bits, bits, 0, cudaChannelFormatKindUnsigned);
        }

        //allocate data
        status = cudaMallocArray(&device_data, &channels, width, height);
        if (status != cudaSuccess) { std::cerr << "error allocating texture on device \n"; return; }
        
        //copy data
        status = cudaMemcpy2DToArray(device_data, 0, 0, data, pitch, width * typesize, height, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) { std::cerr << "error copying texture on device \n"; return; }

        //set texture to use cuda array
        cudaResourceDesc recource;
        memset(&recource, 0, sizeof(cudaResourceDesc));
        recource.resType = cudaResourceTypeArray;
        recource.res.array.array = device_data;

        //texture settings
        //TODO tweak settings. Potentially change them based on GLTF sampler
        cudaTextureDesc texturesettings;
        memset(&texturesettings, 0, sizeof(cudaTextureDesc));
        texturesettings.normalizedCoords = true;
        texturesettings.filterMode = cudaFilterModePoint;
        texturesettings.addressMode[0] = cudaAddressModeWrap;
        texturesettings.addressMode[1] = cudaAddressModeWrap;

        //finally create texture object
        status = cudaCreateTextureObject(&texture, &recource, &texturesettings, NULL);
        if (status != cudaSuccess) { std::cerr << "error creating texture on device \n"; return; }
        std::cout << bits << " bit " << numchannels << " channel texture loaded succesfully \n";
        //texture now exists
        exists = true;
    }
    //clean up texture
    void destroy() {
        //check if texture exists first
        if (exists) {
            cudaDestroyTextureObject(texture);
            cudaFreeArray(device_data);
            exists = false;
            std::cout << "texture destroyed \n";
        }
    }

};
