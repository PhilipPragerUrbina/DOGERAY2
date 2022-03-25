#pragma once
#include "world.hpp"
#include <chrono>
#include <iostream>
//comment and uncomment this to print render time to console. Normally disabled for max performance(les reialble IMGUI counter can be used instead),but is usful for testing
//#define recordspeed

//the actual ray tarce kernel. It cannot be a member function
__global__ void raytracekernel(uint8_t* image,World scene) {
	//get indexes 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = (y * scene.settings.w + x) * 3;
	
	Noise noise(x,y);
	
	//create randomized uv coords of image
	float u = (x + noise.rand()) / (scene.settings.w - 1);
	float v = (y + noise.rand())/ (scene.settings.h - 1);
	
	//gnerate ray
	Ray currentray = scene.settings.cam.getray(u, v, noise.unitdisk());
	//trace ray
	Vec3 color = scene.color(currentray,noise);

	//set image color if first sample(mult by 255 and clamp since color are currently in randge of 0-1)
	if (scene.settings.samples <= 0) {
		image[w    ] = fmin(color[0] * 255, 255.0f);
		image[w + 1] = fmin(color[1] * 255, 255.0f);
		image[w + 2] = fmin(color[2] * 255, 255.0f);
	}
	else {
		//if not first sample, contenusly update average of output
		//we add to the average rather than calculating it in order to avoid having divide on the host side
		image[w    ] = (scene.settings.samples * image[w    ] + fmin(color[0] * 255, 255.0f)) / (scene.settings.samples + 1);
		image[w + 1] = (scene.settings.samples * image[w + 1] + fmin(color[1] * 255, 255.0f)) / (scene.settings.samples + 1);
		image[w + 2] = (scene.settings.samples * image[w + 2] + fmin(color[2] * 255,255.0f)) / (scene.settings.samples + 1);
	}
}

//class for the actual CUDA ray tracer
class Tracekernel {
public:
	//status for error handling
	cudaError_t status;
	//threads per block. 
	dim3 threadsPerBlock;

	//constructor allocates memory and does setup
	Tracekernel(config settings, bvhnode* geometry, Mat* materials) {
		//for texture cleanup later
		nummaterials = settings.matsize;
		materialstoclean = materials;

		//setup device
		cudaGetDevice(&device);
		threadsPerBlock = dim3(8, 8);
		numBlocks = dim3(settings.w / threadsPerBlock.x, settings.h / threadsPerBlock.y);

		//allocate memory
		imagesize = settings.h * settings.w * 3 * sizeof(uint8_t);
		status = cudaMalloc((void**)&device_image, imagesize);
		if (status != cudaSuccess) { std::cerr << "error creating output image on device \n"; return; }

		size_t geosize = sizeof(bvhnode) * settings.bvhsize ;
		status = cudaMalloc((void**)&device_geometry, geosize);
		if (status != cudaSuccess) { std::cerr << "error allocating geometry on device \n"; return; }

		size_t matsize = sizeof(Mat) * settings.matsize;
		status = cudaMalloc((void**)&device_materials, matsize);
		if (status != cudaSuccess) { std::cerr << "error allocating materials on device \n"; return; }
		std::cout << "GPU memory succesfully allocated \n";

		//copy over normal geometry
		status = cudaMemcpy(device_geometry, geometry, geosize, cudaMemcpyHostToDevice); 
		if (status != cudaSuccess) { std::cerr << "error copying geometry on device \n"; return; }

		status = cudaMemcpy(device_materials, materials, matsize, cudaMemcpyHostToDevice);
		if (status != cudaSuccess) { std::cerr << "error copying materials on device \n"; return; }
		std::cout << "GPU memory succesfully copied over \n";
	}
	//render out an image
	void render(uint8_t* image, config settings) {
		#ifdef recordspeed
		begin = std::chrono::steady_clock::now();
		#endif

		//create scene object
		World scene(device_geometry, settings, device_materials);

		//run kenrel
		raytracekernel << <numBlocks, threadsPerBlock >> > (device_image, scene);

		//copy to host
		cudaMemcpy(image, device_image, imagesize, cudaMemcpyDeviceToHost);

		#ifdef recordspeed
		end = std::chrono::steady_clock::now();
		rendertime();
		#endif
	}
	void cleantextures() {
		for (int i = 0; i < nummaterials; i++) {
			//TODO add rest of textures here
			materialstoclean[i].colortexture.destroy();
			//materialstoclean[i].normaltexture.destroy();
			materialstoclean[i].roughtexture.destroy();
			materialstoclean[i].emmisiontexture.destroy();
		}
	}

	~Tracekernel() {
		//free memory
		cudaFree(device_image);
		cudaFree(device_geometry);
		cudaFree(device_materials);
	}

	//check for errors afetr kenrel launch. Not doing very time for performance.
	void errorcheck() {
		status = cudaGetLastError();
		if (status != cudaSuccess){std::cerr << "kernel launch error: " << cudaGetErrorString(status) << "\n";}
	}

	//display render time
	void rendertime() {
		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
	}
	void resize(config settings) {
		cudaFree(device_image);
		threadsPerBlock = dim3(8, 8);
		numBlocks = dim3(settings.w / threadsPerBlock.x, settings.h / threadsPerBlock.y);
		imagesize = settings.h * settings.w * 3 * sizeof(uint8_t);
		status = cudaMalloc((void**)&device_image, imagesize);
		if (status != cudaSuccess) { std::cerr << "error editing output image on device \n"; return; }
	}

private:

	//number of blocks,calculated at runtime
	dim3 numBlocks;

	//cuda device number
	int device = -1;

	//image data
	size_t imagesize;
	uint8_t* device_image = 0;

	//geometry data
	bvhnode* device_geometry = 0;
	Mat* device_materials = 0;

	//for profiling
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;

	//for cleanup of texture
	int nummaterials;
	Mat* materialstoclean;
};
