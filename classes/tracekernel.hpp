#pragma once
#include "config.hpp"
#include <iostream>

//the actual ray tarce kernel. It cannot be a member function
__global__ void raytracekernel(uint8_t* image, config settings) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = (y * settings.w + x) * 3;
	float u = float(x) / (settings.w - 1);
	float v = float(y) / (settings.h - 1);
	ray currentray = settings.cam.getray(u, v);
	vec3 color = currentray.raycolor();
	image[w] = color[0]*255;
	image[w + 1] = color[1]*255;
	image[w + 2] = color[2]*255;
}

//class for the actual CUDA ray tracer
class tracekernel {
public:
	//status for error handling
	cudaError_t status;
	//constructor allocates memory and does setup
	tracekernel(config settings) {
		//setup device
		cudaGetDevice(&device);
		threadsPerBlock = dim3(8, 8);
		numBlocks = dim3(settings.w / threadsPerBlock.x, settings.h / threadsPerBlock.y);
		//allocate memory
		imagesize = settings.h * settings.w * 3 * sizeof(uint8_t);
		status = cudaMalloc(&device_image, imagesize);
		if (status != cudaSuccess) { std::cerr << "error creating output image on device \n"; return; }
		

	}
	//render out an image
	void render(uint8_t* image, config settings) {
		//run kenrel
		raytracekernel << <numBlocks, threadsPerBlock >> > (device_image, settings);
		//copy to host
		cudaMemcpy(image, device_image, imagesize, cudaMemcpyDeviceToHost);
	}
	~tracekernel() {
		//free memory
		cudaFree(device_image);
	}
	//check for errors afetr kenrel launch. Not doing very time for performance.
	void errorcheck() {
		status = cudaGetLastError();
		if (status != cudaSuccess){std::cerr << "kernel launch error: " << cudaGetErrorString(status) << "\n";}
	}
private:
	//threads per block. 
	dim3 threadsPerBlock;
	dim3 numBlocks;
	//cuda devic number
	int device = -1;
	//image data
	size_t imagesize;
	uint8_t* device_image = 0;

};
