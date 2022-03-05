#pragma once
#include "config.hpp"
#include "bvhtree.hpp"
#include <chrono>
#include <iostream>

//comment and uncomment this to print render time to console. Normally disabled for max performance(les reialble IMGUI counter can be used instead),but is usful for testing
//#define recordspeed
//seperate ray color fucntion to avoid circualr dependency
__device__ vec3 raycolor(ray curray, bvhnode* tree, config settings) {
	float dist = 100000;
	vec3 uv;
	tri triangle;
	//stackless bvh traversal from paper
	int box_index = 0;
	while (box_index != -1) {
		bvhnode currentnode = tree[box_index];
		bool hit = currentnode.testbox(curray);
		if (hit) {
			//bvh preview
			//color = color + vec3(settings.bvhstrength);
			box_index = currentnode.hitnode; // hit link
			if (currentnode.isleaf) {
				//box_index = -1;
				float tempdist;
				vec3 tempuv;
				//if hit
				if (currentnode.traingle.hit(curray, tempdist,tempuv)) {
					if (tempdist<dist) {
						dist = tempdist;
						triangle = currentnode.traingle;
						uv = tempuv;
					}
				}
			}
		}
		else {
			box_index = currentnode.missnode; // miss link
		}
	}

	vec3 color = vec3(0);
	if (dist < 10000) {
		//color = vec3(dist* settings.bvhstrength);
		color = (vec3(1 - uv[0] - uv[1]) * triangle.verts[0].norm )+ (vec3(uv[0]) * triangle.verts[1].norm) + (vec3(uv[1]) * triangle.verts[2].norm);
		//color = vec3(uv[0], uv[1], 1 - uv[0] - uv[1]);
		//color = triangle.verts[0].pos.normalized();
	}
	//color = vec3(1) - (vec3(1.0f) / color);
	return color;
	//vec3 unit_direction = dir.normalized();
	//float t = 0.5 * (unit_direction[1] + 1.0);
	//return vec3(1.0 - t) * vec3(1.0, 1.0, 1.0) + vec3(t) * vec3(0.5, 0.7, 1.0);
}
//the actual ray tarce kernel. It cannot be a member function
__global__ void raytracekernel(uint8_t* image, config settings, bvhnode* tree) {
	//get indexes 
	//TODO optimize indexes
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = (y * settings.w + x) * 3;
	float u = float(x) / (settings.w - 1);
	float v = float(y) / (settings.h - 1);
	//gnerate ray
	ray currentray = settings.cam.getray(u, v);
	//trace ray
	vec3 color = raycolor(currentray,tree, settings);
	//set image color
	image[w] = color[0]*255;
	image[w + 1] = color[1]*255;
image[w + 2] = color[2]*255;
}

//class for the actual CUDA ray tracer
class tracekernel {
public:
	//status for error handling
	cudaError_t status;
	//threads per block. 
	dim3 threadsPerBlock;
	//constructor allocates memory and does setup
	tracekernel(config settings, bvhnode* geometry) {
		//setup device
		cudaGetDevice(&device);
		threadsPerBlock = dim3(8, 8);
		numBlocks = dim3(settings.w / threadsPerBlock.x, settings.h / threadsPerBlock.y);
		//allocate memory
		imagesize = settings.h * settings.w * 3 * sizeof(uint8_t);
		status = cudaMalloc((void**)&device_image, imagesize);
		if (status != cudaSuccess) { std::cerr << "error creating output image on device \n"; return; }
		size_t geosize = sizeof(bvhnode) * settings.bvhsize;
		status = cudaMalloc((void**)&device_geometry, geosize);
		if (status != cudaSuccess) { std::cerr << "error allocating geometry on device \n"; return; }
		std::cout << "GPU memory succesfully allocated \n";
		//copy over normal geometry
		status = cudaMemcpy(device_geometry, geometry, geosize, cudaMemcpyHostToDevice); 
		if (status != cudaSuccess) { std::cerr << "error copying geometry on device \n"; return; }
		std::cout << "GPU memory succesfully copied over \n";
	//status = cudaMemPrefetchAsync(device_geometry,geosize, device);
		//if (status != cudaSuccess) { std::cerr << "error prefetching geometry on device \n"; return; }
	}
	//render out an image
	void render(uint8_t* image, config settings) {
		#ifdef recordspeed
		begin = std::chrono::steady_clock::now();
		#endif
		//run kenrel
		raytracekernel << <numBlocks, threadsPerBlock >> > (device_image, settings, device_geometry);
		//copy to host
		cudaMemcpy(image, device_image, imagesize, cudaMemcpyDeviceToHost);
		#ifdef recordspeed
		end = std::chrono::steady_clock::now();
		rendertime();
		#endif
	}
	~tracekernel() {
		//free memory
		cudaFree(device_image);
		cudaFree(device_geometry);
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
	//for profiling
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;


};
