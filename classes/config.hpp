#pragma once
#include "camera.hpp"
//information taht shpuld be passed to GPU for ray tracing
struct config {
public:
	//number of bvh nodes for alloc
	int bvhsize;
	int matsize;
	//image width and height
	int w;
	int h;
	float scale = 1;
	//strength of bvh preview
	float bvhstrength = 0.01;
	//camera
	Camera cam;
	//should window save image
	bool saveimage = false;
	//max bounces
	int maxdepth = 3;

	//# of samaples taken
	int samples = 0;
};