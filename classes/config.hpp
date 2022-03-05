#pragma once
#include "camera.hpp"
//information taht shpuld be passed to GPU for ray tracing
struct config {
public:
	//number of bvh nodes for alloc
	int bvhsize;
	//image width and height
	int w;
	int h;
	//strength of bvh preview
	float bvhstrength = 0.01;
	//camera
	camera cam;
	//should window save image
	bool saveimage = false;

};