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
	//Is actually rendering
	bool preview = true;
	//bvh view
	bool bvh = false;
	//background
	Vec3 backgroundcolor{0.1,0.1,0.1};
	float backgroundintensity = 1.5;
	//camera
	Camera cam;
	//should window save image
	bool saveimage = false;
	//max bounces
	int maxdepth = 3;

	//# of samaples taken
	int samples = 0;
};