#pragma once
#include "camera.hpp"
//information taht shpuld be passed to GPU for ray tracing
struct config {
public:
	//image width and height
	int w;
	int h;
	//camera
	camera cam;

};