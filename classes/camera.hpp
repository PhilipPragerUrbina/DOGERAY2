#pragma once
#include "ray.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>
//camera class. based on ray tracing in one weekend
class Camera {
public:
    //camera settings
	Vec3 position;
    Vec3 lookposition;
    float degreefov = 45;
    Vec3 up;
   
    //calculate camera paramters
    void calculate() {
        //convert fov from degrees to radians
        radfov = degreefov * M_PI / 180;
        float h = tan(radfov / 2);
        float viewportheight = 2.0 * h;
        float viewportwidth = aspectratio * viewportheight;
       
        Vec3 w = (position - lookposition).normalized();
        Vec3 u = up.cross(w).normalized();
        auto v = w.cross(u);

        //set values for ray generation on gpu
        horizontal = Vec3(viewportwidth) * u;
        vertical = Vec3(viewportheight) * v;
        llc = position - horizontal / 2 - vertical / 2 - w;
    }

    //get ray on device
    __device__ Ray getray(float u, float v) {
        return Ray(position, llc + Vec3(u) * horizontal + Vec3(v) * vertical - position);
    }

    void calcaspectratio(int width, int height) {
        aspectratio = float(width) / float(height);
    }

private:

    float radfov;
    float aspectratio = 1;
    //lower left corner
    Vec3 llc;
    Vec3 horizontal;
    Vec3 vertical;
};