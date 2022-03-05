#pragma once
#include "ray.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
//camera class. based on ray tracing in one weekend
class camera {
public:
    //camera settings
	vec3 position;
    vec3 lookposition;
    float degreefov = 45;
    vec3 up;
   
    //calculate camera paramters
    void calculate() {
      
        //convert fov from degrees to radians
        radfov = degreefov * M_PI / 180;
        float h = tan(radfov / 2);
        float viewportheight = 2.0 * h;
        float viewportwidth = aspectratio * viewportheight;
       
        vec3 w = (position - lookposition).normalized();
        vec3 u = up.cross(w).normalized();
        auto v = w.cross(u);
        //set values for ray generation on gpu
        horizontal = vec3(viewportwidth) * u;
        vertical = vec3(viewportheight) * v;
        llc = position - horizontal / 2 - vertical / 2 - w;
    }

    //get ray on device
    __device__ ray getray(float u, float v) {
        return ray(position, llc + vec3(u) * horizontal + vec3(v) * vertical - position);
    }

    void calcaspectratio(int width, int height) {
        aspectratio = float(width) / float(height);
    }
private:
    float radfov;
    float aspectratio = 1;
    //lower left corner
    vec3 llc;
    vec3 horizontal;
    vec3 vertical;
};