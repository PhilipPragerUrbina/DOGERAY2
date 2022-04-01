#pragma once
#include "ray.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

//camera class. based on ray tracing in one weekend
class Camera {
public:
    //camera settings
    Vec3 position{ 1, 1, 100 };
    Vec3 lookposition{ 0, 0, 0 };
    Vec3 rotation{ 0,0,0 };
    bool lookat = true;
    float degreefov = 45;
    Vec3 up{ 0, 1, 0 };
    float aperture = 0;
    float focusoffset = 0;
    

    //no constructor to allow this to be in struct
    //calculate camera paramters
    void calculate() {
        //autofocus
        Vec3 focusdistance = (position - lookposition).length() + focusoffset;
        //convert fov from degrees to radians
        radfov = degreefov * M_PI / 180;
        float h = tan(radfov / 2);

        lensrad = aperture / 2;
        float viewportheight = 2.0 * h;
        float viewportwidth = aspectratio * viewportheight;

        Vec3 w = (position - lookposition).normalized();
        if (!lookat) {
            w = rotation.normalized();
        }

        Vec3 u = up.cross(w).normalized();
        uu = u;
        Vec3 v = w.cross(u);
        vv = v;
        //set values for ray generation on gpu
        horizontal = Vec3(focusdistance * viewportwidth) * u;
        vertical = Vec3(focusdistance * viewportheight) * v;
        llc = position - horizontal / 2 - vertical / 2 - w * focusdistance;


    }

    //get ray on device
    //calculate DOF
    __device__ Ray getray(float u, float v, Vec3 randomdisk) {
        Vec3 dir = randomdisk * lensrad;
        Vec3 offset = uu * dir[0] + vv * dir[1];
        return Ray(position + offset, llc + Vec3(u) * horizontal + Vec3(v) * vertical - position - offset);
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
    float lensrad;
    Vec3 uu;
    Vec3 vv;
};


  