//class for loaading 3d files
#include "program.hpp"
int main(int argc, char* args[])
{
    Program raytracer;
    raytracer.displaytitle();
    //check what file to open. if none specified, try to open defualt file. Normally you use "open with" to open scenes.
    std::string filename = "defualt.gltf";
    if (argc > 1)  filename = args[1];

    //defualt settings
    raytracer.settings.cam.position = vec3(1, 1, 100);
    raytracer.settings.cam.lookposition = vec3(0, 0, 0);
    raytracer.settings.cam.up = vec3(0, 1, 0);
    //width and height
    int wi = 1280;
    int h = 720;

    //load file
    raytracer.loadfile(filename);
    //create ui
    raytracer.initui(wi, h);
    //run program
    raytracer.runmainloop();
    return 0;
}

