#include "program.hpp"
int main(int argc, char* args[])
{
    Program raytracer;
    raytracer.displaytitle();
    //check what file to open. if none specified, try to open default file. Normally you use "open with" to open scenes.
    std::string filename = "defualt.gltf";
    if (argc > 1)  filename = args[1];
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

   