//class for loaading 3d files
#include "loader.hpp"
#include "window.hpp"
#include "gui.hpp"
#include "tracekernel.hpp"
#include "bvhtree.hpp"
int main(int argc, char* args[])
{
    //display title
    //The R is for correct multiline formatting
    std::cout << R"(
 ______   ______   _______    ______   ______    ________   __  __     _____       
/_____/\ /_____/\ /______/\  /_____/\ /_____/\  /_______/\ /_/\/_/\   /_____/\     
\:::_ \ \\:::_ \ \\::::__\/__\::::_\/_\:::_ \ \ \::: _  \ \\ \ \ \ \  \:::_:\ \    
 \:\ \ \ \\:\ \ \ \\:\ /____/\\:\/___/\\:(_) ) )_\::(_)  \ \\:\_\ \ \     _\:\|    
  \:\ \ \ \\:\ \ \ \\:\\_  _\/ \::___\/_\: __ `\ \\:: __  \ \\::::_\/    /::_/__   
   \:\/.:| |\:\_\ \ \\:\_\ \ \  \:\____/\\ \ `\ \ \\:.\ \  \ \ \::\ \    \:\____/\ 
    \____/_/ \_____\/ \_____\/   \_____\/ \_\/ \_\/ \__\/\__\/  \__\/     \_____\/  )" << std::endl;

    std::cout << "V.2.0   by Philip Prager Urbina   2022" << std::endl;
    std::cout << "Find on github: https://github.com/PhilipPragerUrbina/DOGERAY2" << std::endl;
    //check what file to open. if none specified, try to open defualt file. Normally you use "open with" to open scenes.
    std::string filename = "defualt.gltf";
    if (argc > 1)  filename = args[1];
    //load file
    config settings;
    settings.cam.position = vec3(-600, 0, 1);
    settings.cam.lookposition = vec3(0, 0, 0);
    settings.cam.up = vec3(0, 1, 0);
  

    loader file(filename, &settings);
    file.loadGLTF();

   
    settings.cam.calculate();

    bvhtree tree(file.loadedtris);
    tree.build();
    int treesize = 0;
    bvhnode* finishedtree = tree.getNodes(treesize);
    settings.bvhsize = treesize;

    //width and height
    int wi = 1280;
    int h = 720;
    settings.cam.calcaspectratio(wi, h);
    //create window and gui
    gui g("DOGEGUI",200,200);
    window win("PAIN",wi, h);
  
    settings.h = h;
    settings.w = wi;
    //output data
    uint8_t* data; 

    //main loop
    int i = 0;
   
   
    tracekernel shader(settings, finishedtree);
    while (!g.exit) {
        //edit output data. Later will be moved to kernel
        data = win.getTex();
        shader.render(data, settings);
  

        //update 
        g.update(&settings);
        i++;
        win.update(data);

        if (settings.saveimage) {
            win.saveimage(filename);
            settings.saveimage = false;
        }

      
    }
    
    //clean up
    delete[] data;
  
  
  
    return 0;
}

