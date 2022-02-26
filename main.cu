//class for loaading 3d files
#include "loader.hpp"
#include "window.hpp"
#include "imgui.h"
#include "imgui_impl_sdlrenderer.h"
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
    loader file(filename);
    file.loadGLTF();

    //print file vertices
    for (tri i : file.loadedtris) {
        i.verts[0].pos.print();
        i.verts[1].pos.print();
        i.verts[2].pos.print();
    }
      
    int wi = 200;
    int h = 100;
    window win(wi, h, "PAIN");
    uint8_t* data; 

  
    int i = 0;
    while (i<10000) {
        data = win.getTex();
        for (int x = 0; x < wi; x++) {
            for (int y = 0; y < h; y++) {
                int w = (y * wi + x) * 3;
                data[w] = x;
                data[w + 1] = y;
                data[w + 2] = sin(i/1000.0f)*200;
            }
        }
        i++;
        win.update(data);
        cout << i << "work \n";
        SDL_Delay(5);
    }
    
  
    delete[] data;
  
  
    //temporary to stop console from closing
    system("pause");
    return 0;
}

