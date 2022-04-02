#pragma once
#include "loader.hpp"
#include "window.hpp"
#include "gui.hpp"
#include "tracekernel.hpp"
#include "bvhtree.hpp"
#include<thread>

//for checking if thread finished
bool threaddone = true;
//render on spereate thread to not slow down program gui
void render(config set, uint8_t* data, Window* win, Tracekernel* shader) {
    if(threaddone) {
        return;
    }
    //get window data to copy to
    data = win->getTex();
    //render frame
    shader->render(data, set);
    //update window
    win->update(data);
    threaddone = true;
}

//main program class
class Program {
public:
    //congfiguration for host side to device side
    config settings;
    //output data
    uint8_t* data;
    //window and gui
    Gui* gui;
    Window* win;
    //kernel class
    Tracekernel* shader;
    //geometry data
    bvhnode* finishedtree;
    Loader* file;
    //data is dynamically allocated to make sure it's data is not destructed before it can be put on the gpu
    Bvhtree* tree;
    Mat* materials;
    //filename of open file
    string openfilename;

    ~Program() {
        //clean up
        delete win;
        delete gui;
        delete tree;
        shader->cleantextures();
        delete shader;
        delete file;
        delete[] data;
        delete[] finishedtree;
        settings.backgroundtexture.destroy();
    }

    //display console intro title
	void displaytitle() {
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
	}

  
    void loadfile(string filename) {
        //open file and build tree
        file = new Loader(filename, &settings);
        file->loadGLTF();
        buildbvh();
        openfilename = filename;

        //load materials
        materials = file->loadedmaterials.data();
        settings.matsize = file->loadedmaterials.size();
        //load background
        loadbackground();
    }

    void loadbackground() {
        string filename = "";
        //find file named background
        for (const auto& entry : std::experimental::filesystem::directory_iterator(std::experimental::filesystem::current_path())) {
            if (entry.path().stem().string() == "background") {
                filename = entry.path().filename().string();
                break;
            }
        } 
        //chack if background file exists
        if (filename != "") {
            cout << "background found: " << filename << "\n";
            //load image
            int x, y, n;
            unsigned char *data = stbi_load(filename.c_str(), &x, &y, &n, 4);
            //create texture
            settings.backgroundtexture.create(data, x, y, 4, 8);
            //clean up
             stbi_image_free(data);
             cout << "background loaded \n";

        }
    }

    //intitialize ui
    void initui(int width, int height) {
        //update aspect ratio of camera based on resolution
        settings.cam.calcaspectratio(width, height);
        settings.cam.calculate();

        //create window and gui objects
        gui = new Gui("DOGERAY-gui", 500, 400);
        win = new Window("DOGERAY2", width, height);

        //set configuration dimenasions
        settings.h = height;
        settings.w = width;
    }

    void runmainloop() {
        //setup kernel
        shader = new Tracekernel(settings, finishedtree, materials);
        //start initial render
        prevw = settings.w;
        prevh = settings.h;
        thread renderthread(render, settings, data, win, shader);
      
        //main loop
        while (!gui->exit) {
            //update settings with gui input    
            gui->update(&settings);
            //check if render is done
            if (threaddone) {
                //join thread
                renderthread.join();
                //set as not done
                threaddone = false;
                //save render if specified in config from gui
                if (settings.saveimage) {
                    win->saveimage(openfilename);
                    settings.saveimage = false;
                }
                //get window size in case resized
                win->getsize(&settings.w, &settings.h);
                //apply user modifier
                settings.w *= settings.scale;
                settings.h *= settings.scale;
                if (settings.preview) {
                    settings.w *= 0.5;
                    settings.h *=0.5;

                }
                //make divisble by 8 for cuda blocks
                settings.w = 8 * int(settings.w / 8);
                settings.h = 8 * int(settings.h / 8);
                if (prevw != settings.w || prevh != settings.h) {
                    //resolution changed
                    //update erverthing
                    shader->resize(settings);
                    win->resizeresolution(settings);
                    settings.cam.calcaspectratio(settings.w, settings.h);
                    prevw = settings.w;
                    prevh = settings.h;
                    //settings samples to zero resets render
                    settings.samples = 0;
                }
                config tempsettings = settings;
                if (settings.preview) {
                    tempsettings.maxdepth = 1;
                }
                //start thread
                renderthread = thread(render, tempsettings, data, win, shader);
                //increment samples
                settings.samples++;    
           }
        }
        //wait for thread to finsih before exiting
        if (!threaddone) {
            renderthread.join();
        }
    }

private:
    //prev dimensions
    int prevw;
    int prevh;

    void buildbvh() {
        //create tree
        tree = new Bvhtree(file->loadedtris);
        //build tree
        tree->build();
        //get finished tree data
        int treesize = 0;
       finishedtree = tree->getNodes(treesize);
       //set size in config
        settings.bvhsize = treesize;
    }
};

