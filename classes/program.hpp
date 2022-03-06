#pragma once
#include "loader.hpp"
#include "window.hpp"
#include "gui.hpp"
#include "tracekernel.hpp"
#include "bvhtree.hpp"

//multithreading stufff
#include<thread>
//for checking if thread finished
bool threaddone = false;
//render on spereate thread to not slow down program
void render(config set, uint8_t* data, window* win, tracekernel* shader) {
    //get winbdow data to copy to
    data = win->getTex();
    //render frame
    shader->render(data, set);
    //update window
    win->update(data);
    threaddone = true;
}

//main program class
class program {
public:
    //congfiguration for host side to device side
    config settings;
    //output data
    uint8_t* data;
    //window and gui
    gui* Gui;
    window* win;
    //kernel class
    tracekernel* shader;
    //geometry data
    bvhnode* finishedtree;
    loader* file;
    bvhtree* tree;
    //filename of open file
    string openfilename;
    //diplsay console intro title
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
    //open file and build tree
    void loadfile(string filename) {
        file = new loader(filename, &settings);
        file->loadGLTF();
        buildbvh();
        openfilename = filename;
    }
    //intitialize ui
    void initui(int width, int height) {
        //update aspect ratio of camera based on resolution
        settings.cam.calcaspectratio(width, height);
        settings.cam.calculate();
        //create window and gui objects
        Gui = new gui("DOGERAY-gui", 200, 200);
        win = new window("DOGERAY2", width, height);
        //set configuration dimenasions
        settings.h = height;
        settings.w = width;
    }

    void runmainloop() {
        //setup kernel
        shader = new tracekernel(settings, finishedtree);
        //start initial render
        thread renderthread(render, settings, data, win, shader);
        //main loop
        while (!Gui->exit) {
            //update settings with gui input
            Gui->update(&settings);
            //check if render is done
            if (threaddone) {
                //join thread
                renderthread.join();
                //set as noy done
                threaddone = false;
                //save render if specified in config from gui
                if (settings.saveimage) {
                    win->saveimage(openfilename);
                    settings.saveimage = false;
                }
                //start thread
                renderthread = thread(render, settings, data, win, shader);
                //increment samples
                settings.samples++;
           }
        }
        //wait for thread to finsih before exiting
        if (!threaddone) {
            renderthread.join();
        }
    }
    ~program() {
        //clean up
        delete win;
        delete Gui;
        delete file;
        delete tree;
        delete shader;
        delete[] data;
        delete[] finishedtree;
    }
private:
    void buildbvh() {
        //create tree
        tree = new bvhtree(file->loadedtris);
        //build tree
        tree->build();
        //get finished tree data
        int treesize = 0;
       finishedtree = tree->getNodes(treesize);
       //set size in config
        settings.bvhsize = treesize;
    }
};

