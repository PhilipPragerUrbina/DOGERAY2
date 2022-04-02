#pragma once
//we are usign the SDL libary to handel windows
#include <SDL.h>
#include <iostream>
#include <string>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>

//class for creating and displaying windows
class Window {
public:

    //name of window for top
    std::string name;

    //get height and width. Fucntions to prevent modifying
    int getW() { return width;}
    int getH() { return width; } 

    //constructor
    Window(std::string n,int w, int h) {

        //constructor sets intitial dimensions.
        width = w;
        height = h;
        //The name is also set on construction. I coudl have hardcoded it, but this class may be used for other things so it is good tokeep it flexible
        name = n;

        //initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO) >= 0)
        {
            //create SDL window
            SDLwindow = SDL_CreateWindow(name.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
            //create SDL renderer
            SDLrenderer = SDL_CreateRenderer(SDLwindow, 0, 0);
            //create texture
            SDLtexture = SDL_CreateTexture(SDLrenderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);
            std::cout << "Window inititialized succesfully \n";
        }
        else {
            //error intitizlaing SDL
            std::cerr << "SDL init error: " << SDL_GetError();
        }
    }

    //clean up SDL
    ~Window() {
        SDL_DestroyTexture(SDLtexture);
        SDL_DestroyRenderer(SDLrenderer);
        SDL_DestroyWindow(SDLwindow);
        SDL_Quit();
    }

    //update the window with new data
    //pixel format RGB24. this means to access: w = (y * width + x)*3;  r = data[w];  g = data[w+1]; b = data[w+2]
    void update(uint8_t* pixels) {
        //clean up texture
        SDL_UnlockTexture(SDLtexture);

        //present texture
        SDL_RenderCopy(SDLrenderer, SDLtexture, NULL, NULL);
        SDL_RenderPresent(SDLrenderer);
    }
    //get array to edit. Has to be called before every edit.
    uint8_t* getTex() {
        //pointer of data to edit
        uint8_t* data;
        //wdith of texture
        int pitch;
        //allow writing of tex.
        SDL_LockTexture(SDLtexture, NULL, (void**)&data, &pitch);
        return data;
    }
    //save bmp screenshot
    void saveimage(string filename) {
       
        //allow saving of multiple images
        while (std::experimental::filesystem::exists(filename + ".jpg")) {
            filename = filename +  "_";
        }
        filename = filename + ".jpg";
        //create surface
        SDL_Surface* tempsurface = SDL_CreateRGBSurface(0, width, height, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
        //transfer pixels to surface
        SDL_RenderReadPixels(SDLrenderer, NULL, SDL_PIXELFORMAT_ABGR8888 , tempsurface->pixels, tempsurface->pitch);
        //save as JPG. Formally BMP but jpg is easier to share
        stbi_write_jpg((filename).c_str(), width, height, 4, tempsurface->pixels, 80);
        // SDL_SaveBMP(tempsurface, (filename + ".bmp").c_str());

        //delete surface
        SDL_FreeSurface(tempsurface);
    }
    void resizeresolution(config settings) {
        width = settings.w;
        height = settings.h;
        SDL_DestroyTexture(SDLtexture);
        SDLtexture = SDL_CreateTexture(SDLrenderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, settings.w, settings.h);

    }
    void getsize(int* w, int* h) {
        SDL_GetWindowSize(SDLwindow,  w,h);
    }

private:

    int width;
    int height;
    SDL_Window* SDLwindow;
    SDL_Renderer* SDLrenderer;
    SDL_Texture* SDLtexture;
};

