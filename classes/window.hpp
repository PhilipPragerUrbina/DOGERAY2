#pragma once
//we are usign the SDL libary to handel windows
#include <SDL.h>
#include <iostream>
#include <string>
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
            SDLwindow = SDL_CreateWindow(name.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
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

        //create surface
        SDL_Surface* tempsurface = SDL_CreateRGBSurface(0, width, height, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);

        //transfer pixels to surface
        SDL_RenderReadPixels(SDLrenderer, NULL, SDL_PIXELFORMAT_ARGB8888, tempsurface->pixels, tempsurface->pitch);
        SDL_SaveBMP(tempsurface, (filename + ".bmp").c_str());

        //delete surface
        SDL_FreeSurface(tempsurface);
    }

private:

    int width;
    int height;
    SDL_Window* SDLwindow;
    SDL_Renderer* SDLrenderer;
    SDL_Texture* SDLtexture;
};
