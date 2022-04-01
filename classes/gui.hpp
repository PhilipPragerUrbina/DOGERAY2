#pragma once
//we are using imgui library for gui
#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_sdlrenderer.h"
#include <stdio.h>
//also suing sdl as backend for gui
#include <SDL.h>
#include <iostream>
#include <string>
#include "config.hpp"
//class for handling gui and user input
class Gui {
public:
    //window title
    std::string name;
    //bg color
    ImVec4 backgroundcolor = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    //allow main loop outisde of class to check if exit has been called
    bool exit = false;

    //constructor does initial setup
    Gui(std::string name, int width, int height) {

        //init sdl and check for errors
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)  { std::cerr << "error:" << SDL_GetError() << "\n"; return;}
        // Setup window flags
        SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

        //create window
        GUIwindow = SDL_CreateWindow(name.c_str(), 100, SDL_WINDOWPOS_CENTERED, width, height, window_flags);

        // create renderer
        GUIrenderer = SDL_CreateRenderer(GUIwindow, -1, SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
        if (GUIrenderer == NULL) { std::cerr << "Error creating gui renderer \n"; return; };

        //  ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        //style
        ImGui::StyleColorsDark();

        // set SDL as backend
        ImGui_ImplSDL2_InitForSDLRenderer(GUIwindow, GUIrenderer);
        ImGui_ImplSDLRenderer_Init(GUIrenderer);
        std::cout << "GUI inititialized succesfully \n";
    }

    ~Gui() {
        ImGui_ImplSDLRenderer_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        SDL_DestroyRenderer(GUIrenderer);
        SDL_DestroyWindow(GUIwindow);
        SDL_Quit();
    }

    void update(config* settings) {

        //poll events
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            //exit events
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                exit = true;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE)
                exit = true;

            //TODO add hotkeys here
        //    if (event.type == SDL_KEYDOWN) {
          //      switch (event.key.keysym.sym) {

         //       }
         //   }
            //if this class is re-used. Just replace this whith your events.  
        }

        //create IMGUI frame
        ImGui_ImplSDLRenderer_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        //show window contents
        //if this class is re-used. Just replace this whith your gui.

        //start gui box
        ImGui::Begin("Controls");                        
        ImGui::Text("Dogeray controls:");
        bool changed = false;
        //Render button
        if (settings->preview) {
            if (ImGui::Button("Start Render")) {
                settings->preview = false;
            }

        }
        else {
            //only show image save option when rendering
            if (ImGui::Button("Save JPG")) {
                settings->saveimage = true;
            }
            if (ImGui::Button("Stop Render")) {
                settings->preview = true;
            }
        }
        //quality settings
        ImGui::DragFloat("Resolution Scale", &settings->scale, 0.1, 0.1, 3);
        ImGui::SliderInt("# of bounces", &settings->maxdepth, 1, 10);
        ImGui::Text("samples = %d", settings->samples);
        changed |= ImGui::Button("Reset Samples");
        //camera position
        changed |= ImGui::DragFloat3("camera position", settings->cam.position.values);
        //more camera options
        if (ImGui::CollapsingHeader("Lens")) {
            changed |= ImGui::DragFloat("focus distance", &settings->cam.focusoffset);
            changed |= ImGui::SliderFloat("apeture", &settings->cam.aperture, 0, 5);
            changed |= ImGui::SliderFloat("FOV", &settings->cam.degreefov, 0, 180);
            if (settings->cam.lookat) {
                changed |= ImGui::DragFloat3("look at", settings->cam.lookposition.values);
            }
            else {
                changed |= ImGui::DragFloat3("rotation", settings->cam.rotation.values, 0.02);
            }
            changed |= ImGui::Checkbox("Look At", &settings->cam.lookat);
         
        
        }
     
        if (ImGui::CollapsingHeader("Background")) {
            changed |= ImGui::ColorEdit3("Background color", settings->backgroundcolor.values);
            changed |= ImGui::DragFloat("Background intestity", &settings->backgroundintensity, 0.1);
        }

         ImGui::Checkbox("Show BVH", &settings->bvh);
         ImGui::Text("GUI average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
         ImGui::End();

         if (changed) {
             settings->samples = 0;
         }
         //update camera
         settings->cam.calculate();

        
        // render
        ImGui::Render();
        //set backfround/clear color
        SDL_SetRenderDrawColor(GUIrenderer, (Uint8)(backgroundcolor.x * 255), (Uint8)(backgroundcolor.y * 255), (Uint8)(backgroundcolor.z * 255), (Uint8)(backgroundcolor.w * 255));
        //reset canvas
        SDL_RenderClear(GUIrenderer);
        //present to SDL
        ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());
        SDL_RenderPresent(GUIrenderer);
    }

private:
    SDL_Window* GUIwindow;
    SDL_Renderer* GUIrenderer;
};

