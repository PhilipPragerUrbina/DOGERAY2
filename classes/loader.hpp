#pragma once
//tiny gltf 
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
//standard library
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
//include triangle classs
#include "tri.hpp"
#include "config.hpp"

using namespace std;
//class for loading 3d files
class Loader {
public:

    int vertexcount = 0;
	//set filename on creation of class
	string filename = "";
    //store triangles afetr they have been loaded
    vector<Tri> loadedtris;

	Loader(string name, config* editablesettings) {
        settings = editablesettings;
        filename = name;
        //set format specific settings
        settings->cam.up = vec3(0, -1, 0);
		cout << "Loading file: " << filename << "\n";
	}

	//load GLTF file
	void loadGLTF() {
        //create model for library
        tinygltf::Model model = loadgltfmodel(filename);
        //recusivley load it into array
        creategltfmodel(model);
        //print info
        if (containsnontris) { cout << "warning: Model may contain non tris \n"; }
        cout << "GLTF model parsed succesfully! \n";
        cout << loadedtris.size() << " tris \n";
        cout << vertexcount << " verts \n";
	};

private:
    config* settings;
    bool containsnontris = false;

    //use library to open up model
    tinygltf::Model loadgltfmodel(string filename) {
        tinygltf::TinyGLTF gltfloader;
        tinygltf::Model model;
        string err;
        string warn;
        bool res;

        //check if binary gltf file(GLB)
        if (filename.substr(filename.find_last_of(".") + 1) == "glb") {
            res = gltfloader.LoadBinaryFromFile(&model, &err, &warn, filename);
        }
        else {
            res = gltfloader.LoadASCIIFromFile(&model, &err, &warn, filename);
        }
            
        if (!warn.empty()) {
            cout << "GLTF warning: " << warn << "\n";
        }
        if (!err.empty()) {
            cerr << "GLTF error: " << err << "\n";
        }
        if (!res) {
            cerr << "GLTF failed";
            exit(1);
        }
        else {
            cout << "GLTF model loaded succesfully \n";
        }

        return model;
    }

    //create triangles from GLTF mesh and set attributes
    void createmesh(tinygltf::Model& model, tinygltf::Mesh& mesh, vec3 pos) {
        //for each tri
        for (size_t i = 0; i < mesh.primitives.size(); ++i) {

            //get indeces for correct number of tris
            tinygltf::Accessor indexAccessor = model.accessors[mesh.primitives[i].indices];
            tinygltf::BufferView& ibufferView = model.bufferViews[indexAccessor.bufferView];
            tinygltf::Buffer& ibuffer = model.buffers[ibufferView.buffer];
           
            //2 byte indexes
            const uint16_t* smallindexes;
            //4 byte indexes
            const int* indexes;

            //some have ints of 2, some have of 4
            bool indexis16bit = indexAccessor.ByteStride(ibufferView) == 2;

            //reinpret once for efficency
            if (indexis16bit) {
                smallindexes = reinterpret_cast<const uint16_t*>(&ibuffer.data[ibufferView.byteOffset + indexAccessor.byteOffset]);

            }else{
                indexes = reinterpret_cast<const int*>(&ibuffer.data[ibufferView.byteOffset + indexAccessor.byteOffset]);
            }

            //get tri postitons
            tinygltf::Accessor& accessor = model.accessors[mesh.primitives[i].attributes["POSITION"]];
            tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

            //get tri normals
            tinygltf::Accessor& accessor1 = model.accessors[mesh.primitives[i].attributes["NORMAL"]];
            tinygltf::BufferView& bufferView1 = model.bufferViews[accessor1.bufferView];
            tinygltf::Buffer& buffer1 = model.buffers[bufferView1.buffer];
            const float* normals = reinterpret_cast<const float*>(&buffer1.data[bufferView1.byteOffset + accessor1.byteOffset]);
     
            //get tri texture coordinates
            tinygltf::Accessor& accessor2 = model.accessors[mesh.primitives[i].attributes["TEXCOORD_0"]];
            tinygltf::BufferView& bufferView2 = model.bufferViews[accessor2.bufferView];
            tinygltf::Buffer& buffer2 = model.buffers[bufferView2.buffer];
            const float* tex = reinterpret_cast<const float*>(&buffer2.data[bufferView2.byteOffset + accessor2.byteOffset]);
          
            //triange index. Which vertex out of 3 is it.
            int e = 0;
            Tri newtri;
            //update vertex count
            vertexcount += accessor.count;
        
            //loop through the indeces(vertices in traingles)
            for (size_t i = 0; i < indexAccessor.count; i++) {
                //get index
                int index;
                if (indexis16bit) {
                    index = smallindexes[i];
                }
                else {
                    index = indexes[i];
                }

                //set pos
                newtri.verts[e].pos = vec3(positions[index * 3 + 0]+pos[0], positions[index * 3 + 1] + pos[0], positions[index * 3 + 2] + pos[0]);
                //set norm 
                newtri.verts[e].norm = vec3(normals[index * 3 + 0], normals[index * 3 + 1], normals[index * 3 + 2]);
                //set tex 
                //TODO set texture and amtrial. Potentially could be stored in tex.z
                newtri.verts[e].tex = vec3(tex[index * 3 + 0], tex[index * 3 + 1], 0);

                e++;
                if (e == 3) {
                    //finsished triangle. Submit to all triangles
                    e = 0;
                    loadedtris.push_back(newtri);
                }
            }     
            if (e > 0) {
                containsnontris = true;
            }
        }
    }

    //traverse gltf nodes
    void gltfnode(tinygltf::Model& model,
        tinygltf::Node& node, vec3 pos) {
        
        if (node.camera > 0) {
            settings->cam.position = pos;
            cout << "camera found \n";
        }

        //local to world coordinates
        if (node.translation.size() > 0) {
            pos = pos + vec3(node.translation[0], node.translation[1], node.translation[2]);
        }

        //if mesh load vertices
        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
         
            createmesh(model, model.meshes[node.mesh], pos);
        }

        //then check children
        for (size_t i = 0; i < node.children.size(); i++) {
            gltfnode(model, model.nodes[node.children[i]], pos);
        }
    }

    //opens gltf scene
    void creategltfmodel(tinygltf::Model& model) {
        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i) {
            gltfnode(model, model.nodes[scene.nodes[i]], vec3(0));
        }
    }

};