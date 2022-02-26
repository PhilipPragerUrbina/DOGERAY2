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

using namespace std;
//class for loading 3d files
class loader {
public:
    int vertexcount = 0;
	//set filename on creation of class
	string filename = "";
    //store triangles afetr they have been loaded
    vector<tri> loadedtris;
	loader(string name) {
		filename = name;
		cout << "Loading file: " << filename << "\n";
	}
	//load GLTF file
	void loadGLTF() {
        //create model for library
        tinygltf::Model model = loadgltfmodel(filename);
        //recusivley load it into array
        creategltfmodel(model);
        if (containsnontris) { cout << "warning: Model may contain non tris \n"; }
        cout << "GLTF model parsed succesfully! \n";
        cout << loadedtris.size() << " tris \n";
        cout << vertexcount << " verts \n";
	};
private:
    bool containsnontris = false;
    //use library to open up model
    tinygltf::Model loadgltfmodel(string filename) {
        tinygltf::TinyGLTF gltfloader;
        tinygltf::Model model;
        string err;
        string warn;
        bool res = gltfloader.LoadASCIIFromFile(&model, &err, &warn, filename);
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
    void createmesh(tinygltf::Model& model, tinygltf::Mesh& mesh) {
        //for each tri
        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            //get indecis for correct number of tris
            tinygltf::Accessor indexAccessor = model.accessors[mesh.primitives[i].indices];
            tinygltf::BufferView& ibufferView = model.bufferViews[indexAccessor.bufferView];
            tinygltf::Buffer& ibuffer = model.buffers[ibufferView.buffer];
            const int* indexes = reinterpret_cast<const int*>(&ibuffer.data[ibufferView.byteOffset + indexAccessor.byteOffset]);

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
            tri newtri;
            //update vertex count
            vertexcount += accessor.count;
            //loop through all the accessors(vertices). 
            for (size_t i = 0; i < indexAccessor.count; ++i) {

                int index = indexes[i];
                //set pos
                newtri.verts[e].pos = vec3(positions[index * 3 + 0], positions[index * 3 + 1], positions[index * 3 + 2]);
                //set norm 
                newtri.verts[e].norm = vec3(normals[index * 3 + 0], normals[index * 3 + 1], normals[index * 3 + 2]);
                //set tex 
                //TODO set texture and amtrial. Potentially could be stored in tex.z
                newtri.verts[e].tex = vec3(tex[index * 3 + 0], tex[index * 3 + 1],0);
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
        tinygltf::Node& node) {
        //if mesh load vertices
        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
            createmesh(model, model.meshes[node.mesh]);
        }
        //then check children
        for (size_t i = 0; i < node.children.size(); i++) {
            gltfnode(model, model.nodes[node.children[i]]);
        }
    }
    //oepns gltf scene
    void creategltfmodel(tinygltf::Model& model) {
        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i) {
            gltfnode(model, model.nodes[scene.nodes[i]]);
        }
    }

};