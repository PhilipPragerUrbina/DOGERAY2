#pragma once
//tiny gltf 
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "tri.hpp"
#include "config.hpp"
//needed for transformations
#include <linalg.h>

//for output
using namespace std;

//class for loading 3d files from gltf
//TODO create loader base class and create multiple file types
class Loader {
public:

    int vertexcount = 0;
	//set filename on creation of class
	string filename = "";
    //store triangles after they have been loaded
    vector<Tri> loadedtris;
    //store materials
    vector<Mat> loadedmaterials;

	Loader(string name, config* editablesettings) {
        settings = editablesettings;
        filename = name;
        //set format specific settings
        settings->cam.up = Vec3(0, -1, 0);
		cout << "Loading file: " << filename << "\n";
	}

	//load GLTF file
	void loadGLTF() {
        //create model for library
        tinygltf::Model model = loadgltfmodel(filename);
        //recursively load it into array
        creategltfmodel(model);
        //check
        if (loadedmaterials.size() == 0) {
            std::cerr << "no materials found \n";
        }
        if (loadedtris.size() == 0) {
            std::cerr << "no tris found \n";
        }
        if (!camerafound) {
            cout << "No camera found \n";
            //no camera found in file. Auto generate one
            //choose between x or z axis for camera
            int axis = 0;
            if (maxdim[2] < maxdim[0]) {
                //axis which camera is on is the axis with the least max dimensions(least width)
                //this is so we get a side view of the object
                axis = 2;
            }
            Vec3 campos{ 0 };
            campos[axis] = (maxdim[0] + maxdim[2])*4;
            settings->cam.position = campos;
            cout << "Automatically created camera at: " << settings->cam.position << "\n";
        }
        //print info
        if (containsnontris) { cout << "warning: Model may contain non tris \n"; }
        cout << "GLTF model parsed successfully! \n";
        cout << loadedtris.size() << " tris \n";
        cout << vertexcount << " verts \n";
	};

private:
    config* settings;
    bool containsnontris = false;
    bool camerafound = false;
    //for choosing auto camera pos
    Vec3 maxdim{0};
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
        }
        else {
            cout << "GLTF model loaded successfully \n";
        }
        return model;
    }
    //create triangles from GLTF mesh and set attributes
    void createmesh(tinygltf::Model& model, tinygltf::Mesh& mesh, linalg::aliases::float4x4 globalmatrix) {
        //for each tri
        for (size_t i = 0; i < mesh.primitives.size(); ++i) {
            //set up material
            //get GLTF mat id
            int gltfmaterialid = mesh.primitives[i].material;
            //actual id in material array
            int arraymaterialid = -1;
            for (int i = 0; i < loadedmaterials.size(); i++) {
                //check if exists
                if (loadedmaterials[i].id == gltfmaterialid) {
                    arraymaterialid = i;
                    break;
                }
            }
            //create mat if not exist
            if (arraymaterialid == -1) {       
                //get gltf material
                auto gltfmat = model.materials[mesh.primitives[i].material];
                //set color
                Vec3 color(gltfmat.pbrMetallicRoughness.baseColorFactor[0], gltfmat.pbrMetallicRoughness.baseColorFactor[1], gltfmat.pbrMetallicRoughness.baseColorFactor[2]);
                Vec3 emmision(gltfmat.emissiveFactor[0], gltfmat.emissiveFactor[1], gltfmat.emissiveFactor[2]);
                //create and add mat
                Mat newmat(gltfmaterialid, color, gltfmat.pbrMetallicRoughness.metallicFactor,gltfmat.pbrMetallicRoughness.roughnessFactor,emmision);
               
                //load color texture
                if (gltfmat.pbrMetallicRoughness.baseColorTexture.index != -1){
                    auto colorimage = model.images[model.textures[gltfmat.pbrMetallicRoughness.baseColorTexture.index].source];
                    newmat.colortexture.create(colorimage.image.data(), colorimage.width, colorimage.height, colorimage.component, colorimage.bits);
                }
                //load rough texture
                if (gltfmat.pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
                    auto roughimage = model.images[model.textures[gltfmat.pbrMetallicRoughness.metallicRoughnessTexture.index].source];
                    newmat.roughtexture.create(roughimage.image.data(), roughimage.width, roughimage.height, roughimage.component, roughimage.bits);
                }
                //load emission map
                if (gltfmat.emissiveTexture.index != -1) {
                    auto emimage = model.images[model.textures[gltfmat.emissiveTexture.index].source];
                    newmat.emmisiontexture.create(emimage.image.data(), emimage.width, emimage.height, emimage.component, emimage.bits);
                }

                //load normal map
             //   if (gltfmat.normalTexture.index != -1) {
               //     auto normalimage = model.images[model.textures[gltfmat.normalTexture.index].source];
               //     newmat.normaltexture.create(normalimage.image.data(), normalimage.width, normalimage.height, normalimage.component, normalimage.bits);
               // }
               loadedmaterials.push_back(newmat);
                //set id
                arraymaterialid = loadedmaterials.size()-1;
            }
            //get indices for correct number of tris
            tinygltf::Accessor indexAccessor = model.accessors[mesh.primitives[i].indices];
            tinygltf::BufferView& ibufferView = model.bufferViews[indexAccessor.bufferView];
            tinygltf::Buffer& ibuffer = model.buffers[ibufferView.buffer];
           
            //2 byte indexes
            const uint16_t* smallindexes;
            //4 byte indexes
            const int* indexes;
            //some have ints of 2, some have of 4
            bool indexis16bit = indexAccessor.ByteStride(ibufferView) == 2;
            //reinterpret once for efficiency
            if (indexis16bit) {
                smallindexes = reinterpret_cast<const uint16_t*>(&ibuffer.data[ibufferView.byteOffset + indexAccessor.byteOffset]);

            }else{
                indexes = reinterpret_cast<const int*>(&ibuffer.data[ibufferView.byteOffset + indexAccessor.byteOffset]);
            }

            //get tri positions
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
          
            //triangle index. Which vertex out of 3 is it.
            int e = 0;
            Tri newtri;
            newtri.materialid = arraymaterialid;
            //update vertex count
            vertexcount += accessor.count;
        
            //loop through the indices(vertices in triangles)
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
                newtri.verts[e].pos =Vec3(positions[index * 3 + 0], positions[index * 3 + 1], positions[index * 3 + 2]);
                    //apply matrix transform on point
                    //change point to linalg vector 4
                    linalg::aliases::float4 old;
                    old.x = newtri.verts[e].pos[0];
                    old.y = newtri.verts[e].pos[1];
                    old.z = newtri.verts[e].pos[2];
                    old.w = 1;
                    //apply matrix
                    old = mul(globalmatrix,old);
                    //update vert
                    newtri.verts[e].pos[0] = old.x;
                    newtri.verts[e].pos[1] = old.y;
                    newtri.verts[e].pos[2] = old.z;
                    //set new max dimensions for camera positioning algorithm
                    if (old.x > maxdim[0]) {
                        maxdim[0] = old.x;
                    }
                    if (old.z > maxdim[2]) {
                        maxdim[2] = old.z;
                    }

                //set norm 
                newtri.verts[e].norm = Vec3(normals[index * 3 + 0], normals[index * 3 + 1], normals[index * 3 + 2]);
                //get rotation from transformation matrix
                linalg::aliases::float4x4 rotationmatrix = linalg::identity;
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        rotationmatrix[x][y] = globalmatrix[x][y];
                    }
                }
                //apply matrix rotation to normal
                linalg::aliases::float4 old2;
                old2.x = newtri.verts[e].norm[0];
                old2.y = newtri.verts[e].norm[1];
                old2.z = newtri.verts[e].norm[2];
                old2.w = 1;
                //apply matrix
                old2 = mul(rotationmatrix, old2);
                //update vert
                newtri.verts[e].norm[0] = old2.x;
                newtri.verts[e].norm[1] = old2.y;
                newtri.verts[e].norm[2] = old2.z;
                newtri.verts[e].norm = newtri.verts[e].norm.normalized();

                //set tex 
                newtri.verts[e].tex = Vec3(tex[index * 2 + 0], tex[index * 2 + 1], 0);

                e++;
                if (e == 3) {
                    //finished triangle. Submit to all triangles
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
        tinygltf::Node& node, linalg::aliases::float4x4 globalmatrix) {
      
       //check if node has matrix
        if (node.matrix.size() > 0) {
            //get node matrix
            linalg::aliases::float4x4 localmatrix;
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    localmatrix[x][y] = node.matrix[(x * 4) + y];
                }
            }
            //update global transform  
                globalmatrix = mul(globalmatrix,localmatrix);
        }
        else{
            //does not have matrix. Needs to be created from TRS
            //create translation matrix
            linalg::aliases::float4x4 translationmatrix = linalg::identity;
            if (node.translation.size() > 0) {
                translationmatrix[3][0] = node.translation[0];
                translationmatrix[3][1] = node.translation[1];
                translationmatrix[3][2] = node.translation[2];
            }
            //create rotation matrix
            linalg::aliases::float4x4 rotationmatrix = linalg::identity;
           
            if (node.rotation.size() > 0) {
                linalg::aliases::float4 rotation{ 0,0,0,1 };
                rotation.x = node.rotation[0];
                rotation.y = node.rotation[1];
                rotation.z = node.rotation[2];
                rotation.w = node.rotation[3];
                linalg::aliases::float3x3 smallrotationmatrix = qmat(rotation);
                //convert to mat4
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        rotationmatrix[x][y] = smallrotationmatrix[x][y];
                    }
                }
            }
       
            //create scale matrix
            linalg::aliases::float4x4 scalematrix = linalg::identity;
            if (node.scale.size() > 0) {
                scalematrix[0][0] = node.scale[0];
                scalematrix[1][1] = node.scale[1];
                scalematrix[2][2] = node.scale[2];
            }
            //get local matrix
            linalg::aliases::float4x4 localmatrix = mul(mul(translationmatrix, rotationmatrix), scalematrix);
          
            //apply transformation
                globalmatrix = mul(globalmatrix,localmatrix);
        }


        if (node.camera > -1 && !camerafound) {
            //empty position
            linalg::aliases::float4 old{ 0,0,0,1 };
            //apply matrix
            old = mul(globalmatrix, old);
            //update camera position
            settings->cam.position = Vec3(old.x, old.y, old.z);
            //fov(convert to degrees)
            settings->cam.degreefov = model.cameras[node.camera].perspective.yfov * (180.0f / 3.141592653589793238463);
            //update camera rotation
            //get rotation from transformation matrix
            linalg::aliases::float4x4 rotationmatrix = linalg::identity;
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    rotationmatrix[x][y] = globalmatrix[x][y];
                }
            }
            //apply matrix rotation to direction
            linalg::aliases::float4 old2{ 0,0,1,1 };
            old2 = mul(rotationmatrix, old2); 
            //add a little bit to avoid zero errors
            settings->cam.rotation = Vec3(old2.x , old2.y, old2.z + 0.001f);
            settings->cam.lookat = false;
            camerafound = true;
            cout << "camera found at: " << settings->cam.position  <<  " with rotation: " << settings->cam.rotation  << " with fov: " << settings->cam.degreefov<< "\n";
        }

        //if mesh load vertices
        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
            createmesh(model, model.meshes[node.mesh],globalmatrix);
        }

        //then check children
        for (size_t i = 0; i < node.children.size(); i++) {
            gltfnode(model, model.nodes[node.children[i]], globalmatrix);
        }
    }

    //opens gltf scene
    void creategltfmodel(tinygltf::Model& model) {
        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i) {
           linalg::aliases::float4x4 globalmatrix = linalg::identity;
            gltfnode(model, model.nodes[scene.nodes[i]], globalmatrix);
        }
    }

};