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
//TODO add to github docs
//needed for transformations
#include <linalg.h>

//for output
using namespace std;

//struct to stroe gltf rotation values
struct Quaternion {
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 1;
};
//fucntion for rotating point by quaternion
Vec3 rotated(Vec3 v, Quaternion q)
{
    if (q.x == 0 && q.y == 0 && q.z == 0 && q.w == 1) {
        return v;
    }
    Vec3 u(q.x, q.y, q.z);
    float s = q.w;
    // Do the math
    return Vec3(2.0f * u.dot(v)) * u
        + Vec3(s * s - u.dot(u)) * v
        + Vec3(2.0f * s) * u.cross(v);
}
Quaternion qmul(Quaternion q1, Quaternion q2) {
    Quaternion out;
    out.x = q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
    out.y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
    out.z = q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
    out.w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
    return out;
}

//class for loading 3d files from gltf
//TODO create loader base class and create multiple file types
class Loader {
public:

    int vertexcount = 0;
	//set filename on creation of class
	string filename = "";
    //store triangles afetr they have been loaded
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
        //recusivley load it into array
        creategltfmodel(model);
        //check
        if (loadedmaterials.size() == 0) {
            std::cerr << "no matirials found \n";
        }
        if (loadedtris.size() == 0) {
            std::cerr << "no tris found \n";
        }
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
    void createmesh(tinygltf::Model& model, tinygltf::Mesh& mesh, Vec3 pos, Vec3 scale, Quaternion rot, linalg::aliases::float4x4 a, bool hasmatrix) {
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
         
                //get gltf matirial
                auto gltfmat = model.materials[mesh.primitives[i].material];
                //set color
                Vec3 color;
                color[0] = gltfmat.pbrMetallicRoughness.baseColorFactor[0];
                color[1] = gltfmat.pbrMetallicRoughness.baseColorFactor[1];
                color[2] = gltfmat.pbrMetallicRoughness.baseColorFactor[2];
                //create and add mat
                Mat newmat(gltfmaterialid, color, gltfmat.pbrMetallicRoughness.metallicFactor,gltfmat.pbrMetallicRoughness.roughnessFactor);
               
           

                //load color tetxure
                if (gltfmat.pbrMetallicRoughness.baseColorTexture.index != -1){
                    auto colorimage = model.images[model.textures[gltfmat.pbrMetallicRoughness.baseColorTexture.index].source];
                    newmat.colortexture.create(colorimage.image.data(), colorimage.width, colorimage.height, colorimage.component, colorimage.bits);
                }
                //load rough tetxure
                if (gltfmat.pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
                    auto roughimage = model.images[model.textures[gltfmat.pbrMetallicRoughness.metallicRoughnessTexture.index].source];
                    newmat.roughtexture.create(roughimage.image.data(), roughimage.width, roughimage.height, roughimage.component, roughimage.bits);
                }
                //load normal map
                if (gltfmat.normalTexture.index != -1) {
                    auto normalimage = model.images[model.textures[gltfmat.normalTexture.index].source];
                    newmat.normaltexture.create(normalimage.image.data(), normalimage.width, normalimage.height, normalimage.component, normalimage.bits);
                }
                loadedmaterials.push_back(newmat);
    

                //set id
                arraymaterialid = loadedmaterials.size()-1;
            }



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
            newtri.materialid = arraymaterialid;
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
                newtri.verts[e].pos = rotated(Vec3(positions[index * 3 + 0], positions[index * 3 + 1], positions[index * 3 + 2]) * scale, rot) +pos;
                if (hasmatrix) {
                    //if there is matrix. Transofrm point
                    linalg::aliases::float4 old;
                    old.x = newtri.verts[e].pos[0];
                    old.y = newtri.verts[e].pos[1];
                    old.z = newtri.verts[e].pos[2];
                    old.w = 1;
                    using namespace linalg::aliases;
                    old = mul( a,old);
                    newtri.verts[e].pos[0] = old.x;
                    newtri.verts[e].pos[1] = old.y;
                    newtri.verts[e].pos[2] = old.z;
                }

                //set norm 
                newtri.verts[e].norm = Vec3(normals[index * 3 + 0], normals[index * 3 + 1], normals[index * 3 + 2]);
                //set tex 
      
                newtri.verts[e].tex = Vec3(tex[index * 2 + 0], tex[index * 2 + 1], 0);

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
        tinygltf::Node& node, Vec3 pos, Vec3 scale, Quaternion rot, linalg::aliases::float4x4 a, bool hasmatrix) {
        
        if (node.camera > 0) {
            settings->cam.position = pos;
            cout << "camera found \n";
        }

       
        if (node.matrix.size() > 0) {
            //get node matrix
            linalg::aliases::float4x4 b;
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    b[x][y] = node.matrix[(x * 4) + y];
                }
            }
           
            if (hasmatrix) {
                a = mul(a, b);
            }
            else {
                a = b;
            }
            hasmatrix = true;
        }
        else if(!hasmatrix) {
            //local to world coordinates
            if (node.translation.size() > 0) {
                pos = pos + Vec3(node.translation[0], node.translation[1], node.translation[2]);
            }
            if (node.scale.size() > 0) {
                scale = scale * Vec3(node.scale[0], node.scale[1], node.scale[2]);
            }
            if (node.rotation.size() > 0) {
                Quaternion q;
                q.x = node.rotation[0];
                q.y = node.rotation[1];
                q.z = node.rotation[2];
                q.w = node.rotation[3];
                rot = qmul(rot, q);

            }


        }


        //if mesh load vertices
        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
         
            createmesh(model, model.meshes[node.mesh], pos, scale,rot,a ,hasmatrix);
        }

        //then check children
        for (size_t i = 0; i < node.children.size(); i++) {
            gltfnode(model, model.nodes[node.children[i]], pos,scale,rot,a,hasmatrix);
        }
    }

    //opens gltf scene
    void creategltfmodel(tinygltf::Model& model) {
        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i) {
            Quaternion rot;
           linalg::aliases::float4x4 a;
            gltfnode(model, model.nodes[scene.nodes[i]], Vec3(0), Vec3(1), rot, a, false);
        }
    }

};