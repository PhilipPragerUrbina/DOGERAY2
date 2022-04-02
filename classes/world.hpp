#pragma once
#include "bvhtree.hpp"
#include "mat.hpp"
//class to contain kernel operations
class World {
public:
	//host defined settings
	config settings;
	//geometry pointer
	bvhnode* tree;
	//materials pointer
	Mat* materials;

	//constructor for host
	World(bvhnode* BVH, config con, Mat* mat) {
		settings = con;
		materials = mat;
		tree = BVH;
	}

	//ray color function
	__device__ Vec3 color(Ray ray, Noise noise) {
		//bvh view
		if (settings.bvh) {
			intersect(ray);
			return Vec3(traversalcount/255.0f);
		}
		//ray color 
		ray.attenuation = Vec3(1.0f);
		//for each ray bounce
		for (int bounce = 0; bounce < settings.maxdepth; bounce++) {
			if (intersect(ray)) {
				//calculate additional hit information
				Vec3 hitpoint = ray.at(distance);
				Vec3 normal = getnormal();
				Vec3 texcoords = gettexcoords();
				//use material
				if (materials[triangle.materialid].interact(&ray, texcoords, hitpoint, normal, noise)) {
					//missive
					return ray.attenuation;
				}
			}
			else {
				//ray misses
				if (settings.backgroundtexture.exists) {
					//use bg texture
					Vec3 normalizeddir = ray.dir.normalized();
					float m = 2.0f * sqrt(pow(normalizeddir[0], 2.0f) + pow(normalizeddir[1], 2.0f) + pow(normalizeddir[2] , 2.0f));
					Vec3 t = normalizeddir / m + 0.5;
					t[1] = -t[1] + 1;
					Vec3 bgcolor = settings.backgroundtexture.get(t);
					return  ray.attenuation * bgcolor * settings.backgroundintensity;
				}
				else {
					//use generic bg
					float t = 0.5 * (-ray.dir.normalized()[1] + 1.0);
					Vec3 color = settings.backgroundcolor / 255.0f;
					return ray.attenuation * Vec3(1.0 - t) * Vec3(settings.backgroundintensity) + Vec3(t) * settings.backgroundcolor;
				}
			
			}
		}
		//return color if one bounce to avoid black preview
		if (settings.maxdepth == 1) {
			return   ray.attenuation;
		}
		return Vec3(0);
	
	}

private:
	//hit info
	Vec3 uv;
	Tri triangle;
	float distance;
	int traversalcount = 0;
	//intersect bvh
	__device__ bool intersect(Ray ray) {
		//minimum distance so far
		float mindist = 100000;
		//is something hit
		bool ishit = false;
		//stackless bvh traversal
		int index = 0;
		while (index != -1) {
			//keep track of # of traversals for BVH heatmap
			traversalcount++;
			//get current node
			bvhnode currentnode = tree[index];
			//test node
			if (currentnode.testbox(ray)) {
				//if is a triangle test triangle
				if (currentnode.isleaf) {
					//triangle hit output
					float tempdist;
					Vec3 tempuv;
					if (currentnode.traingle.hit(ray, tempdist, tempuv)) {
						if (tempdist < mindist) {
							mindist = tempdist;
							//set hit info
							distance = mindist;
							triangle = currentnode.traingle;
							uv = tempuv;
							ishit = true;
						}

					}
				}
				index = currentnode.hitnode; //go down tree
			}
			else {
				//missed go up the tree
				index = currentnode.missnode;
			}
		}
		return ishit;
	}
	//interpolate normal and tex coords
	__device__ Vec3 getnormal() {
		return Vec3(1.0f - uv[0] - uv[1]) * triangle.verts[0].norm + Vec3(uv[0]) * triangle.verts[1].norm + Vec3(uv[1]) * triangle.verts[2].norm;
	}
	__device__ Vec3 gettexcoords() {
		return ((Vec3(1.0f - uv[0] - uv[1]) * triangle.verts[0].tex) + (Vec3(uv[0]) * triangle.verts[1].tex) + (Vec3(uv[1]) * triangle.verts[2].tex));
	}
};
