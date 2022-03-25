#pragma once
#include "bvhtree.hpp"
#include "mat.hpp"


//class to contain kernel opertations
class World {
public:
	//host defined settings
	config settings;
	//geomtry pointer
	bvhnode* tree;
	//materials pointer
	Mat* materials;

	//constructor for host
	World(bvhnode* BVH, config con, Mat* mat) {
		settings = con;
		materials = mat;
		tree = BVH;
	}

	//ray color fucntion
	__device__ Vec3 color(Ray ray, Noise noise) {
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
				//use matririal
				if (materials[triangle.materialid].interact(&ray, texcoords, hitpoint, normal, noise)) {
					return ray.attenuation;
				}
			}
			else {
				//ray misses
				//bg color
				float t = 0.5 * (-ray.dir.normalized()[1] + 1.0);
				Vec3 color = settings.backgroundcolor / 255.0f;
				return ray.attenuation * Vec3(1.0 - t) * Vec3(settings.backgroundintensity) + Vec3(t) * settings.backgroundcolor;
			}
		}
		return   ray.attenuation;
	}

private:

	//hit info
	Vec3 uv;
	Tri triangle;
	float distance;
	int traversalcount = 0;
	//intersect wolrd
	__device__ bool intersect(Ray ray) {
		//minimum distance so far
		float mindist = 100000;
		//is soemthing hit
		bool ishit = false;
		//stackless bvh traversal
		int box_index = 0;
		while (box_index != -1) {
			//keep track of # of travsersals for BVH heatmap
			traversalcount++;
			//get current node
			bvhnode currentnode = tree[box_index];
			//test node
			if (currentnode.testbox(ray)) {
				//if is a triangle test triangle
				if (currentnode.isleaf) {
					//traingle hit output
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
				box_index = currentnode.hitnode; //go down tree
			}
			else {
				//missed go up the tree
				box_index = currentnode.missnode;
			}
		}
		return ishit;
	}
	//get tringle normal from hit
	__device__ Vec3 getnormal() {
		return ((Vec3(1 - uv[0] - uv[1]) * triangle.verts[0].norm) + (Vec3(uv[0]) * triangle.verts[1].norm) + (Vec3(uv[1]) * triangle.verts[2].norm));
	}
	__device__ Vec3 gettexcoords() {
		return ((Vec3(1 - uv[0] - uv[1]) * triangle.verts[0].tex) + (Vec3(uv[0]) * triangle.verts[1].tex) + (Vec3(uv[1]) * triangle.verts[2].tex));
	}
};
