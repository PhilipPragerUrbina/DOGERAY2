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
	 World (bvhnode* BVH, config con, Mat* mat) {
		settings = con;
		materials = mat;
		tree = BVH;
	}

	 //ray color fucntion
	 __device__ Vec3 color(Ray ray, Noise noise) {
		 //ray color 
		 ray.attenuation = Vec3(1.0f);
		 //max bounces
		 const int maxdepth = 1;

		 //for each ray bounce
		 for (int bounce = 0; bounce < maxdepth; bounce++) {
			 if (intersect(ray)) {
				 //calculate additional hit information
				 Vec3 hitpoint = ray.at(distance);
				 Vec3 normal = getnormal() ;
				 Vec3 texcoords = gettexcoords();

				
				 materials[triangle.materialid].interact(&ray, texcoords, hitpoint, normal, noise);
				

			
			 }
			 else {
				 //ray misses
				 //bg color
				 float t = 0.5 * (-ray.dir.normalized()[1] + 1.0);
				 return ray.attenuation * (Vec3(1.0 - t) * Vec3(1.5) + Vec3(t) * Vec3(0.1, 0.1, 0.1));
			 }
		 }
		 ray.attenuation = Vec3(number/255.0f);
		 return   ray.attenuation;
	}

private:

	//hit info
	Vec3 uv;
	Tri triangle;
	float distance;
	int number = 0;
	//intersect wolrd
	__device__ bool intersect(Ray ray) {
		int dir = getbvhdirection(ray);
		//minimum distance so far
		float mindist = 100000;
		//as soemthing hot
		bool ishit = false;
		//stackless bvh traversal from paper
		int box_index = 0;



		while (box_index != -1) {
			number++;
			//get current node
			bvhnode currentnode = tree[box_index];
			//test node
			if (currentnode.testbox(ray)) {

			

				//if is a triangle test triangle
				if (currentnode.isleaf) {

					//traingle hitoutput
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
				//bvh preview
			//color = color + Vec3(settings.bvhstrength);
				box_index = currentnode.hitnode[dir]; //go down tree
			}
			else {
				//missed go up the tree
				box_index = currentnode.missnode[dir]; 
			}
		}
		return true;
	}
	//get tringle normal from hit
	__device__ Vec3 getnormal() {
		return ((Vec3(1 - uv[0] - uv[1]) * triangle.verts[0].norm) + (Vec3(uv[0]) * triangle.verts[1].norm) + (Vec3(uv[1]) * triangle.verts[2].norm));
	}
	__device__ Vec3 gettexcoords() {
			return ((Vec3(1 - uv[0] - uv[1]) * triangle.verts[0].tex) + (Vec3(uv[0]) * triangle.verts[1].tex) + (Vec3(uv[1]) * triangle.verts[2].tex));
	}
	__device__ int getbvhdirection(Ray ray) {

		Vec3 absdir = Vec3(abs(ray.dir[0]), abs(ray.dir[1]), abs(ray.dir[2]));
		int a = absdir.extent();
		if (ray.dir[a] < 0) {
			return a + 3;
		}
	
		return a;
	}
};
//leftover color modes
//color = Vec3(dist* settings.bvhstrength);
//color = Vec3(uv[0], uv[1], 1 - uv[0] - uv[1]);
//color = triangle.verts[0].pos.normalized();
//color = Vec3(1) - (Vec3(1.0f) / color);
//	bool isfront = ray.dir.dot(normal) < 0;
//Vec3 newnorm = isfront ? normal : normal * Vec3(-1);
		