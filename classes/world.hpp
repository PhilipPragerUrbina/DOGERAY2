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
	 __device__ vec3 color(Ray ray, curandState* seed) {
		 //ray color 
		 ray.attenuation = vec3(1.0f);
		 //max bounces
		 const int maxdepth = 5;

		 //for each ray bounce
		 for (int bounce = 0; bounce < maxdepth; bounce++) {
			 if (intersect(ray)) {
				 //calculate additional hit information
				 vec3 hitpoint = ray.at(distance);
				 vec3 normal = getnormal() ;
				 vec3 texcoords = gettexcoords();

				 Mat mat = materials[triangle.materialid];
				 mat.interact(&ray, texcoords, hitpoint, normal, seed);
				

			
			 }
			 else {
				 //ray misses
				 //bg color
				 float t = 0.5 * (ray.dir.normalized()[1] + 1.0);
				 return ray.attenuation * (vec3(1.0 - t) * vec3(1.0, 1.0, 1.0) + vec3(t) * vec3(0.1, 0.1, 0.1));
			 }
		 }
		 return  vec3(0);
	}

private:

	//hit info
	vec3 uv;
	Tri triangle;
	float distance;
	//intersect wolrd
	__device__ bool intersect(Ray ray) {
		//minimum distance so far
		float mindist = 100000;
		//as soemthing hot
		bool ishit = false;
		//stackless bvh traversal from paper
		int box_index = 0;

		while (box_index != -1) {
			//get current node
			bvhnode currentnode = tree[box_index];
			//test node
			if (currentnode.testbox(ray)) {

				//bvh preview
				//color = color + vec3(settings.bvhstrength);
				box_index = currentnode.hitnode; //go down tree

				//if is a triangle test triangle
				if (currentnode.isleaf) {

					//traingle hitoutput
					float tempdist;
					vec3 tempuv;

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
			}
			else {
				//missed go up the tree
				box_index = currentnode.missnode; 
			}
		}
		return ishit;
	}
	//get tringle normal from hit
	__device__ vec3 getnormal() {
		return ((vec3(1 - uv[0] - uv[1]) * triangle.verts[0].norm) + (vec3(uv[0]) * triangle.verts[1].norm) + (vec3(uv[1]) * triangle.verts[2].norm));
	}
	__device__ vec3 gettexcoords() {
			return ((vec3(1 - uv[0] - uv[1]) * triangle.verts[0].tex) + (vec3(uv[0]) * triangle.verts[1].tex) + (vec3(uv[1]) * triangle.verts[2].tex));
	}
};
//leftover color modes
//color = vec3(dist* settings.bvhstrength);
//color = vec3(uv[0], uv[1], 1 - uv[0] - uv[1]);
//color = triangle.verts[0].pos.normalized();
//color = vec3(1) - (vec3(1.0f) / color);
//	bool isfront = ray.dir.dot(normal) < 0;
//vec3 newnorm = isfront ? normal : normal * vec3(-1);
		