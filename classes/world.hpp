#pragma once
#include "bvhtree.hpp"
#include "config.hpp"
//gpu random sphere vec3 function for matirals
__device__ vec3 randomvec3insphere(curandState* seed) {
	while (true) {
		vec3 p = vec3((curand_uniform_double(seed) * 2) - 1, (curand_uniform_double(seed) * 2) - 1, (curand_uniform_double(seed) * 2) - 1);
		if (pow(p.length(), 2.0f) >= 1) continue;
		return p;
	}
}


//class to contain kernel opertations
class world {
public:
	//host defined settings
	config settings;
	//geomtry pointer
	bvhnode* tree;
	//constructor for host
	 world (bvhnode* BVH, config con) {
		settings = con;
		tree = BVH;
	}
	 //ray color fucntion
	 __device__ vec3 color(ray curray, curandState* seed) {
		 //ray color 
		 curray.attenuation = vec3(1.0f);
		 //max bounces
		 const int maxdepth = 5;
		 //for each ray bounce
		 for (int bounce = 0; bounce < maxdepth; bounce++) {
			 if (intersect(curray)) {
				 //calculate additional hit information
				 vec3 hitpoint = curray.at(distance);
				 vec3 normal = getnormal(curray) ;
				 vec3 target = hitpoint + normal + randomvec3insphere(seed);
				 //mat properties
				 vec3 matcolor = vec3(0.5f, 0.5f,0.5f);
				 //calculate direction
				 //update ray
				 curray.origin = hitpoint;
				 curray.dir = (target - hitpoint).normalized();
				 curray.attenuation = curray.attenuation * matcolor;
				 
			 }
			 else {
				 //ray misses
				 //bg color
				 float t = 0.5 * (curray.dir.normalized()[1] + 1.0);
				 return curray.attenuation * (vec3(1.0 - t) * vec3(1.0, 1.0, 1.0) + vec3(t) * vec3(0.5, 0.7, 1.0));
			 }
		 }
		 return  curray.attenuation;
		 
	}

private:
	//hit info
	vec3 uv;
	tri triangle;
	float distance;
	//intersect wolrd
	__device__ bool intersect(ray curray) {
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
			if (currentnode.testbox(curray)) {

				//bvh preview
				//color = color + vec3(settings.bvhstrength);

				box_index = currentnode.hitnode; //go down tree
				//if is a triangle test triangle
				if (currentnode.isleaf) {
					//traingle hitoutput
					float tempdist;
					vec3 tempuv;
					if (currentnode.traingle.hit(curray, tempdist, tempuv)) {
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
	__device__ vec3 getnormal(ray curray) {
		vec3 normal =  ((vec3(1 - uv[0] - uv[1]) * triangle.verts[0].norm) + (vec3(uv[0]) * triangle.verts[1].norm) + (vec3(uv[1]) * triangle.verts[2].norm));
		bool isfront = curray.dir.dot(normal) < 0;
		vec3 newnorm = isfront ? normal : normal * vec3(-1);
		return newnorm;

	}
};
//leftover color modes
//color = vec3(dist* settings.bvhstrength);
//color = vec3(uv[0], uv[1], 1 - uv[0] - uv[1]);
//color = triangle.verts[0].pos.normalized();
//color = vec3(1) - (vec3(1.0f) / color);

		