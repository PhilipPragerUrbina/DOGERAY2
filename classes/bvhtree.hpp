#pragma once
#include <vector>
#include <algorithm>
#include "tri.hpp"

//bvhnode struct
struct bvhnode {
public:
	//boundning box of node
	boundingbox box;
	// is a leaf node
	bool isleaf = false;
	//traingle object it contains if it is a leaf node
	Tri traingle;
	//children
	int indexofchilda;
	int indexofchildb;
	//links
	int hitnode;
	int missnode;
	//axis aligned bounding box (inline since it is called a huge number of times)
	inline __device__ bool testbox(Ray ray) {
		//based on pixar AABB function
		//tmin and tmax still need to be tweaked for max performance
		float tmin = (box.min[0] - ray.origin[0]) / ray.dir[0];
		float tmax = (box.max[0] - ray.origin[0]) / ray.dir[0];
		//t_min must be less than max
		if (tmin > tmax) {
			float temp = tmax;
			tmax = tmin;
			tmin = temp;
		}
		for (int a = 0; a < 3; a++) {
			auto invD = 1.0f / ray.dir[a];
			auto t0 = (box.min[a] - ray.origin[a]) * invD;
			auto t1 = (box.max[a] - ray.origin[a]) * invD;
			if (invD < 0.0f) {
				//CUDA does nat support std::swap
				float temp = t1;
				t1 = t0;
				t0 = temp;
			}
			tmin = t0 > tmin ? t0 : tmin;
			tmax = t1 < tmax ? t1 : tmax;
			if (tmax <= tmin) {
				return false;
			}
		}
		return true;
	}
};


//helper fucntioons for bvh building
//function for computing the boudning box of multiple objects
boundingbox arrayboundingbox(std::vector<Tri>& in) {

	boundingbox out = in[0].getboundingbox();
	for (Tri object : in) {
		boundingbox objectboundingbox = object.getboundingbox();
		out.min = out.min.min(objectboundingbox.min);
		out.max = out.max.max(objectboundingbox.max);
	}
	return out;
}
//boudnign box comparison fucntions
bool boundingboxcompare(Tri a, Tri b, int xyz) {
	return a.box.center[xyz] < b.box.center[xyz];
}

//one for each axis
bool boundingboxcomparex(const Tri a, const Tri b) {
	return boundingboxcompare(a, b, 0);
}

bool boundingboxcomparey(const Tri a, const Tri b) {
	return boundingboxcompare(a, b, 1);
}

bool boundingboxcomparez(const Tri a, const Tri b) {
	return boundingboxcompare(a, b, 2);
}

//get the range of the positons of triangles along an axis
float getaxisrange(std::vector<Tri> objects, int axis)
{
	float min = objects[0].box.center[axis];
	float max = objects[0].box.center[axis];
	for (Tri t : objects)
	{
		float boxpos = t.box.center[axis];
		if (boxpos < min) {
			min = boxpos;
		}
		if (boxpos > max) {
			max = boxpos;
		}
	}
	return max - min;

}
//bounding volume hearchy class class
class Bvhtree {
public:
	Bvhtree(std::vector<Tri> in) {
		traingles = in;
	}
	//build bvh tree
	void build() {
		//split
		std::cout << "building BVH \n";
		recursivesplit(traingles);
		//link
		createlinks(0, -1);
		std::cout << "BVH built! \n";
	}
	//copy final nodes to array
	//set size as well
	bvhnode* getNodes(int& size) {
		size = nodes.size();
		return nodes.data();
	}
private:
	//store triangles to be sorted into tree
	std::vector<Tri> traingles;
	//array of nodes to be put on gpu later
	std::vector<bvhnode> nodes;
	//recusibly split bvh into nodes until there are elaf nodes with one triangle
	int recursivesplit(std::vector<Tri> remaining) {
		//node is created on the stack since it will be stored on an array later and passed to the gpu
		bvhnode parent;
		//store index of current node. The node has node been pushed yet so the fact that .size() returns an index 1 bigger is good
		int parentindex = nodes.size();
		//create boudning box of node
		parent.box = arrayboundingbox(remaining);
		//check if leaf
		if (remaining.size() == 1) {
			//set as leaf
			parent.isleaf = true;
			//add object to node
			parent.traingle = remaining[0];
			//push and return id
			nodes.push_back(parent);
			return parentindex;
		}

		//push current node
		nodes.push_back(parent);
		//get axis with most differnce
		int axis = 0;
		int x = getaxisrange(remaining, 0);
		int y = getaxisrange(remaining, 1);
		if (y > x) { axis = 1; }
		int z = getaxisrange(remaining, 2);
		if (z > x && z > y) { axis = 2; }
		//sort current triangles based on axis
		switch (axis) {
		case 0:
			//x axis sort
			std::sort(remaining.begin(), remaining.end(), boundingboxcomparex);
			break;

		case 1:
			//y axis sort
			std::sort(remaining.begin(), remaining.end(), boundingboxcomparey);
			break;
		case 2:
			//z axis sort
			std::sort(remaining.begin(), remaining.end(), boundingboxcomparez);
			break;
		}
		//split current triangles
		std::vector<Tri> a;
		std::vector<Tri> b;
		for (int i = 0; i < remaining.size(); i++) {
			if (i < remaining.size() / 2) {
				a.push_back(remaining[i]);
			}
			else {
				b.push_back(remaining[i]);
			}
		}
		//proccess children nodes 
		//set the ids where the children are
		nodes[parentindex].indexofchilda = recursivesplit(a);
		nodes[parentindex].indexofchildb = recursivesplit(b);
		//return id of this node
		return parentindex;
	}

	//create links for stackless bvh traversal later
	void createlinks(int current, int right) {
		if (nodes[current].isleaf) {
			//is at end. hit or miss means go to up right node in tree
			nodes[current].hitnode = right;
			nodes[current].missnode = right;
		}
		else {
			//get children ids
			int child1 = nodes[current].indexofchilda;
			int child2 = nodes[current].indexofchildb;
			//hit means go to child
			nodes[current].hitnode = child1;
			//miss means go to up right node
			nodes[current].missnode = right;
			//reucrisvley create links
			createlinks(child1, child2);
			createlinks(child2, right);
		}
	}

};
