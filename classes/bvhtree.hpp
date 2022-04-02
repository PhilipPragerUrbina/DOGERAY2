#pragma once
#include <vector>
#include <algorithm>
#include "tri.hpp"

struct bvhnode {
public:
	//bounding box of node
	boundingbox box;
	// is a leaf node
	bool isleaf = false;
	//triangle object it contains if it is a leaf node
	Tri traingle;
	//children
	int indexofchilda;
	int indexofchildb;
	//links
	int hitnode;
	int missnode;
	//axis aligned bounding box (in-line since it is called a huge number of times)
	inline __device__ bool testbox(Ray ray) {
		//based on Pixar AABB function
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
				//CUDA does not support std::swap
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

//helper functions for BVH building
//function for computing the bounding box of multiple objects
boundingbox arrayboundingbox(std::vector<Tri>& in) {

	boundingbox out = in[0].getboundingbox();
	for (int i = 1; i < in.size(); i++) {
		boundingbox objectboundingbox = in[i].getboundingbox();
		out.min = out.min.min(objectboundingbox.min);
		out.max = out.max.max(objectboundingbox.max);
	}
	out.center = (out.min + out.max) / Vec3(2.0f);
	return out;
}

class Bvhtree {
public:
	Bvhtree(std::vector<Tri> in) {
		traingles = in;
	}
	//build BVH tree
	void build() {
		//split
		std::cout << "building BVH \n";
		recursivesplit(traingles);
		std::cout << "linking BVH \n";
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
	//array of nodes to be put on GPU later
	std::vector<bvhnode> nodes;
	//recursively split BVH into nodes until there are leaf nodes with one triangle
	int recursivesplit(std::vector<Tri> remaining) {
		//node is created on the stack since it will be stored on an array later and passed to the GPU
		bvhnode parent;
		//store index of current node. The node has node been pushed yet so the fact that .size() returns an index 1 bigger is good
		int parentindex = nodes.size();
		//create bounding box of node
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
		//get axis with most difference
		int axis = (parent.box.max - parent.box.min).extent();
		//split by middle
		int mid = remaining.size() / 2;
		//sort and split by center of bounding box
		std::nth_element(remaining.begin(),remaining.begin()+mid, remaining.end(),
			[axis](const Tri& a, const Tri& b) {
				return a.box.center[axis] < b.box.center[axis];
			});
		//split current triangles
		std::vector<Tri> a;
		std::vector<Tri> b;
		for (int i = 0; i < remaining.size(); i++) {
			if (i < mid) {
				a.push_back(remaining[i]);
			}
			else {
				b.push_back(remaining[i]);
			}
		}
		//process children nodes 
		//set the ids where the children are
		nodes[parentindex].indexofchilda = recursivesplit(a);
		nodes[parentindex].indexofchildb = recursivesplit(b);
		//return id of this node
		return parentindex;
	}

	//create links for stack less BVH traversal later
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
			//recursively create links
			createlinks(child1, child2);
			createlinks(child2, right);
		}
	}

};
