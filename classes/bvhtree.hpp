#pragma once
#include "bvhnode.hpp"
#include <vector>
#include <algorithm>
//helper fucntioons for bvh building
//function for computing the boudning box of multiple objects
boundingbox arrayboundingbox(std::vector<tri> in) {

	boundingbox out = in[0].getboundingbox();
	for (tri object : in) {
		boundingbox objectboundingbox = object.getboundingbox();
		out.min = out.min.min(objectboundingbox.min);
		out.max = out.max.min(objectboundingbox.max);
	}
	return out;
}
//boudnign box comparison fucntions
bool boundingboxcompare(tri a, tri b, int xyz) {
	return a.box.min[xyz] < b.box.min[xyz];
}

//one for each axis
bool boundingboxcomparex(const tri a, const tri b) {
	return boundingboxcompare(a, b, 0);
}

bool boundingboxcomparey(const tri a, const tri b) {
	return boundingboxcompare(a, b, 1);
}

bool boundingboxcomparez(const tri a, const tri b) {
	return boundingboxcompare(a, b, 2);
}

//calculate standard devation
float getaxisdeviation(std::vector<tri> objects, int axis)
{
	//get mean of array
	float sum = 0.0f;
	for (int i = 0; i < objects.size(); i++)
	{
		sum += objects[i].box.min[axis];
	}
	float mean = sum / objects.size();

	//get deviation
	float deviation = 0.0;
	for (int i = 0; i < objects.size(); i++) {
		deviation += pow(objects[i].box.min[axis] - mean, 2);
	}
	return  sqrt(deviation / objects.size());
}
//bounding volume hearchy class class
class bvhtree {
public:
	bvhtree(std::vector<tri> in) {
		traingles = in;
	}
	//build bvh tree
	void build(){
		//split
		std::cout << "building BVH \n";
		recursivesplit(traingles);
		//link
		createlinks();
		std::cout << "BVH built! \n";
	}
	//copy final nodes to array
	//set size as well
	bvhnode* getNodes(int& size) {
		size =  nodes.size();
		return nodes.data();
	}
private:
    //store triangles to be sorted into tree
	std::vector<tri> traingles;
	//array of nodes to be put on gpu later
	std::vector<bvhnode> nodes;
	//recusibly split bvh into nodes until there are elaf nodes with one triangle
	void recursivesplit(std::vector<tri> remaining) {
		//node is created on the stack since it will be stored on an array later and passed to the gpu
		bvhnode parent;
		//create boudning box of node
		parent.box = arrayboundingbox(remaining);
		//check if leaf
		if (remaining.size() <= 1) {
			parent.isleaf = true;
			parent.traingle = remaining[0];
			nodes.push_back(parent);
			return;
		}
		//set children indexes. Nodes.size() is one alrger than the actual last index. This is good since we have no pushed the current node yet.
		parent.indexofchilda = nodes.size();
		parent.indexofchildb = nodes.size()+1;
		//push current node
		nodes.push_back(parent);
		//get axis with most differnce
		int axis = 0;
		int x = getaxisdeviation(remaining, 0);
		int y = getaxisdeviation(remaining, 0);
		if (y > x) { axis = 1; }
		int z = getaxisdeviation(remaining, 0);
		if (z > x && z > y) { axis = 2; }
		//sort current triangles based on axis
		switch (axis){
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
		std::vector<tri> a;
		std::vector<tri> b;
		for (int i = 0; i < remaining.size(); i++) {
			if (i < remaining.size() / 2) {
				a.push_back(remaining[i]);
			}
			else {
				b.push_back(remaining[i]);
			}
		}
		//proccess children nodes
		recursivesplit(a);
		recursivesplit(b);
	}
	//create links for stackless bvh traversal later
	void createlinks() {

	}

};
