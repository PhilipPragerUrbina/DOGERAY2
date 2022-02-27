#pragma once
#include "tri.hpp"
struct bvhnode {
public:
	//boudning box of node
	boundingbox box;

	// is a leaf node
	bool isleaf = false;
	//traingle object it contains if it is a leaf node
	tri traingle;

	//children
	int indexofchilda;
	int indexofchildb;
	//links
	int hitnode;
	int missnode;

};

