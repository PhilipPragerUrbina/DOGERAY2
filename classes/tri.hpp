#pragma once
#include "mat.hpp"
//vertex struct.
struct vertex {
	//vertex position
	Vec3 pos;
	//vertex normal
	Vec3 norm;
	//tex coord. ignore z for now
	Vec3 tex;
};
//boundning box struct
struct boundingbox {
	Vec3 max{ 10000000000000 };
	Vec3 min{ -1000000000000 };
	Vec3 center;
	
};

//tiangle class
class Tri {
public:


	boundingbox box;
	vertex verts[3];
	int materialid = 0;

	boundingbox getboundingbox() {
		//return minimum and maximum position values from traingle for bvh construction
		//add slight bit for perfectly flat triangles to work
		const float flaterror = 0.001f;
		box.min = Vec3(fmin(verts[0].pos[0], fmin(verts[1].pos[0], verts[2].pos[0])), fmin(verts[0].pos[1], fmin(verts[1].pos[1], verts[2].pos[1])), fmin(verts[0].pos[2], fmin(verts[1].pos[2], verts[2].pos[2]))) - Vec3(flaterror);
		box.max = Vec3(fmax(verts[0].pos[0], fmax(verts[1].pos[0], verts[2].pos[0])), fmax(verts[0].pos[1], fmax(verts[1].pos[1], verts[2].pos[1])), fmax(verts[0].pos[2], fmax(verts[1].pos[2], verts[2].pos[2]))) + Vec3(flaterror);
		box.center = (box.min + box.max) / Vec3(2.0f);
		return box;
	}
	

	__device__ bool hit(Ray ray, float& dist, Vec3& uv) {

		Vec3 v0v1 = verts[1].pos - verts[0].pos;
		Vec3 v0v2 = verts[2].pos - verts[0].pos;
		Vec3 pvec = ray.dir.cross(v0v2);
		float det = v0v1.dot(pvec);

		//floating point error range. Larger for larger objects to avoid speckling problem.
		//TODO either create system for choosig n epslon or  make file coords smaller
		const float epsilon = 0.000001f;
		if (fabs(det) < epsilon) return false;

		float invDet = 1.0f / det;
		Vec3 tvec = ray.origin - verts[0].pos;
		uv[0] = tvec.dot(pvec) * invDet;
		if (uv[0] < 0.0f || uv[0] > 1.0f) return false;

		Vec3 qvec = tvec.cross(v0v1);
		uv[1] = ray.dir.dot(qvec) * invDet;
		if (uv[1] < 0.0f || uv[0] + uv[1] > 1.0f) return false;

		dist = v0v2.dot(qvec) * invDet;
		const float delta = 0.0001f;
		if (dist > delta) //check if in small range. this si to stop ray from intersecting with traingle again after bounce.
		{
			return true;
		}
		else {
			return false;
		}

	}
};