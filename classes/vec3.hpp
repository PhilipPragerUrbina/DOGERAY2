#pragma once
//cuda runtime
#include <cuda_runtime.h>
//Vec3 class to be used all over
class Vec3 {
public:
    //public for editing by imgui
    float values[3];

	//constructors
    __host__ __device__ Vec3(float x, float y, float z) {
        values[0] = x;
        values[1] = y;
        values[2] = z;
    }
    __host__ __device__ Vec3(float x) {
        values[0] = x;
        values[1] = x;
        values[2] = x;
    }
    __host__ __device__ Vec3() {
        values[0] = 0;
        values[1] = 0;
        values[2] = 0;
    }
    //copy constructor
    __host__ __device__  Vec3(const Vec3& b) {
        values[0] = b.values[0];
        values[1] = b.values[1];
        values[2] = b.values[2];
    }

    //operations on vector
    //returns dot product of current vector and other vector
    __host__ __device__ float dot(const Vec3& b)
    const {
        return (values[0] * b[0]) + (values[1] * b[1]) + (values[2] * b[2]);
    }
    //get cross product(right hand rule :)! )
    __host__ __device__ Vec3 cross(const Vec3& b)
    const {
        return Vec3(values[1] * b[2] - values[2] * b[1], values[2] * b[0] - values[0] * b[2], values[0] * b[1] - values[1] * b[0]);
    }
    //returns normalized version of vector
    __host__ __device__ Vec3 normalized() const {
        return Vec3(*this) / length();
    }
    //length
    __host__ __device__ float length() const {
        return sqrtf(dot(*this));
    }
    //clamp vector between two values
    __host__ __device__ Vec3 clamped(float min, float max) const {
        Vec3 out = *this;
        for (int i = 0; i < 3; i++) {
            if (out[i] < min) out[i] = min;
            else if (out[i] > max) out[i] = max;
       }
        return out;
    }
    //reflect vector over other vector(normal)
    __host__ __device__ Vec3 reflected(const Vec3& n) const  {
        return Vec3(*this) - (n * 2.0 * dot(n));
    }
    //change order
    __host__ __device__ Vec3 inverse() const {
        return Vec3(values[2], values[1], values[0]);
    }
    //get largest axis
    __host__ __device__ int extent() const {
        int max = 0;
        if (values[1] > values[0]) { max = 1; }
        if (values[2] > values[0] && values[2] > values[1]) { max = 2; }
        return max;
    }

    //gets minimum and max values of two vectors
    Vec3 min(const Vec3& b) const{
        return Vec3(fmin(values[0], b[0]), fmin(values[1], b[1]), fmin(values[2], b[2]));
    }
    Vec3 max(const Vec3& b) const{
        return Vec3(fmax(values[0], b[0]), fmax(values[1], b[1]), fmax(values[2], b[2]));
    }

    //operators
    //access x like Vec3[0] for conveince. This is better thnat vec2.x since you can use loops
    __host__ __device__  inline  float operator [](int i) const { return values[i]; }
    __host__ __device__  inline  float& operator [](int i) { return values[i]; }
    __host__ __device__ inline  Vec3 operator+(const Vec3& b) const {
        return Vec3(values[0] + b.values[0],values[1] + b.values[1],values[2] + b.values[2]);
    }
    __host__ __device__ inline  Vec3 operator-(const Vec3& b) const {
        return Vec3(values[0] - b.values[0],values[1] - b.values[1], values[2] - b.values[2]);
    }
    __host__ __device__ inline  Vec3 operator*(const Vec3& b) const{
        return Vec3(values[0] * b.values[0], values[1] * b.values[1], values[2] * b.values[2]);
    }
    __host__ __device__ inline  Vec3 operator/(const Vec3& b) const {
        return Vec3(values[0] / b.values[0],values[1] / b.values[1],values[2] / b.values[2]);
    }
    __host__ __device__ Vec3 operator-() const { return Vec3(-values[0], -values[1], -values[2]); }


    //print Vec3 to console for debugging
    void print() const{
        std::cout << "(" << values[0] << "," << values[1] << "," << values[2] << ")" << std::endl;
    }
};

//cout overload for printing Vec3 c++ way
std::ostream& operator<<(std::ostream& os,  Vec3& a)
{
    os << "(" << a[0] << "," << a[1] << "," << a[2] << ")";
    return os;
}
