#pragma once
//cuda runtime
#include <cuda_runtime.h>
//vec3 class to be used all over
class vec3 {
public:
    //public for editing by imgui
    float values[3];

	//constructors
    __host__ __device__ vec3(float x, float y, float z) {
        values[0] = x;
        values[1] = y;
        values[2] = z;
    }
    __host__ __device__ vec3(float x) {
        values[0] = x;
        values[1] = x;
        values[2] = x;
    }
    __host__ __device__ vec3() {
        values[0] = 0;
        values[1] = 0;
        values[2] = 0;
    }


    //operations on vector
    //returns dot product of current vector and other vector
    __host__ __device__ float dot(vec3 b)
    {
        return values[0] * b[0] + values[1] * b[1] + values[2] * b[2];
    }
    //get cross product(right hand rule :)! )
    __host__ __device__ vec3 cross(vec3 b)
    {
        return vec3(values[1] * b[2] - values[2] * b[1], values[2] * b[0] - values[0] * b[2], values[0] * b[1] - values[1] * b[0]);
    }
    //returns normalized version of vector
    __host__ __device__ vec3 normalized() {
        return *this * vec3(1.0f / length());
    }
    //length
    __host__ __device__ float length() {
        return sqrtf(this->dot(*this));
    }
    //clamp vector between two values
    __host__ __device__ vec3 clamped(float min, float max) {
        vec3 out = *this;
        for (int i = 0; i < 3; i++) {
            if (out[i] < min) out[i] = min;
            else if (out[i] > max) out[i] = max;
       }
        return out;
    }
    //gets minimum and max values of two vectors
    vec3 min(vec3 b) {
        return vec3(fmin(values[0], b[0]), fmin(values[1], b[1]), fmin(values[2], b[2]));
    }
    vec3 max(vec3 b) {
        return vec3(fmax(values[0], b[0]), fmax(values[1], b[1]), fmax(values[2], b[2]));
    }

    //operators
    //access x like vec3[0] for conveince. This is better thnat vec2.x since you can use loops
    __host__ __device__ constexpr float& operator[](int x) {
        return values[x];
    }
    __host__ __device__ vec3 operator+(const vec3& b) {
        return vec3(this->values[0] + b.values[0], this->values[1] + b.values[1], this->values[2] + b.values[2]);
    }
    __host__ __device__ vec3 operator-(const vec3& b) {
        return vec3(this->values[0] - b.values[0], this->values[1] - b.values[1], this->values[2] - b.values[2]);
    }
    __host__ __device__ vec3 operator*(const vec3& b) {
        return vec3(this->values[0] * b.values[0], this->values[1] * b.values[1], this->values[2] * b.values[2]);
    }
    __host__ __device__ vec3 operator/(const vec3& b) {
        return vec3(this->values[0] / b.values[0], this->values[1] / b.values[1], this->values[2] / b.values[2]);
    }


    //print vec3 to console for debugging
    void print() {
        std::cout << "(" << values[0] << "," << values[1] << "," << values[2] << ")" << std::endl;
    }
 
};

//cout overload fr printing vec3 c++ way
std::ostream& operator<<(std::ostream& os,  vec3& a)
{
    os << "(" << a[0] << "," << a[1] << "," << a[2] << ")";
    return os;
}
