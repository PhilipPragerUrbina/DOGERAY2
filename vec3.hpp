#pragma once
//vec3 class to be used all over
class vec3 {
public:
	//constructors
    vec3(float x, float y, float z) {
        values[0] = x;
        values[1] = y;
        values[2] = z;
    }
    vec3(float x) {
        values[0] = x;
        values[1] = x;
        values[2] = x;
    }
    vec3() {
        values[0] = 0;
        values[1] = 0;
        values[2] = 0;
    }


    //operations on vector
    //returns dot product of current vector and other vector
    float dot(vec3 b)
    {
        return values[0] * b[0] + values[1] * b[1] + values[2] * b[2];
    }
    //returns normalized version of vector
    vec3 normalized() {
        return *this * vec3(1.0f / sqrtf(this->dot(*this)));
    }

    //operators
    //access x like vec3[0] for conveince. This is better thnat vec2.x since you can use loops
    constexpr float& operator[](int x) {
        return values[x];
    }
    vec3 operator+(const vec3& b) {
        return vec3(this->values[0] + b.values[0], this->values[1] + b.values[1], this->values[2] + b.values[2]);
    }
    vec3 operator-(const vec3& b) {
        return vec3(this->values[0] - b.values[0], this->values[1] - b.values[1], this->values[2] - b.values[2]);
    }
    vec3 operator*(const vec3& b) {
        return vec3(this->values[0] * b.values[0], this->values[1] * b.values[1], this->values[2] * b.values[2]);
    }
    vec3 operator/(const vec3& b) {
        return vec3(this->values[0] / b.values[0], this->values[1] / b.values[1], this->values[2] / b.values[2]);
    }


    //print vec3 to console for debugging
    void print() {
        std::cout << "(" << values[0] << "," << values[1] << "," << values[2] << ")" << std::endl;
    }
  

private:
    float values[3];
};

//cout overload fr printing vec3 c++ way
std::ostream& operator<<(std::ostream& os,  vec3& a)
{
    os << "(" << a[0] << "," << a[1] << "," << a[2] << ")";
    return os;
}