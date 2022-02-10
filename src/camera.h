#pragma once

#include <cuda_runtime.h>
#include <deps/sutil/vec_math.h>

/// Camera definition.
struct camera {
    float3 origin;
    float3 u; // screen right
    float3 v; // screen up
    float3 w; // screen forward
};

/// Coordinate frame.
struct frame {
    __forceinline__ __device__ frame(const float3 &p_normal)
    {
        normal = p_normal;

        if (fabsf(normal.x) > fabsf(normal.z)) {
            binormal.x = -normal.y;
            binormal.y = normal.x;
            binormal.z = 0;
        } else {
            binormal.x = 0;
            binormal.y = -normal.z;
            binormal.z = normal.y;
        }

        binormal = normalize(binormal);
        tangent = cross(binormal, normal);
    }

    /// Convert a direction in coordinate frame (0,0,1) to this coordinate frame.
    __forceinline__ __device__ void inverse_transform(float3 &p) const
    { p = p.x * tangent + p.y * binormal + p.z * normal; }

    float3 tangent;
    float3 binormal;
    float3 normal;
};

__forceinline__ __device__ float3 cosine_sample_hemisphere(
        const float u1, const float u2)
{
    // uniformly sample disk
    const float r = sqrtf(u1);
    const float phi = 2.f * M_PIf * u2;
    float3 p;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // project up to hemisphere
    p.z = sqrtf(fmaxf(0.f, 1.f - p.x * p.x - p.y * p.y));
    return p;
}
