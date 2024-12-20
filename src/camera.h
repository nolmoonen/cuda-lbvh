// Copyright (c) 2022-2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>

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
