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

#include <sutil/vec_math.h>

struct scene {
    float3 *positions;
    unsigned int position_count;
    float3* normals;
    unsigned int normal_count;
    /// Three consecutive indices define the three points of a triangle.
    /// The two indices are the vertex position and normal.
    uint2 *indices;
    unsigned int index_count;
    float3 soffset;
    float3 sextent;
};

/// Only accepts triangle faces defined by v/t/n (so no v//n for example).
/// Calculates the scene minimum and maximum points.
/// Normals are assumed to be unitized.
/// Returns 0 on success.
int read_scene(scene *s, const char *filename);

void free_scene(scene*s);
