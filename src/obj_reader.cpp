// Copyright (c) 2022-2026 Nol Moonen
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

#include "obj_reader.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

bool read_scene(scene& s, const char* filename)
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::fstream stream(filename);
    if (stream.bad()) {
        fprintf(stderr, "Could not open scene file \"%s\".\n", filename);
        return false;
    }

    // Read file line by line.
    std::string line;

    float3 smin = make_float3(+std::numeric_limits<float>::max());
    float3 smax = make_float3(-std::numeric_limits<float>::max());

    while (std::getline(stream, line)) {
        if (line.size() < 2) continue;

        if (line[0] == 'v' && line[1] == ' ') {
            // vertex position
            float3 v;
            sscanf(line.c_str() + 2, "%f %f %f", &v.x, &v.y, &v.z);

            // TODO these points may not necessarily be featured in a face.
            // However, finding the bounding box from the list of faces
            // is more computationally intensive.
            smin = fminf(smin, v);
            smax = fmaxf(smax, v);

            s.positions.emplace_back(v);
        } else if (line[0] == 'v' && line[1] == 'n') {
            // vertex normal
            float3 n;
            sscanf(line.c_str() + 2, "%f %f %f", &n.x, &n.y, &n.z);

            s.normals.emplace_back(n);
        } else if (line[0] == 'f') {
            // face
            uint2 i, j, k;
            unsigned int tmp;
            sscanf(
                line.c_str() + 2,
                "%u/%u/%u %u/%u/%u %u/%u/%u",
                &i.x,
                &tmp,
                &i.y,
                &j.x,
                &tmp,
                &j.y,
                &k.x,
                &tmp,
                &k.y);

            s.indices.emplace_back(i - make_uint2(1u));
            s.indices.emplace_back(j - make_uint2(1u));
            s.indices.emplace_back(k - make_uint2(1u));
        }
    }

    s.soffset = smin;
    s.sextent = smax - smin;

    const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    const std::chrono::duration<float> diff          = stop - start;

    printf("loaded %s with %zu triangles in %fs\n", filename, s.indices.size() / 3, diff.count());

    return true;
}

#undef BUFFER_SIZE
