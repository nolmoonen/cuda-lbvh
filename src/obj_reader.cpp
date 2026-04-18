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
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

template <typename t>
bool parse(t& x, const char*& begin, const char* end)
{
    auto [ptr, ec] = std::from_chars(begin, end, x);
    if (ec != std::errc()) {
        return false;
    }

    begin = ptr;
    return true;
}

bool read_scene(scene& s, const std::filesystem::path& filename)
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::fstream stream(filename);
    if (stream.bad()) {
        fprintf(stderr, "Could not open scene file \"%s\".\n", filename.c_str());
        return false;
    }

    // TODO reading the full file into memory takes about 25-35% of total scene loading,
    // as well as a significant amount of memory. This can be improved by doing a
    // buffered read.
    std::vector<char> file(std::filesystem::file_size(filename));
    stream.read(file.data(), file.size());
    const std::chrono::steady_clock::time_point mid = std::chrono::steady_clock::now();

    float3 smin = make_float3(+std::numeric_limits<float>::max());
    float3 smax = make_float3(-std::numeric_limits<float>::max());

    const char* begin     = file.data();
    const char* const end = file.data() + file.size();

    while (begin < end) {
        if (end - begin >= 2 && *begin == 'v' && *(begin + 1) == ' ') {
            begin += 2; // 'v '
            // Vertex position.
            float3 v;
            RETURN_IF_FALSE(parse(v.x, begin, end));
            ++begin; // skip ' '
            RETURN_IF_FALSE(parse(v.y, begin, end));
            ++begin; // skip ' '
            RETURN_IF_FALSE(parse(v.z, begin, end));
            ++begin; // skip '\n'

            // TODO these points may not necessarily be featured in a face.
            // However, finding the bounding box from the list of faces
            // is more computationally intensive.
            smin = fminf(smin, v);
            smax = fmaxf(smax, v);

            s.positions.emplace_back(v);
        } else if (end - begin >= 2 && *begin == 'v' && *(begin + 1) == 'n') {
            begin += 3; // 'vn '
            // Vertex normal.
            float3 n;
            RETURN_IF_FALSE(parse(n.x, begin, end));
            ++begin; // skip ' '
            RETURN_IF_FALSE(parse(n.y, begin, end));
            ++begin; // skip ' '
            RETURN_IF_FALSE(parse(n.z, begin, end));
            ++begin; // skip '\n'

            s.normals.emplace_back(n);
        } else if (end - begin >= 1 && *begin == 'f') {
            begin += 2; // 'f '
            // Face.
            uint2 i, j, k;
            unsigned int tmp;
            RETURN_IF_FALSE(parse(i.x, begin, end));
            ++begin; // skip '/'
            RETURN_IF_FALSE(parse(tmp, begin, end));
            ++begin; // skip '/'
            RETURN_IF_FALSE(parse(i.y, begin, end));
            ++begin; // skip ' '
            RETURN_IF_FALSE(parse(j.x, begin, end));
            ++begin; // skip '/'
            RETURN_IF_FALSE(parse(tmp, begin, end));
            ++begin; // skip '/'
            RETURN_IF_FALSE(parse(j.y, begin, end));
            ++begin; // skip ' '
            RETURN_IF_FALSE(parse(k.x, begin, end));
            ++begin; // skip '/'
            RETURN_IF_FALSE(parse(tmp, begin, end));
            ++begin; // skip '/'
            RETURN_IF_FALSE(parse(k.y, begin, end));
            ++begin; // skip '\n'

            s.indices.emplace_back(i - make_uint2(1u));
            s.indices.emplace_back(j - make_uint2(1u));
            s.indices.emplace_back(k - make_uint2(1u));
        } else {
            // Points `begin` to next '\n' if it exists, else `nullptr`.
            begin = reinterpret_cast<const char*>(std::memchr(begin, '\n', end - begin));
            if (begin == nullptr) break;
            ++begin; // skip '\n'
        }
    }

    s.soffset = smin;
    s.sextent = smax - smin;

    // TODO do a sanity check that all faces are complete?

    const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    const std::chrono::duration<float> diff_fileread = mid - start;
    const std::chrono::duration<float> diff_total    = stop - start;

    printf(
        "loaded %s with %zu triangles in %fs (file read took %fs)\n",
        filename.c_str(),
        s.indices.size() / 3,
        diff_total.count(),
        diff_fileread.count());

    return true;
}

#undef BUFFER_SIZE
