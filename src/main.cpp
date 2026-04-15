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

#include "build.h"
#include "trace.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cstdio>
#include <vector>

bool run(
    const char* file_in,
    const char* file_out,
    unsigned int size_x,
    unsigned int size_y,
    unsigned int sample_count,
    float3 origin,
    float3 target,
    float3 up)
{
    // parse obj file
    scene s;
    RETURN_IF_FALSE(read_scene(s, file_in));

    // build bvh
    bvh bvh;
    RETURN_IF_FALSE(build(s, bvh));

    // generate image
    buf_cpu<uchar> image;
    RETURN_IF_FALSE(image.resize(size_y * size_x * 3));
    RETURN_IF_FALSE(
        generate(size_x, size_y, sample_count, image.get_ptr(), bvh, origin, target, up));

    // write image to file
    stbi_flip_vertically_on_write(1);
    stbi_write_png(file_out, size_x, size_y, 3, image.get_ptr(), size_x * 3);
    printf("generated %s\n", file_out);

    return true;
}

int main(int argc, char* argv[])
{
    if (argc != 15) {
        fprintf(stderr, "did not specify correct amount of parameters\n");
        return EXIT_FAILURE;
    }

    // read input
    const char* file_in  = argv[1];
    const char* file_out = argv[2];
    unsigned int size_x, size_y, sample_count;
    sscanf(argv[3], "%u", &size_x);
    sscanf(argv[4], "%u", &size_y);
    sscanf(argv[5], "%u", &sample_count);
    float3 origin, target, up;
    sscanf(argv[6], "%f", &origin.x);
    sscanf(argv[7], "%f", &origin.y);
    sscanf(argv[8], "%f", &origin.z);
    sscanf(argv[9], "%f", &target.x);
    sscanf(argv[10], "%f", &target.y);
    sscanf(argv[11], "%f", &target.z);
    sscanf(argv[12], "%f", &up.x);
    sscanf(argv[13], "%f", &up.y);
    sscanf(argv[14], "%f", &up.z);

    // TODO
    // - fix visual glitches. it's not (only) caused by traversing a nullptr
    // - further improve scene loading performance
    // - fix color interpretation

    if (!run(file_in, file_out, size_x, size_y, sample_count, origin, target, up)) {
        return EXIT_FAILURE;
    }
}
