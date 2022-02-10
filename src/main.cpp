#include <stdio.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <deps/stb/stb_image_write.h>
#include "trace.h"
#include "build.h"

int main(int argc, char *argv[])
{
    int ret;
    if (argc != 15) {
        fprintf(stderr, "did not specify correct amount of parameters\n");
    }

    // read input
    const char *file_in = argv[1];
    const char *file_out = argv[2];
    unsigned int size_x, size_y, sample_count;
    sscanf(argv[3], "%d", &size_x);
    sscanf(argv[4], "%d", &size_y);
    sscanf(argv[5], "%d", &sample_count);
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

    // parse obj file
    scene s;
    ret = read_scene(&s, file_in);
    if (ret != 0) return -1;
    printf("loaded %s with %d triangles\n", file_in, s.index_count / 3);

    // build bvh
    bvh bvh;
    ret = build(&s, &bvh);
    if (ret != 0) return -1;

    // generate image
    uchar *image = (uchar *) malloc(sizeof(char) * 3 * size_x * size_y);
    generate(size_x, size_y, sample_count, image, bvh, origin, target, up);

    // write image to file
    stbi_flip_vertically_on_write(1);
    stbi_write_png(file_out, size_x, size_y, 3, image, size_x * 3);
    free(image);
    printf("generated %s\n", file_out);

    // clean up
    clean(&bvh);
    free_scene(&s);
}