#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <assert.h>
#include "obj_reader.h"

#define BUFFER_SIZE 1024

/// Returns a null-terminated string in buffer with a length that does not
/// include the null-terminator. String is cut off to be at most BUFFER_SIZE
/// including null-terminator.
/// Returns true if this is the last line in the file, false otherwise.
static bool read_line(
        FILE *file, char buffer[BUFFER_SIZE], unsigned int *len)
{
    *len = 0; // index of the next character to put in the buffer.
    int next;
    while (true) {
        next = fgetc(file);
        // if we hit end of file or line, break and do not increment len
        if (next == EOF || next == '\n') break;
        // we find a character, set in buffer and increment len
        buffer[(*len)++] = (char) next;
        // if we have no more space for characters, break and let the last
        // character be the null-terminator
        if (*len == BUFFER_SIZE - 1) break;
    }

    // ensure the last character in the line is a terminator
    buffer[*len] = '\0';

    return next == EOF;
}

int read_scene(scene *s, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "could not open scene file \"%s\"\n", filename);
        return -1;
    }

    char buffer[BUFFER_SIZE];
    unsigned int len;

    s->position_count = 0;
    s->normal_count = 0;
    s->index_count = 0;

    // scan for sizes
    while (!read_line(file, buffer, &len)) {
        if (buffer[0] == 'v') {
            if (buffer[1] == ' ') {
                // vertex position
                s->position_count++;
            } else if (buffer[1] == 'n') {
                // vertex normal
                s->normal_count++;
            }
        } else if (buffer[0] == 'f') {
            // face, each line defines three pairs of indices
            s->index_count += 3;
        }
    }

    s->positions = (float3 *) malloc(sizeof(float3) * s->position_count);
    s->normals = (float3 *) malloc(sizeof(float3) * s->normal_count);
    s->indices = (uint2 *) malloc(sizeof(uint2) * s->index_count);

    float3 smin = make_float3(+FLT_MAX);
    float3 smax = make_float3(-FLT_MAX);

    // scan for data
    fseek(file, 0, SEEK_SET);

    unsigned int position_idx = 0;
    unsigned int normal_idx = 0;
    unsigned int index_idx = 0;

    while (!read_line(file, buffer, &len)) {
        if (buffer[0] == 'v') {
            if (buffer[1] == ' ') {
                // vertex position
                float3 v;
                sscanf(&buffer[2], "%f %f %f", &v.x, &v.y, &v.z);

                // note: these points may not necessarily be featured in a face!
                smin = fminf(smin, v);
                smax = fmaxf(smax, v);

                s->positions[position_idx++] = v;
            } else if (buffer[1] == 'n') {
                // vertex normal
                float3 n;
                sscanf(&buffer[2], "%f %f %f", &n.x, &n.y, &n.z);

                s->normals[normal_idx++] = n;
            }
        } else if (buffer[0] == 'f') {
            // face
            uint2 i, j, k;
            unsigned int tmp;
            sscanf(
                    &buffer[2], "%d/%d/%d %d/%d/%d %d/%d/%d",
                    &i.x, &tmp, &i.y, &j.x, &tmp, &j.y, &k.x, &tmp, &k.y);

            s->indices[index_idx++] = i - make_uint2(1u);
            s->indices[index_idx++] = j - make_uint2(1u);
            s->indices[index_idx++] = k - make_uint2(1u);
        }
    }

    assert(s->position_count == position_idx);
    assert(s->normal_count == normal_idx);
    assert(s->index_count == index_idx);

    s->soffset = smin;
    s->sextent = smax - smin;

    fclose(file);

    return 0;
}

void free_scene(scene *s)
{
    free(s->indices);
    free(s->normals);
    free(s->positions);
}

#undef BUFFER_SIZE
