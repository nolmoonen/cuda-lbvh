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
