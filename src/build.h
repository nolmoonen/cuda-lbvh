#pragma once

#include "bvh.h"
#include "obj_reader.h"

/// Builds a BVH on the device based on the input scene. Scene must have at
/// at least two triangles.
/// Returns 0 on success.
int build(scene *s, bvh *bvh);

void clean(bvh *bvh);