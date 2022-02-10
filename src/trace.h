#pragma once

#include "camera.h"
#include "obj_reader.h"
#include "util.h"
#include "bvh.h"

/// Number of ray bounces.
#define MIN_BOUNCE_COUNT 0
#define MAX_BOUNCE_COUNT 80

void generate(
        uint size_x, uint size_y, uint sample_count, uchar *image,
        bvh bvh, float3 origin, float3 target, float3 up);
