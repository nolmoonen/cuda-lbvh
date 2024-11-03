#include <cuda_runtime.h>
#include <stdio.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <sutil/vec_math.h>
#include <sutil/random.h>

#include "vec_math_helper.h"
#include "camera.h"
#include "trace.h"
#include "bvh.h"
#include "cuda_check.h"

/// Returns true if the ray intersects with the bounding box.
/// Ray Tracing Gems 2, Chapter 2: Ray Axis-Aligned Bounding Box Intersection
__forceinline__ __device__ bool hit_aabb(
        bvh_node *child, const float3 &origin, const float3 &inv_dir, float length)
{
    float3 t_lower = (child->min - origin) * inv_dir;
    float3 t_upper = (child->max - origin) * inv_dir;

    float3 tmin = fminf(t_lower, t_upper);
    float3 tmax = fmaxf(t_lower, t_upper);

    // use three comparisons for both min and max, seven in total
    return fmaxf4(0.f, tmin) <= fminf4(length, tmax);
}

/// Returns the length along the ray as well as the barycentric coords (u, v).
/// https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
__forceinline__ __device__ float3 get_hit(
        bvh_node *child, float3 origin, float3 direction, bvh bvh)
{
    // get triangle position indices
    unsigned int v0_idx = bvh.d_scene.indices[3 * child->object_id + 0].x;
    unsigned int v1_idx = bvh.d_scene.indices[3 * child->object_id + 1].x;
    unsigned int v2_idx = bvh.d_scene.indices[3 * child->object_id + 2].x;

    // get triangle positions
    float3 v0 = bvh.d_scene.positions[v0_idx];
    float3 v1 = bvh.d_scene.positions[v1_idx];
    float3 v2 = bvh.d_scene.positions[v2_idx];

    // calculate the intersection
    float3 v1v0 = v1 - v0;
    float3 v2v0 = v2 - v0;
    float3 rov0 = origin - v0;
    // winding determines face normal
    float3 n = cross(v1v0, v2v0);
    float3 q = cross(rov0, direction);
    float d = 1.f / dot(direction, n);
    float u = d * dot(-q, v2v0);
    float v = d * dot(q, v1v0);
    float t = d * dot(-n, rov0);
    if (u < 0.f || v < 0.f || (u + v) > 1.f) t = -1.f;

    return make_float3(t, u, v);
}

/// Returns the hit with the bounding box.
__forceinline__ __device__ hit traverse(
        bvh bvh, float3 origin, float3 direction, float tmin, float tmax)
{
    // inverse direction for faster aabb test
    float3 inv_dir = 1.f / direction;

    // allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes
    bvh_node *stack[64];
    bvh_node **stack_ptr = stack;
    *stack_ptr++ = NULL;

    // closest hit, u and v
    float3 closest = make_float3(tmax, 0.f, 0.f);
    // id for that closest hit
    unsigned int object_id;

    // traverse nodes starting from the root, which is the first internal node
    bvh_node *curr = bvh.internal_nodes;
    do {
        // check each child node for overlap.
        bvh_node *child_l = curr->child_a;
        bvh_node *child_r = curr->child_b;
        bool hit_l = hit_aabb(child_l, origin, inv_dir, tmax);
        bool hit_r = hit_aabb(child_r, origin, inv_dir, tmax);

        // query overlaps a leaf node => report collision
        if (hit_l && child_l->is_leaf()) {
            float3 hit = get_hit(child_l, origin, direction, bvh);
            if (hit.x > tmin && hit.x < closest.x) {
                closest = hit;
                object_id = child_l->object_id;
            }
        }

        if (hit_r && child_r->is_leaf()) {
            float3 hit = get_hit(child_r, origin, direction, bvh);
            if (hit.x > tmin && hit.x < closest.x) {
                closest = hit;
                object_id = child_r->object_id;
            }
        }

        // query overlaps an internal node => traverse
        bool traverse_l = (hit_l && !child_l->is_leaf());
        bool traverse_r = (hit_r && !child_r->is_leaf());

        if (!traverse_l && !traverse_r) {
            curr = *--stack_ptr; // pop
        } else {
            curr = (traverse_l) ? child_l : child_r;
            if (traverse_l && traverse_r) {
                *stack_ptr++ = child_r; // push
            }
        }
    } while (curr != NULL);

    hit h;
    if (closest.x < tmax) {
        h.hitpoint = origin + closest.x * direction;
        h.hit = true;

        // get triangle normal indices
        unsigned int v0_idx = bvh.d_scene.indices[3 * object_id + 0].y;
        unsigned int v1_idx = bvh.d_scene.indices[3 * object_id + 1].y;
        unsigned int v2_idx = bvh.d_scene.indices[3 * object_id + 2].y;

        // get triangle normals
        float3 n0 = bvh.d_scene.normals[v0_idx];
        float3 n1 = bvh.d_scene.normals[v1_idx];
        float3 n2 = bvh.d_scene.normals[v2_idx];

        // use barycentric coords to interpolate normal
        h.normal = normalize(
                n0 * (1.f - closest.y - closest.z) + n1 * closest.y +
                n2 * closest.z);
    } else {
        // no hit
        h.hit = false;
    }

    return h;
}

/// Generates a radiance value for the ith sample of this pixel.
__forceinline__ __device__ float3 generate_pixel(
        uint image_idx, uint image_idx_x, uint image_idx_y, uint sample_idx,
        uint size_x, uint size_y, camera *camera, bvh bvh)
{
    // initialize random based on sample index and image index
    uint seed = tea<16>(image_idx, sample_idx);

    // generate a ray though the pixel, randomly offset within the pixel
    float2 jitter = make_float2(rnd(seed), rnd(seed));
    float2 res = make_float2(size_x, size_y);
    float2 idx = make_float2(image_idx_x, image_idx_y);
    float2 d = ((idx + jitter) / res) * 2.f - 1.f; // position on raster
    float3 ray_origin = camera->origin;
    float3 ray_direction = normalize(
            d.x * camera->u + d.y * camera->v + camera->w);

    float3 throughput = make_float3(1.f);
    float3 radiance = make_float3(0.f);

    // keep bounding until the maximum number of bounces is hit,
    // or the ray does not intersect with the sdf
    for (int i = 0; i < MAX_BOUNCE_COUNT; i++) {
        hit h = traverse(bvh, ray_origin, ray_direction, 1e-4f, 1e16f);

        if (!h.hit) {
            // 'sky' color
            const float3 color = make_float3(.6, .8f, 1.f);
            radiance += throughput * color;
            break;
        }

        // pick a static diffuse color
        float3 diff_color = make_float3(.9f);

        // check if we continue using russian roulette, where the max component
        // of the color (with some maximum value) dictates the probability
        if (i > MIN_BOUNCE_COUNT) {
            float rr_prob = fminf(fmaxf(diff_color), .95f);
            if (rr_prob < rnd(seed)) break;
            // if continuing, scale with probability
            throughput /= rr_prob;
        }

        // surface model is lambertian, attenuation is equal to diffuse
        // color, assuming we sampled with cosine weighted hemisphere
        throughput *= diff_color;

        // set new origin and generate new direction
        ray_origin = h.hitpoint;
        float3 w_in = cosine_sample_hemisphere(rnd(seed), rnd(seed));
        frame onb(h.normal);
        onb.inverse_transform(w_in);
        ray_direction = w_in;
    }

    return radiance;
}

/// Implementation with regeneration: create a number of persistent threads that
/// complete samples one by one, starting new ones when the current one is
/// terminated.
__global__ void generate_pixel_regeneration(
        uint size_x, uint size_y, uint sample_count, float *buffer,
        camera *camera, unsigned long long int *idx, bvh bvh)
{
    const ulong max_count = size_x * size_y * sample_count;
    while (true) {
        // obtain the next index. if is it out of bounds, stop
        unsigned long long int this_idx = atomicAdd(idx, 1);
        if (this_idx >= max_count) break;

        uint sample_idx = this_idx / (size_x * size_y);
        uint image_idx = this_idx - sample_idx * size_x * size_y;
        uint image_idx_y = image_idx / size_x;
        uint image_idx_x = image_idx - image_idx_y * size_x;

        // obtain radiance
        float3 radiance = generate_pixel(
                image_idx, image_idx_x, image_idx_y, sample_idx,
                size_x, size_y, camera, bvh);

        // atomically add to buffer
        atomicAdd(&buffer[4 * image_idx + 0], radiance.x / float(sample_count));
        atomicAdd(&buffer[4 * image_idx + 1], radiance.y / float(sample_count));
        atomicAdd(&buffer[4 * image_idx + 2], radiance.z / float(sample_count));
    }
}

/// Converts a linear radiance value to a sRGB pixel value.
uchar radiance_to_srgb(float val)
{ return (uchar) (clamp(powf(val, 1.f / 2.4f), 0.f, 1.f) * 255.f); }

void generate(
        uint size_x, uint size_y, uint sample_count, uchar *image,
        bvh bvh, float3 origin, float3 target, float3 up)
{
    // events for measuring elapsed time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // initialize camera based on the passed in variables
    float aspect = float(size_x) / float(size_y);
    camera cam;
    cam.origin = origin;
    cam.w = normalize(target - cam.origin);       // lookat direction
    cam.u = normalize(cross(cam.w, up)) * aspect; // screen right
    cam.v = normalize(cross(cam.u, cam.w));       // screen up

    // copy camera parameters to device
    camera *d_cam = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cam, sizeof(camera)));
    CUDA_CHECK(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));

    // create output buffer on device
    float *d_buffer = nullptr;
    size_t buffer_size = sizeof(float) * 4 * size_x * size_y;
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    CUDA_CHECK(cudaMemset(d_buffer, 0, buffer_size));

    // additionally, allocate a single long int counter
    unsigned long long int *d_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_idx, sizeof(unsigned long long int)));
    CUDA_CHECK(cudaMemset(d_idx, 0, sizeof(unsigned long long int)));

    // launch kernel
    generate_pixel_regeneration<<<1024, 1024>>>(
            size_x, size_y, sample_count, d_buffer, d_cam, d_idx, bvh);

    // when kernel is done, copy buffer back to host
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_cam));
    float *buffer = (float *) malloc(buffer_size);
    CUDA_CHECK(cudaMemcpy(
            buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buffer));

    // convert buffer to format accepted by image writer
    for (uint i = 0; i < size_x * size_y; i++) {
        image[3 * i + 0] = radiance_to_srgb(buffer[4 * i + 0]);
        image[3 * i + 1] = radiance_to_srgb(buffer[4 * i + 1]);
        image[3 * i + 2] = radiance_to_srgb(buffer[4 * i + 2]);
    }
    free(buffer);

    // print elapsed time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("tracing took %fs\n", milliseconds * 1e-3f);
}
