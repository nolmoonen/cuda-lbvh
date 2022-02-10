#include <deps/sutil/vec_math.h>
#include "build.h"
#include "bvh.h"
#include "vec_math_helper.h"
#include "util.h"
#include "util.cuh"
#include "cub_helper.h"
#include "cuda_check.h"

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ unsigned int expand_bits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// Calculates a 30-bit Morton code for the given 3D point located
/// within the unit cube [0,1].
__forceinline__ __device__ unsigned int morton_3d(float x, float y, float z)
{
    x = fclampf(x * 1024.f, 0.f, 1023.f);
    y = fclampf(y * 1024.f, 0.f, 1023.f);
    z = fclampf(z * 1024.f, 0.f, 1023.f);
    unsigned int xx = expand_bits((unsigned int) x);
    unsigned int yy = expand_bits((unsigned int) y);
    unsigned int zz = expand_bits((unsigned int) z);
    return xx * 4 + yy * 2 + zz;
}

__global__ void assign_morton(
        bvh bvh, unsigned int *d_morton, unsigned int *d_ids,
        unsigned int object_count)
{
    unsigned int block_id = get_block_id();
    unsigned int thread_id = get_thread_id(block_id);
    if (thread_id >= object_count) return;

    // obtain center of triangle
    unsigned int idx_u = bvh.d_scene.indices[3 * thread_id + 0].x;
    unsigned int idx_v = bvh.d_scene.indices[3 * thread_id + 1].x;
    unsigned int idx_w = bvh.d_scene.indices[3 * thread_id + 2].x;
    float3 pos = (1.f / 3.f) * (
            bvh.d_scene.positions[idx_u] +
            bvh.d_scene.positions[idx_v] +
            bvh.d_scene.positions[idx_w]);

    // normalize position
    float x = (pos.x - bvh.d_scene.soffset.x) / bvh.d_scene.sextent.x;
    float y = (pos.y - bvh.d_scene.soffset.y) / bvh.d_scene.sextent.y;
    float z = (pos.z - bvh.d_scene.soffset.z) / bvh.d_scene.sextent.z;
    // clamp to deal with numeric issues
    x = fclampf(x, 0.f, 1.f);
    y = fclampf(y, 0.f, 1.f);
    z = fclampf(z, 0.f, 1.f);

    // obtain and set morton code based on normalized position
    d_morton[thread_id] = morton_3d(x, y, z);
    d_ids[thread_id] = thread_id;
}

// todo this kernel is pretty small, can it be combined with another?
__global__ void leaf_nodes(
        unsigned int *sorted_object_ids, unsigned int num_objects, bvh bvh)
{
    unsigned int block_id = get_block_id();
    unsigned int thread_id = get_thread_id(block_id);
    if (thread_id >= num_objects) return;

    // no need to set parent to nullptr, each child will have a parent
    bvh.leaf_nodes[thread_id].object_id = sorted_object_ids[thread_id];
    // needed to recognize that this node is a leaf
    bvh.leaf_nodes[thread_id].child_a = nullptr;

    // need to set for internal node parent to nullptr, for testing later
    // there is one less internal node than leaf node, test for that
    if (thread_id >= num_objects - 1) return;
    bvh.internal_nodes[thread_id].parent = nullptr;
}

__forceinline__ __device__ int delta(
        int a, int b, unsigned int n, unsigned int *c, unsigned int ka)
{
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if (b < 0 || b > n - 1) return -1;
    unsigned int kb = c[b];
    if (ka == kb) {
        // if keys are equal, use id as fallback
        // (+32 because they have the same morton code)
        return 32 + __clz((unsigned int) a ^ (unsigned int) b);
    }
    // clz = count leading zeros
    return __clz(ka ^ kb);
}

__forceinline__ __device__ int2 determine_range(
        unsigned int *sorted_morton_codes, unsigned int n, int i)
{
    unsigned int *c = sorted_morton_codes;
    unsigned int ki = c[i]; // key of i

    // determine direction of the range (+1 or -1)
    const int delta_l = delta(i, i - 1, n, c, ki);
    const int delta_r = delta(i, i + 1, n, c, ki);

    int d; // direction
    int delta_min; // min of delta_r and delta_l
    if (delta_r < delta_l) {
        d = -1;
        delta_min = delta_r;
    } else {
        d = 1;
        delta_min = delta_l;
    }

    // compute upper bound of the length of the range
    unsigned int l_max = 2;
    while (delta(i, i + l_max * d, n, c, ki) > delta_min) {
        l_max <<= 1;
    }

    // find other end using binary search
    unsigned int l = 0;
    for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * d, n, c, ki) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d;

    // ensure i <= j
    return i < j ? make_int2(i, j) : make_int2(j, i);
}

__forceinline__ __device__ int find_split(
        unsigned int *sorted_morton_codes, int first, int last, unsigned int n)
{
    const unsigned int first_code = sorted_morton_codes[first];

    // calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic

    const int common_prefix =
            delta(first, last, n, sorted_morton_codes, first_code);

    // use binary search to find where the next bit differs
    // specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one

    int split = first; // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        const int new_split = split + step; // proposed new position

        if (new_split < last) {
            const int split_prefix = delta(
                    first, new_split, n, sorted_morton_codes, first_code);
            if (split_prefix > common_prefix) {
                split = new_split; // accept proposal
            }
        }
    } while (step > 1);

    return split;
}

__global__ void internal_nodes(
        unsigned int *sorted_morton_codes, unsigned int *sorted_object_ids,
        unsigned int num_objects, bvh_node *d_leaf_nodes, bvh_node *d_internal_nodes)
{
    unsigned int block_id = get_block_id();
    unsigned int thread_id = get_thread_id(block_id);
    // notice the -1, we want i in range [0, num_objects - 2]
    if (thread_id >= num_objects - 1) return;

    // find out which range of objects the node corresponds to
    const int2 range = determine_range(
            sorted_morton_codes, num_objects, thread_id);

    // determine where to split the range
    const int split = find_split(
            sorted_morton_codes, range.x, range.y, num_objects);

    // select child a
    bvh_node *child_a;
    if (split == range.x) {
        child_a = &d_leaf_nodes[split];
    } else {
        child_a = &d_internal_nodes[split];
    }

    // select child b
    bvh_node *child_b;
    if (split + 1 == range.y) {
        child_b = &d_leaf_nodes[split + 1];
    } else {
        child_b = &d_internal_nodes[split + 1];
    }

    // record parent-child relationships
    d_internal_nodes[thread_id].child_a = child_a;
    d_internal_nodes[thread_id].child_b = child_b;
    d_internal_nodes[thread_id].visited = 0;
    child_a->parent = &d_internal_nodes[thread_id];
    child_b->parent = &d_internal_nodes[thread_id];
}

__global__ void set_aabb(
        unsigned int num_objects, bvh_node *d_leaf_nodes,
        bvh_node *d_internal_nodes, bvh bvh)
{
    unsigned int block_id = get_block_id();
    unsigned int thread_id = get_thread_id(block_id);
    if (thread_id >= num_objects) return;

    const unsigned int object_id = d_leaf_nodes[thread_id].object_id;

    unsigned int idx_u = bvh.d_scene.indices[3 * object_id + 0].x;
    unsigned int idx_v = bvh.d_scene.indices[3 * object_id + 1].x;
    unsigned int idx_w = bvh.d_scene.indices[3 * object_id + 2].x;

    float3 u = bvh.d_scene.positions[idx_u];
    float3 v = bvh.d_scene.positions[idx_v];
    float3 w = bvh.d_scene.positions[idx_w];

    // set bounding box of leaf node
    d_leaf_nodes[thread_id].min = fminf(u, fminf(v, w));
    d_leaf_nodes[thread_id].max = fmaxf(u, fmaxf(v, w));

    // recursively set tree bounding boxes
    // {current_node} is always an internal node (since it is parent of another)
    bvh_node *current_node = d_leaf_nodes[thread_id].parent;
    while (true) {
        // we have reached the parent of the root node: terminate
        if (current_node == nullptr) break;

        // we have reached an inner node: check whether the node was visited
        unsigned int visited = atomicAdd(&current_node->visited, 1);

        // this is the first thread entering: terminate
        if (visited == 0) break;

        // this is the second thread entering, we know that our sibling has
        // reached the current node and terminated,
        // and hence the sibling bounding box is correct

        // set running bounding box to be the union of bounding boxes
        current_node->min = fminf(
                current_node->child_a->min, current_node->child_b->min);
        current_node->max = fmaxf(
                current_node->child_a->max, current_node->child_b->max);

        // continue traversal
        current_node = current_node->parent;
    }
}

int build(scene *s, bvh *bvh)
{
    uint triangle_count = s->index_count / 3;
    // must have at least two triangles. we cannot build a bvh for zero
    // triangles, and a bvh of one triangle has no internal nodes
    // which requires special handling which we forgo
    if (triangle_count <= 1) {
        fprintf(stderr, "too few triangles in scene: %d", triangle_count);
        return -1;
    }

    // events for measuring elapsed time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

#define BLOCK_SIZE 1024
    dim3 block_count(ROUND_UP(triangle_count, BLOCK_SIZE), 1, 1);
    uint block_size(BLOCK_SIZE);

    bvh->d_scene = *s;
    // copy scene to device
    CUDA_CHECK(cudaMalloc(
            &bvh->d_scene.positions, sizeof(float3) * s->position_count));
    CUDA_CHECK(cudaMemcpy(
            bvh->d_scene.positions, s->positions,
            sizeof(float3) * s->position_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(
            &bvh->d_scene.normals, sizeof(float3) * s->normal_count));
    CUDA_CHECK(cudaMemcpy(
            bvh->d_scene.normals, s->normals,
            sizeof(float3) * s->normal_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(
            &bvh->d_scene.indices, sizeof(uint2) * s->index_count));
    CUDA_CHECK(cudaMemcpy(
            bvh->d_scene.indices, s->indices,
            sizeof(uint2) * s->index_count, cudaMemcpyHostToDevice));

    // allocate array for morton and ids in dimension of triangles
    unsigned int *d_morton = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_morton),
            sizeof(unsigned int) * triangle_count));
    unsigned int *d_ids = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_ids),
            sizeof(unsigned int) * triangle_count));

    assign_morton<<<block_count, block_size>>>(
            *bvh, d_morton, d_ids, triangle_count);
    CUDA_SYNC_CHECK();

    // sort the key, value pairs according to morton codes
    unsigned int *d_morton_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_morton_sorted),
            sizeof(unsigned int) * triangle_count));
    unsigned int *d_ids_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_ids_sorted),
            sizeof(unsigned int) * triangle_count));
    // sort is stable (https://groups.google.com/g/cub-users/c/1iXn3sVMEuA)
    radix_sort(triangle_count, d_morton, d_morton_sorted, d_ids, d_ids_sorted);
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ids)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_morton)));

    // allocate BVH (n leaf nodes, n - 1 internal nodes)
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&bvh->leaf_nodes),
            sizeof(bvh_node) * triangle_count));
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&bvh->internal_nodes),
            sizeof(bvh_node) * (triangle_count - 1)));
    // construct leaf nodes
    leaf_nodes<<<block_count, block_size>>>(d_ids_sorted, triangle_count, *bvh);

    CUDA_SYNC_CHECK();

    // construct internal nodes
    internal_nodes<<<block_count, block_size>>>(
            d_morton_sorted, d_ids_sorted, triangle_count,
            bvh->leaf_nodes, bvh->internal_nodes);

    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ids_sorted)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_morton_sorted)));

    // calculate bounding boxes by walking the hierarchy toward the root
    set_aabb<<<block_count, block_size>>>(
            triangle_count, bvh->leaf_nodes, bvh->internal_nodes, *bvh);

    // print elapsed time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("building took %fs\n", milliseconds * 1e-3f);

    CUDA_SYNC_CHECK();

    return 0;
#undef BLOCK_SIZE
}

void clean(bvh *bvh)
{
    // free bvh
    CUDA_CHECK(cudaFree(bvh->leaf_nodes));
    CUDA_CHECK(cudaFree(bvh->internal_nodes));

    // free device scene
    CUDA_CHECK(cudaFree(bvh->d_scene.indices));
    CUDA_CHECK(cudaFree(bvh->d_scene.normals));
    CUDA_CHECK(cudaFree(bvh->d_scene.positions));
}