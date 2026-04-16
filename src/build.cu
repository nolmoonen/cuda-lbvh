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
#include "bvh.h"
#include "util.h"
#include "vec_math_helper.h"

#include <cassert>
#include <cub/device/device_radix_sort.cuh>
#include <sutil/vec_math.h>

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__device__ unsigned int expand_bits(unsigned int v)
{
    assert(v < 1 << 10);
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// Calculates a 30-bit Morton code for the given 3D point located
/// within the unit cube [0,1].
__device__ unsigned int morton_3d(float x, float y, float z)
{
    x               = fclampf(x * 1024.f, 0.f, 1023.f);
    y               = fclampf(y * 1024.f, 0.f, 1023.f);
    z               = fclampf(z * 1024.f, 0.f, 1023.f);
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

// For every triangle, assign a Morton code based on its center.
__global__ void assign_morton(
    const float3* positions,
    const uint2* indices,
    float3 scene_offset,
    float3 scene_extent,
    unsigned int* d_morton,
    unsigned int* d_ids,
    unsigned int object_count)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= object_count) return;

    // obtain center of triangle
    unsigned int idx_u = indices[3 * thread_id + 0].x;
    unsigned int idx_v = indices[3 * thread_id + 1].x;
    unsigned int idx_w = indices[3 * thread_id + 2].x;
    float3 pos         = (1.f / 3.f) * (positions[idx_u] + positions[idx_v] + positions[idx_w]);

    // normalize position
    float x = (pos.x - scene_offset.x) / scene_extent.x;
    float y = (pos.y - scene_offset.y) / scene_extent.y;
    float z = (pos.z - scene_offset.z) / scene_extent.z;
    // clamp to deal with numeric issues
    x = fclampf(x, 0.f, 1.f);
    y = fclampf(y, 0.f, 1.f);
    z = fclampf(z, 0.f, 1.f);

    // obtain and set morton code based on normalized position
    d_morton[thread_id] = morton_3d(x, y, z);
    d_ids[thread_id]    = thread_id;
}

// todo this kernel is pretty small, can it be combined with another?
__global__ void leaf_nodes(
    unsigned int* sorted_object_ids,
    unsigned int num_objects,
    bvh_node* leaf_nodes,
    bvh_node* internal_nodes)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_objects) return;

    // no need to set parent to nullptr, each child will have a parent
    leaf_nodes[thread_id].object_id = sorted_object_ids[thread_id];
    // needed to recognize that this node is a leaf
    leaf_nodes[thread_id].child_a = nullptr;

    // need to set for internal node parent to nullptr, for testing later
    // there is one less internal node than leaf node, test for that
    if (thread_id >= num_objects - 1) return;
    internal_nodes[thread_id].parent = nullptr;
}

__device__ int delta(int a, int b, unsigned int n, unsigned int* c)
{
    assert(0 <= a && a < n);
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if (b < 0 || b >= n) return -1;
    const unsigned int ka = (uint64_t{c[a]} << 32) | a;
    const unsigned int kb = (uint64_t{c[b]} << 32) | b;
    // if (ka == kb) {
    //     // if keys are equal, use id as fallback
    //     // (+32 because they have the same morton code)
    //     return 64 + __clzll((unsigned int)a ^ (unsigned int)b);
    // }
    // clz = count leading zeros
    return __clzll(ka ^ kb);
}

// Build the internal nodes.
__global__ void internal_nodes(
    unsigned int* sorted_morton_codes,
    unsigned int* sorted_object_ids,
    unsigned int num_objects,
    bvh_node* d_leaf_nodes,
    bvh_node* d_internal_nodes)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // N.B., we want i in range [0, num_objects - 1) since every thread sets one internal node.
    if (thread_id >= num_objects - 1) return;

    // find out which range of objects the node corresponds to
    unsigned int n = num_objects;
    int i          = thread_id;

    unsigned int* c = sorted_morton_codes;

    // Determine direction of the range (+1 or -1).
    const int delta_l = delta(i, i - 1, n, c);
    const int delta_r = delta(i, i + 1, n, c);

    // Compute upper bound for the length of the range.
    int d; // direction
    int delta_min; // delta(i, i - d)
    if (delta_r < delta_l) {
        d         = -1;
        delta_min = delta_r; // delta(i, i - (-1)) = delta(i, i + 1)
    } else {
        d         = +1;
        delta_min = delta_l; // delta(i, i - (+1)) = delta(i, i - 1)
    }

    unsigned int l_max = 2;
    while (delta(i, i + l_max * d, n, c) > delta_min) {
        l_max <<= 1;
    }

    // Find the other end using binary search.
    unsigned int l = 0;
    for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * d, n, c) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d;

    // Find the split position using binary search.

    int split;
    // if (c[i] == c[j]) {
    //     split = (i + j) >> 1;
    // } else {
    const int delta_node = delta(i, j, n, c);

    int s = 0;
    for (unsigned int t = (l + 1) >> 1; t > 1; t = (t + 1) >> 1) {
        if (delta(i, i + (s + t) * d, n, c) > delta_node) {
            s += t;
        }
    }
    // t = 1 iteration
    if (delta(i, i + (s + 1) * d, n, c) > delta_node) {
        ++s;
    }

    split = i + s * d + min(d, 0);
    // }

    // Output child pointers
    const int2 range = i < j ? make_int2(i, j) : make_int2(j, i);

    // select child a
    bvh_node* child_a;
    if (split == range.x) {
        child_a = &d_leaf_nodes[split];
    } else {
        child_a = &d_internal_nodes[split];
    }

    // select child b
    bvh_node* child_b;
    if (split + 1 == range.y) {
        child_b = &d_leaf_nodes[split + 1];
    } else {
        child_b = &d_internal_nodes[split + 1];
    }

    // record parent-child relationships
    d_internal_nodes[thread_id].child_a = child_a;
    d_internal_nodes[thread_id].child_b = child_b;
    d_internal_nodes[thread_id].visited = 0;
    child_a->parent                     = &d_internal_nodes[thread_id];
    child_b->parent                     = &d_internal_nodes[thread_id];
}

// Set internal node bounding boxes by traversing the tree from the leaf nodes.
__global__ void set_aabb(
    unsigned int num_objects,
    bvh_node* d_leaf_nodes,
    bvh_node* d_internal_nodes,
    const float3* positions,
    const uint2* indices)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_objects) return;

    const unsigned int object_id = d_leaf_nodes[thread_id].object_id;

    unsigned int idx_u = indices[3 * object_id + 0].x;
    unsigned int idx_v = indices[3 * object_id + 1].x;
    unsigned int idx_w = indices[3 * object_id + 2].x;

    float3 u = positions[idx_u];
    float3 v = positions[idx_v];
    float3 w = positions[idx_w];

    // set bounding box of leaf node
    d_leaf_nodes[thread_id].min = fminf(u, fminf(v, w));
    d_leaf_nodes[thread_id].max = fmaxf(u, fmaxf(v, w));

    // recursively set tree bounding boxes
    // {current_node} is always an internal node (since it is parent of another)
    bvh_node* current_node = d_leaf_nodes[thread_id].parent;
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
        current_node->min = current_node->child_a->min;
        current_node->max = current_node->child_a->max;
        if (current_node->child_b != nullptr) {
            // If the number of nodes is not a power of two, some nodes will not have
            // a right child.
            current_node->min = fminf(current_node->min, current_node->child_b->min);
            current_node->max = fmaxf(current_node->max, current_node->child_b->max);
        }

        // continue traversal
        current_node = current_node->parent;
    }
}

bool build(const scene& s, bvh& bvh)
{
    const int num_triangles = s.indices.get_num_elements() / 3;
    // must have at least two triangles. we cannot build a bvh for zero
    // triangles, and a bvh of one triangle has no internal nodes
    // which requires special handling which we forgo
    if (num_triangles <= 1) {
        fprintf(stderr, "too few triangles in scene: %d", num_triangles);
        return false;
    }

    // allocate array for morton and ids in dimension of triangles
    buf_gpu<unsigned int> d_morton;
    RETURN_IF_FALSE(d_morton.resize(num_triangles));
    buf_gpu<unsigned int> d_ids;
    RETURN_IF_FALSE(d_ids.resize(num_triangles));

    // sorted key, value pairs according to morton codes
    buf_gpu<unsigned int> d_morton_sorted;
    RETURN_IF_FALSE(d_morton_sorted.resize(num_triangles));
    buf_gpu<unsigned int> d_ids_sorted;
    RETURN_IF_FALSE(d_ids_sorted.resize(num_triangles));

    auto sort = [&](void* d_tmp, size_t& tmp_size) {
        // We don't actually need `keys_out` and `values_in` can be a
        // counting iterator, but `DeviceRadixSort` needs both as
        // backing storage.
        return cub::DeviceRadixSort::SortPairs(
            d_tmp,
            tmp_size,
            d_morton.get_ptr(), // keys_in
            d_morton_sorted.get_ptr(), // keys_out
            d_ids.get_ptr(), // values_in
            d_ids_sorted.get_ptr(), // values_out
            num_triangles);
    };

    // Determine temporary device storage requirements.
    size_t num_tmp_bytes = 0;
    RETURN_IF_CUDA_ERR(sort(nullptr, num_tmp_bytes));
    // Allocate temporary storage
    buf_gpu<char> d_tmp;
    RETURN_IF_FALSE(d_tmp.resize(num_tmp_bytes));

    // copy scene to device
    RETURN_IF_FALSE(bvh.positions.resize(s.positions.get_num_elements()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.positions.get_ptr(),
        s.positions.get_ptr(),
        sizeof(float3) * s.positions.get_num_elements(),
        cudaMemcpyHostToDevice));
    RETURN_IF_FALSE(bvh.normals.resize(s.normals.get_num_elements()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.normals.get_ptr(),
        s.normals.get_ptr(),
        sizeof(float3) * s.normals.get_num_elements(),
        cudaMemcpyHostToDevice));
    RETURN_IF_FALSE(bvh.indices.resize(s.indices.get_num_elements()));
    RETURN_IF_CUDA_ERR(cudaMemcpy(
        bvh.indices.get_ptr(),
        s.indices.get_ptr(),
        sizeof(uint2) * s.indices.get_num_elements(),
        cudaMemcpyHostToDevice));

    // allocate BVH (n leaf nodes, n - 1 internal nodes)
    RETURN_IF_FALSE(bvh.leaf_nodes.resize(num_triangles));
    RETURN_IF_FALSE(bvh.internal_nodes.resize(num_triangles - 1));

    // events for measuring elapsed time
    cudaEvent_t start, stop;
    RETURN_IF_CUDA_ERR(cudaEventCreate(&start));
    RETURN_IF_CUDA_ERR(cudaEventCreate(&stop));
    RETURN_IF_CUDA_ERR(cudaEventRecord(start));

    const int block_size = 1024;
    const int num_blocks = ceiling_div(num_triangles, static_cast<unsigned int>(block_size));

    assign_morton<<<num_blocks, block_size>>>(
        bvh.positions.get_ptr(),
        bvh.indices.get_ptr(),
        s.soffset,
        s.sextent,
        d_morton.get_ptr(),
        d_ids.get_ptr(),
        num_triangles);
    RETURN_IF_CUDA_ERR(cudaGetLastError());

    // Run sorting operation, sorting is stable.
    // https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html
    RETURN_IF_CUDA_ERR(sort(d_tmp.get_ptr(), num_tmp_bytes));

    // construct leaf nodes
    leaf_nodes<<<num_blocks, block_size>>>(
        d_ids_sorted.get_ptr(),
        num_triangles,
        bvh.leaf_nodes.get_ptr(),
        bvh.internal_nodes.get_ptr());
    RETURN_IF_CUDA_ERR(cudaGetLastError());

    // construct internal nodes
    internal_nodes<<<num_blocks, block_size>>>(
        d_morton_sorted.get_ptr(),
        d_ids_sorted.get_ptr(),
        num_triangles,
        bvh.leaf_nodes.get_ptr(),
        bvh.internal_nodes.get_ptr());
    RETURN_IF_CUDA_ERR(cudaGetLastError());

    // calculate bounding boxes by walking the hierarchy toward the root
    set_aabb<<<num_blocks, block_size>>>(
        num_triangles,
        bvh.leaf_nodes.get_ptr(),
        bvh.internal_nodes.get_ptr(),
        bvh.positions.get_ptr(),
        bvh.indices.get_ptr());
    RETURN_IF_CUDA_ERR(cudaGetLastError());

    // print elapsed time
    RETURN_IF_CUDA_ERR(cudaEventRecord(stop));
    RETURN_IF_CUDA_ERR(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    RETURN_IF_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    const float seconds = milliseconds * 1e-3f;
    printf(
        "building took %5.5fs, %5.2f million triangles per second\n",
        seconds,
        num_triangles / seconds * 1e-6f);

    return true;
}
