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

#pragma once

#include "obj_reader.h"

struct hit {
    float3 hitpoint;
    /// Whether a hit is recorded.
    int hit;
    float3 normal;
};

struct bvh_node {
    // If `-1`, then this node is a leaf.
    int child_l;
    int child_r;
    int paren;
    /** Bounding box. */
    float3 min;
    float3 max;

    union {
        /// If leaf, holds the id of the object.
        unsigned int object_id;
        /// If internal node, holds whether the node has been visited once
        /// while setting bounding boxes. The first thread (child) sets
        /// it equal to its own bounding box and continues up the tree.
        /// The second thread (child) sets it to their union and terminates.
        unsigned int visited;
    };

    __forceinline__ __device__ bool is_leaf() const { return child_l == -1; }
};

struct bvh {
    // First N - 1 internal nodes.
    // Followed by N leaf nodes, one for every triangle.
    buf_gpu<bvh_node> nodes;
    // See `scene`.
    buf_gpu<float3> positions;
    buf_gpu<float3> normals;
    buf_gpu<int> pos_indices;
    buf_gpu<int> nor_indices;
};
