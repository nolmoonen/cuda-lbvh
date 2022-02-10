#pragma once

#include "obj_reader.h"

struct hit {
    float3 hitpoint;
    /// Whether a hit is recorded.
    int hit;
    float3 normal;
};

struct bvh_node {
    /// If nullptr, then this node is a leaf.
    bvh_node *child_a;
    bvh_node *child_b;
    bvh_node *parent;
    /** Bounding box. */
    float3 min;
    float3 max;

    /// If leaf, holds the id of the object.
    unsigned int object_id;
    /// If internal node, holds whether the node has been visited once
    /// while setting bounding boxes. The first thread (child) sets
    /// it equal to its own bounding box and continues up the tree.
    /// The second thread (child) sets it to their union and terminates.
    unsigned int visited;

    __forceinline__ __device__ bool is_leaf() const
    { return child_a == nullptr; }
};

struct bvh {
    /// Leaf nodes, one for every photon data element.
    bvh_node *leaf_nodes;
    /// Internal nodes, amount equal to #leaf nodes - 1.
    bvh_node *internal_nodes;
    /// Scene, but all arrays are allocated on device.
    scene d_scene;
};