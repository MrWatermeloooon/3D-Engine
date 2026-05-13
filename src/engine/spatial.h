#pragma once

// SpatialIndex — per-frame binary BVH for CPU pre-culling.
//
// Workflow:
//   1. clear() at the start of the frame.
//   2. insert(entity, worldMin, worldMax) for every drawable.
//   3. build() to construct a top-down median-split BVH over the insertions.
//   4. queryFrustum(frustum, out) to gather visible entries.
//
// Per-frame rebuild was the right choice given Phase 1.1: TransformComponent's
// dirty cache is value-based (no setter callbacks), so an incremental octree
// would need a separate "moved this frame" channel. A full rebuild every frame
// is O(N log N) on the BVH side and reuses the same view walk the renderer
// already had to do anyway. The win lands in the *heavy* per-entity work
// (LOD resolution, batch keying, candidate write) which only runs on the
// visible subset after the query.
//
// The BVH is binary, not strictly an octree. Loose-octree spatial subdivision
// is unnecessary when we rebuild from scratch each frame — entity-list median
// splits adapt to whatever distribution the current scene happens to have.

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <entt/entt.hpp>

struct Frustum;

class SpatialIndex {
public:
    struct Entry {
        entt::entity entity;
        glm::vec3    wMin;
        glm::vec3    wMax;
        // Opaque payload preserved through BVH build/query. Used by the
        // renderer to index back into a parallel "resolved entity" array
        // populated during the same walk that builds the BVH — avoids a
        // second registry lookup pass after the frustum cull.
        uint32_t     userIndex = 0;
    };

    // Drop all entries; keep capacity to avoid per-frame alloc thrash.
    void clear();
    void reserve(size_t n);
    void insert(entt::entity e, const glm::vec3& wMin, const glm::vec3& wMax);

    // Bulk-insert API for parallel populate. Resize the entry array up front,
    // then have workers fill slots independently. After all writes complete,
    // call build() exactly as for sequential insert().
    void resizeForBulkInsert(size_t n);
    Entry& mutableEntry(size_t i) { return m_entries[i]; }

    // Builds the BVH from inserted entries. Reorders the entry vector by
    // median split, so callers must not assume insertion order is preserved.
    void build();

    // Walks the BVH and emits visible entries to `out`. `out` is cleared first.
    // Falls back to a linear scan when entry count is below a small threshold —
    // the BVH overhead dominates at low N.
    void queryFrustum(const Frustum& f, std::vector<Entry>& out) const;

    size_t entryCount() const { return m_entries.size(); }
    size_t nodeCount()  const { return m_nodes.size(); }

private:
    // Leaves hold a contiguous run of entries [first, first+count); internal
    // nodes have count==0 and left/right child indices. `aabbMin/Max` covers
    // the union of all descendants either way.
    struct Node {
        glm::vec3 aabbMin{ 0.0f };
        glm::vec3 aabbMax{ 0.0f };
        int       first = 0;
        int       count = 0;
        int       left  = -1;
        int       right = -1;
    };

    static constexpr int LEAF_CAP   = 8;     // stop splitting at ≤8 entries
    static constexpr int MAX_DEPTH  = 32;
    static constexpr int LINEAR_CUTOFF = 64; // below this, skip the BVH and linear-scan

    std::vector<Entry> m_entries;
    std::vector<Node>  m_nodes;

    int  buildRecursive(int first, int count, int depth);
    void computeBounds(int first, int count, glm::vec3& mn, glm::vec3& mx) const;
};
