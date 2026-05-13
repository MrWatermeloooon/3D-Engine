#include "spatial.h"
#include "frustum.h"

#include <algorithm>
#include <limits>

void SpatialIndex::clear() {
    m_entries.clear();
    m_nodes.clear();
}

void SpatialIndex::reserve(size_t n) {
    m_entries.reserve(n);
    // ~2N nodes for a binary tree with leaf cap 8; cheap upper bound.
    m_nodes.reserve(std::max<size_t>(8, n / 4 + 1));
}

void SpatialIndex::insert(entt::entity e,
                          const glm::vec3& wMin, const glm::vec3& wMax)
{
    m_entries.push_back({ e, wMin, wMax });
}

void SpatialIndex::resizeForBulkInsert(size_t n) {
    m_entries.resize(n);
    m_nodes.clear();
}

void SpatialIndex::computeBounds(int first, int count, glm::vec3& mn, glm::vec3& mx) const {
    mn = glm::vec3( std::numeric_limits<float>::infinity());
    mx = glm::vec3(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < count; ++i) {
        const Entry& e = m_entries[first + i];
        mn = glm::min(mn, e.wMin);
        mx = glm::max(mx, e.wMax);
    }
}

// Top-down median split on the longest axis. The pivot is chosen by
// nth_element on the entry centroids — partial sort, O(N) per level.
int SpatialIndex::buildRecursive(int first, int count, int depth) {
    int  nodeIdx = static_cast<int>(m_nodes.size());
    m_nodes.emplace_back();
    Node& node = m_nodes.back();
    computeBounds(first, count, node.aabbMin, node.aabbMax);

    if (count <= LEAF_CAP || depth >= MAX_DEPTH) {
        node.first = first;
        node.count = count;
        return nodeIdx;
    }

    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x && extent.y >= extent.z) axis = 1;
    else if (extent.z > extent.x)                    axis = 2;

    int mid = count / 2;
    auto centroid = [axis](const Entry& e) {
        return 0.5f * ((&e.wMin.x)[axis] + (&e.wMax.x)[axis]);
    };

    auto begin = m_entries.begin() + first;
    auto endIt = begin + count;
    auto midIt = begin + mid;
    std::nth_element(begin, midIt, endIt,
        [&](const Entry& a, const Entry& b) { return centroid(a) < centroid(b); });

    // Re-take a node reference after possible vector reallocation in recursion.
    int left  = buildRecursive(first,           mid,             depth + 1);
    int right = buildRecursive(first + mid,     count - mid,     depth + 1);

    Node& n2  = m_nodes[nodeIdx];
    n2.left   = left;
    n2.right  = right;
    n2.count  = 0; // mark internal
    return nodeIdx;
}

void SpatialIndex::build() {
    m_nodes.clear();
    if (m_entries.empty()) return;
    buildRecursive(0, static_cast<int>(m_entries.size()), 0);
}

void SpatialIndex::queryFrustum(const Frustum& f, std::vector<Entry>& out) const {
    out.clear();
    if (m_entries.empty()) return;

    // Linear fallback for small N — the BVH traversal pays its overhead at
    // hundreds-of-entities scale, not dozens.
    if (m_entries.size() < LINEAR_CUTOFF || m_nodes.empty()) {
        out.reserve(m_entries.size());
        for (const Entry& e : m_entries) {
            if (aabbInFrustum(f, e.wMin, e.wMax)) out.push_back(e);
        }
        return;
    }

    // Stack traversal. MAX_DEPTH bounds the depth; doubling that as the stack
    // size gives room for both children of every ancestor.
    int stack[MAX_DEPTH * 2];
    int top = 0;
    stack[top++] = 0;

    out.reserve(m_entries.size() / 2);

    while (top > 0) {
        int idx = stack[--top];
        const Node& n = m_nodes[idx];

        if (!aabbInFrustum(f, n.aabbMin, n.aabbMax)) continue;

        if (n.count > 0) {
            // Leaf — emit per-entry. Re-test each entry against the frustum
            // since the node AABB only proved intersection, not containment.
            for (int i = 0; i < n.count; ++i) {
                const Entry& e = m_entries[n.first + i];
                if (aabbInFrustum(f, e.wMin, e.wMax)) out.push_back(e);
            }
        } else {
            if (n.right >= 0) stack[top++] = n.right;
            if (n.left  >= 0) stack[top++] = n.left;
        }
    }
}
