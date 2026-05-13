#include "frustum.h"

Frustum extractFrustum(const glm::mat4& m) {
    Frustum f;
    // Rows of the view-proj matrix; planes are linear combinations of rows.
    // Using GLM column-major: m[col][row]. To get row i: vec4(m[0][i], m[1][i], m[2][i], m[3][i]).
    auto row = [&](int r) {
        return glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]);
    };
    glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    f.planes[0] = r3 + r0; // left
    f.planes[1] = r3 - r0; // right
    f.planes[2] = r3 + r1; // bottom (or top, depending on Y convention)
    f.planes[3] = r3 - r1; // top
    // Vulkan / GLM_FORCE_DEPTH_ZERO_TO_ONE clip space: z ∈ [0, w].
    // Near plane equation is `z ≥ 0` (NOT `z + w ≥ 0` like OpenGL).
    f.planes[4] = r2;      // near
    f.planes[5] = r3 - r2; // far

    for (auto& p : f.planes) {
        float len = glm::length(glm::vec3(p));
        if (len > 1e-6f) p /= len;
    }
    return f;
}

bool aabbInFrustum(const Frustum& f, const glm::vec3& mn, const glm::vec3& mx) {
    for (const auto& p : f.planes) {
        // Pick the AABB corner farthest along the plane's positive normal.
        glm::vec3 pos(
            p.x >= 0.0f ? mx.x : mn.x,
            p.y >= 0.0f ? mx.y : mn.y,
            p.z >= 0.0f ? mx.z : mn.z);
        if (glm::dot(glm::vec3(p), pos) + p.w < 0.0f) {
            return false; // entirely on negative side of this plane
        }
    }
    return true;
}

void transformAabb(const glm::mat4& model,
                   const glm::vec3& localMin, const glm::vec3& localMax,
                   glm::vec3& outMin, glm::vec3& outMax)
{
    // Method of Graphics Gems: project the local AABB through M to get a tight
    // world-space AABB using only the translation column and abs(rotation*scale).
    glm::vec3 center = (localMin + localMax) * 0.5f;
    glm::vec3 extent = (localMax - localMin) * 0.5f;

    glm::vec3 worldCenter = glm::vec3(model * glm::vec4(center, 1.0f));

    glm::mat3 absM(
        glm::abs(glm::vec3(model[0])),
        glm::abs(glm::vec3(model[1])),
        glm::abs(glm::vec3(model[2])));
    glm::vec3 worldExtent = absM * extent;

    outMin = worldCenter - worldExtent;
    outMax = worldCenter + worldExtent;
}
