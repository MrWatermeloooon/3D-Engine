#pragma once

#include <glm/glm.hpp>
#include <array>

// View-frustum extracted from a view-projection matrix. Planes point inward.
struct Frustum {
    std::array<glm::vec4, 6> planes;   // each plane: ax + by + cz + d = 0, normalized
};

// Build planes from view-proj. Works with any clip-space convention.
Frustum extractFrustum(const glm::mat4& viewProj);

// True if AABB is at least partially inside the frustum.
bool aabbInFrustum(const Frustum& f, const glm::vec3& mn, const glm::vec3& mx);

// World-space AABB enclosing the model-space AABB after applying `model`.
void transformAabb(const glm::mat4& model,
                   const glm::vec3& localMin, const glm::vec3& localMax,
                   glm::vec3& outMin, glm::vec3& outMax);
