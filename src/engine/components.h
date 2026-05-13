#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <cstdint>
#include <limits>

struct MeshHandle    { uint32_t id = 0; };
struct TextureHandle { uint32_t id = 0; };
struct LODGroupHandle { uint32_t id = 0xFFFFFFFFu; }; // sentinel = "no LOD group"

struct TransformComponent {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};

    // Cached world matrix + normal matrix. Value-based invalidation: getMatrix()
    // compares the live fields against the snapshot used when the cache was
    // populated, and only recomputes on mismatch. No setter wrapping is needed —
    // direct writes (ImGui drags, ImGuizmo, scene rotators) are picked up on
    // the next getMatrix() call. The matrix is returned by const-reference so
    // hot paths (renderer.cpp candidate build, shadow draw, RT instance gather)
    // pay one float-compare per call on a hit. Cached state is `mutable` so
    // `const`-correct callers (which TransformComponent is read by everywhere)
    // can still warm the cache.
    const glm::mat4& getMatrix() const {
        if (position != m_cachedPos
         || rotation != m_cachedRot
         || scale    != m_cachedScale)
        {
            glm::mat4 m(1.0f);
            m = glm::translate(m, position);
            m = glm::rotate(m, glm::radians(rotation.x), {1, 0, 0});
            m = glm::rotate(m, glm::radians(rotation.y), {0, 1, 0});
            m = glm::rotate(m, glm::radians(rotation.z), {0, 0, 1});
            m = glm::scale(m, scale);
            m_cachedMatrix = m;

            // Normal matrix: transpose(inverse(mat3(model))). Guard against
            // singular transforms (zero scale or degenerate). glm::inverse on a
            // det≈0 matrix yields NaN, which propagates into shading.
            glm::mat3 m3  = glm::mat3(m);
            float     det = glm::determinant(m3);
            m_cachedNormalMatrix = (glm::abs(det) > 1e-6f)
                ? glm::transpose(glm::inverse(m3))
                : glm::mat3(1.0f);

            m_cachedPos   = position;
            m_cachedRot   = rotation;
            m_cachedScale = scale;
        }
        return m_cachedMatrix;
    }

    const glm::mat3& getNormalMatrix() const {
        getMatrix();  // warm the cache; cheap on a hit.
        return m_cachedNormalMatrix;
    }

private:
    // Sentinel: a NaN guarantees the first comparison fails (NaN != NaN) so the
    // first getMatrix() call always recomputes — even if the user happens to
    // construct the component with position == 0.
    mutable glm::vec3 m_cachedPos   { std::numeric_limits<float>::quiet_NaN() };
    mutable glm::vec3 m_cachedRot   { 0.0f };
    mutable glm::vec3 m_cachedScale { 0.0f };
    mutable glm::mat4 m_cachedMatrix       { 1.0f };
    mutable glm::mat3 m_cachedNormalMatrix { 1.0f };
};

struct MeshComponent {
    MeshHandle handle;
};

// Optional override: when present, the renderer uses the LOD group's per-level
// meshes (selected by view distance in the cull compute shader) instead of
// MeshComponent.handle. The MeshComponent's handle is still used as the
// shadow-pass mesh (so silhouettes stay sharp regardless of camera distance).
struct MeshLODComponent {
    LODGroupHandle group{};
};

struct MaterialComponent {
    TextureHandle texture;
    // Optional tangent-space normal map. Sentinel id==0 means "no normal map"
    // — the shader falls through to the interpolated vertex normal.
    TextureHandle normalTexture;
    // Optional grayscale height map for parallax-occlusion mapping. id==0
    // disables parallax; the fragment shader bypasses the step march entirely
    // when the slot is zero, so this is free for materials that don't use it.
    TextureHandle heightTexture;
    glm::vec4 color{1.0f};
    float metallic  = 0.0f;
    float roughness = 0.5f;
    // Tangent-space depth used by the parallax step march. 0.02–0.08 reads as
    // subtle relief; higher values look stylised / wavy. Stored in world-unit
    // (well, UV-unit) scale, so adjust per-asset.
    float parallaxScale = 0.05f;
};

struct NameComponent {
    std::string name;
};

struct RotatorComponent {
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    float speed = 45.0f;
};

// ── Lights ──────────────────────────────────────────────────────────────────

struct DirectionalLightComponent {
    glm::vec3 direction{-0.4f, -0.8f, -0.4f};
    glm::vec3 color{1.0f, 0.96f, 0.9f};
    float intensity = 3.0f;
    bool castsShadows = true;
};

struct PointLightComponent {
    // Position from TransformComponent.
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 5.0f;
    float range = 10.0f;
};

struct SpotLightComponent {
    // Position from TransformComponent.
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 8.0f;
    float range = 15.0f;
    float innerConeDeg = 20.0f;
    float outerConeDeg = 30.0f;
};

// Marker — engine renders any entity with this component using its single
// shared skinned mesh resource. (Multi-skinned-mesh support is a future extension.)
struct SkinnedMeshComponent {};

struct AnimatorComponent {
    int   animationIndex = 0;
    float time           = 0.0f;
    float speed          = 1.0f;
    bool  playing        = true;
    bool  loop           = true;  // when false, holds the last pose at duration
};
