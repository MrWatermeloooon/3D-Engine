#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <cstdint>

struct MeshHandle    { uint32_t id = 0; };
struct TextureHandle { uint32_t id = 0; };

struct TransformComponent {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};

    glm::mat4 getMatrix() const {
        glm::mat4 m(1.0f);
        m = glm::translate(m, position);
        m = glm::rotate(m, glm::radians(rotation.x), {1, 0, 0});
        m = glm::rotate(m, glm::radians(rotation.y), {0, 1, 0});
        m = glm::rotate(m, glm::radians(rotation.z), {0, 0, 1});
        m = glm::scale(m, scale);
        return m;
    }
};

struct MeshComponent {
    MeshHandle handle;
};

struct MaterialComponent {
    TextureHandle texture;
    glm::vec4 color{1.0f};
    float metallic  = 0.0f;
    float roughness = 0.5f;
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
};
