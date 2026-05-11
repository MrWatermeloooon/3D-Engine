#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <array>

#include "buffer.h"

constexpr uint32_t MAX_BONES = 64;

// ── Skeleton ────────────────────────────────────────────────────────────────

struct Joint {
    std::string name;
    int         parent       = -1;       // -1 = root
    glm::mat4   inverseBind  = glm::mat4(1.0f);
    glm::vec3   localT       = glm::vec3(0.0f);
    glm::quat   localR       = glm::quat(1, 0, 0, 0);
    glm::vec3   localS       = glm::vec3(1.0f);
};

struct Skeleton {
    std::vector<Joint> joints;
};

// ── Animation ───────────────────────────────────────────────────────────────

enum class AnimChannelPath { Translation, Rotation, Scale };

struct AnimSampler {
    std::vector<float> times;
    std::vector<glm::vec4> values; // xyz for T/S, xyzw for R quaternion
};

struct AnimChannel {
    int             joint = 0;
    AnimChannelPath path  = AnimChannelPath::Translation;
    AnimSampler     sampler;
};

struct Animation {
    std::string              name;
    float                    duration = 0.0f;
    std::vector<AnimChannel> channels;
};

// ── SkinnedMesh ─────────────────────────────────────────────────────────────

struct SkinnedVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::uvec4 jointIds; // up to 4 bones per vertex
    glm::vec4  weights;  // weights for those bones (sum should ~= 1)
};

VkVertexInputBindingDescription getSkinnedVertexBinding();
std::array<VkVertexInputAttributeDescription, 5> getSkinnedVertexAttributes();

struct SkinnedMesh {
    std::vector<SkinnedVertex> vertices;
    std::vector<uint32_t>      indices;
    AllocatedBuffer            vertexBuffer;
    AllocatedBuffer            indexBuffer;
    Skeleton                   skeleton;
    std::vector<Animation>     animations;

    glm::vec3 aabbMin{ -1.0f };
    glm::vec3 aabbMax{  1.0f };
};

// Programmatic test mesh: a tapered bone chain that waves
SkinnedMesh createTestBoneChain();

void uploadSkinnedMesh(SkinnedMesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                       VkCommandPool commandPool, VkQueue queue);
void destroySkinnedMesh(VkDevice device, SkinnedMesh& mesh);

// ── Animation playback ──────────────────────────────────────────────────────

// Per-entity GPU-uploadable bone matrix palette.
struct BonePalette {
    glm::mat4 bones[MAX_BONES];
};

// Sample animation at time t, write joint world matrices into outMatrices.
// outMatrices[i] = joint_world(i) * inverseBind(i)
void computeBoneMatrices(const Skeleton& skeleton, const Animation& anim, float timeSec,
                         BonePalette& outPalette);
