#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

#include "buffer.h"

constexpr uint32_t SHADOW_CASCADE_COUNT = 4;
constexpr uint32_t SHADOW_MAP_SIZE      = 2048;
constexpr VkFormat SHADOW_FORMAT        = VK_FORMAT_D32_SFLOAT;

struct CascadeUBO {
    glm::mat4 lightViewProj[SHADOW_CASCADE_COUNT];
    glm::vec4 splitsViewSpace; // x,y,z,w = far depth of each cascade in view space
};

struct ShadowData {
    VkImage         image      = VK_NULL_HANDLE;
    VkDeviceMemory  memory     = VK_NULL_HANDLE;
    VkImageView     arrayView  = VK_NULL_HANDLE;             // sampled in main pass
    VkImageView     layerViews[SHADOW_CASCADE_COUNT]{};      // per-cascade attachments
    VkFramebuffer   framebuffers[SHADOW_CASCADE_COUNT]{};
    VkRenderPass    renderPass = VK_NULL_HANDLE;
    VkSampler       sampler    = VK_NULL_HANDLE;

    // Per-frame UBO with cascade matrices
    std::vector<AllocatedBuffer> cascadeBuffers;
    std::vector<void*>           cascadeMapped;

    // Settings
    float cascadeSplitLambda = 0.7f;   // 0 = uniform, 1 = log
    float depthBias          = 1.25f;
    float slopeBias          = 1.75f;
    float pcfRadius          = 1.5f;   // texel radius for PCF
};

void createShadowResources(ShadowData& shadow, VkPhysicalDevice physicalDevice,
                           VkDevice device, VkCommandPool commandPool, VkQueue queue,
                           uint32_t framesInFlight);

void destroyShadowResources(VkDevice device, ShadowData& shadow);

// Computes cascade view-projection matrices and split distances based on the
// camera, near/far, and the directional light direction. Writes the result
// into the per-frame cascade UBO mapped pointer.
void computeCascades(ShadowData& shadow, uint32_t frameIndex,
                     const glm::mat4& cameraView, const glm::mat4& cameraProj,
                     float nearClip, float farClip,
                     const glm::vec3& lightDir,
                     CascadeUBO& outUbo);
