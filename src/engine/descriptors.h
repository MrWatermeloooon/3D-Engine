#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include "buffer.h"

struct UniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 cameraPos;  // .xyz = position
    // x = RT shadows enabled (0/1)
    // y = sun angular radius for cone sampling
    // z = ray samples per light per fragment (clamped 1..64 in shader)
    // w = sunOnly flag
    alignas(16) glm::vec4 rtParams{0.0f};
    // x = RT reflections enabled (0/1)
    // y = reflection samples (clamped 1..16)
    // z = reflection max ray distance
    // w = reflection intensity multiplier
    alignas(16) glm::vec4 rtParams2{0.0f};
    // x = RT GI enabled (0/1)
    // y = GI samples per fragment (clamped 1..16)
    // z = GI max ray distance
    // w = GI intensity multiplier
    alignas(16) glm::vec4 rtParams3{0.0f};
    // Previous-frame view-projection. Used by mesh.vert to compute motion
    // vectors for DLSS / TAA. Identity on the first frame.
    alignas(16) glm::mat4 prevViewProj{1.0f};
    // Sub-pixel jitter offset applied to the camera proj (DLSS).
    //   .xy = current frame jitter in NDC units (already baked into proj —
    //         shaders use these to subtract jitter when reconstructing motion).
    //   .zw = reserved.
    alignas(16) glm::vec4 jitterOffset{0.0f};
};

struct PushConstants {
    glm::mat4 model;
    glm::vec4 color;
    float metallic;
    float roughness;
    float pad0, pad1;
};

constexpr uint32_t BINDLESS_MAX_TEXTURES = 1024;

struct DescriptorData {
    VkDescriptorSetLayout sceneSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE; // bindless texture array
    VkDescriptorPool descriptorPool         = VK_NULL_HANDLE;

    std::vector<VkDescriptorSet> sceneSets;
    VkDescriptorSet              bindlessTexturesSet = VK_NULL_HANDLE; // single, shared

    std::vector<AllocatedBuffer> uniformBuffers;
    std::vector<void*> uniformBuffersMapped;
};

VkDescriptorSetLayout createSceneSetLayout(VkDevice device);
VkDescriptorSetLayout createMaterialSetLayout(VkDevice device);  // bindless

void allocateBindlessTexturesSet(DescriptorData& data, VkDevice device);

// Bind one texture into a specific slot of the global bindless array.
void writeBindlessTexture(DescriptorData& data, VkDevice device,
                          uint32_t slot, VkImageView view, VkSampler sampler);

void createUniformBuffers(DescriptorData& data, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight);

void createDescriptorPool(DescriptorData& data, VkDevice device,
                          uint32_t framesInFlight, uint32_t maxMaterials);

void createSceneDescriptorSets(DescriptorData& data, VkDevice device, uint32_t framesInFlight,
                               const std::vector<VkBuffer>& lightBuffers,
                               const std::vector<VkBuffer>& cascadeBuffers,
                               VkImageView shadowArrayView, VkSampler shadowSampler);

// Update the TLAS bound at scene set binding 4 for `frame`. Safe to call any
// time the descriptor set isn't in use by the GPU (i.e. after the per-frame
// fence wait at the top of drawFrame).
void writeSceneTlas(DescriptorData& data, VkDevice device, uint32_t frame,
                    VkAccelerationStructureKHR tlas);

// Update the RT per-instance material SSBO bound at scene set binding 5.
// Same lifetime / safety contract as writeSceneTlas. `bufferSize` is the
// authoritative range size; pass VK_WHOLE_SIZE to consume the whole buffer.
void writeSceneRtMaterials(DescriptorData& data, VkDevice device, uint32_t frame,
                           VkBuffer buffer, VkDeviceSize bufferSize);

void updateUniformBuffer(DescriptorData& data, uint32_t currentFrame,
                         const UniformBufferObject& ubo);
