#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include "buffer.h"

struct UniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 cameraPos;  // .xyz = position
};

struct PushConstants {
    glm::mat4 model;
    glm::vec4 color;
    float metallic;
    float roughness;
    float pad0, pad1;
};

struct DescriptorData {
    VkDescriptorSetLayout sceneSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool         = VK_NULL_HANDLE;

    std::vector<VkDescriptorSet> sceneSets;

    std::vector<AllocatedBuffer> uniformBuffers;
    std::vector<void*> uniformBuffersMapped;
};

VkDescriptorSetLayout createSceneSetLayout(VkDevice device);
VkDescriptorSetLayout createMaterialSetLayout(VkDevice device);

void createUniformBuffers(DescriptorData& data, VkPhysicalDevice physicalDevice,
                          VkDevice device, uint32_t framesInFlight);

void createDescriptorPool(DescriptorData& data, VkDevice device,
                          uint32_t framesInFlight, uint32_t maxMaterials);

void createSceneDescriptorSets(DescriptorData& data, VkDevice device, uint32_t framesInFlight,
                               const std::vector<VkBuffer>& lightBuffers,
                               const std::vector<VkBuffer>& cascadeBuffers,
                               VkImageView shadowArrayView, VkSampler shadowSampler);

VkDescriptorSet allocateMaterialDescriptorSet(const DescriptorData& data, VkDevice device,
                                              VkImageView imageView, VkSampler sampler);

void updateUniformBuffer(DescriptorData& data, uint32_t currentFrame,
                         const UniformBufferObject& ubo);
