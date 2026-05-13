#pragma once

#include <vulkan/vulkan.h>
#include <string>

struct PipelineData {
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       graphicsPipeline = VK_NULL_HANDLE;
};

VkShaderModule loadShaderModule(VkDevice device, const std::string& filepath);

PipelineData createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkExtent2D extent,
                                    VkDescriptorSetLayout sceneSetLayout,
                                    VkDescriptorSetLayout materialSetLayout,
                                    const std::string& vertPath, const std::string& fragPath);

PipelineData createShadowPipeline(VkDevice device, VkRenderPass shadowRenderPass,
                                  uint32_t shadowMapSize,
                                  const std::string& vertPath);

// Skinned mesh pipeline (separate vertex layout: SkinnedVertex + bone palette set)
PipelineData createSkinnedPipeline(VkDevice device, VkRenderPass renderPass, VkExtent2D extent,
                                   VkDescriptorSetLayout sceneSetLayout,
                                   VkDescriptorSetLayout materialSetLayout,
                                   VkDescriptorSetLayout boneSetLayout,
                                   const std::string& vertPath, const std::string& fragPath);

// Skinned shadow pipeline: depth-only, takes (lightViewProj, model) as push
// constants and the same bone palette set as the main skinned pipeline.
PipelineData createSkinnedShadowPipeline(VkDevice device, VkRenderPass shadowRenderPass,
                                         uint32_t shadowMapSize,
                                         VkDescriptorSetLayout boneSetLayout,
                                         const std::string& vertPath);
