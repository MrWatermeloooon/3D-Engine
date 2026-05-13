#pragma once

#include <vulkan/vulkan.h>
#include <string>

// Procedural skybox drawn as a fullscreen triangle at far depth, inside the
// main offscreen render pass right after geometry. The sky pipeline writes
// to the same color + motion attachments as mesh.frag and uses
// depthCompareOp = LESS_OR_EQUAL with depthWrite OFF, so it fills only the
// pixels that geometry left at depth = 1.0.
//
// The shader pulls the first directional light out of the LightsUBO (scene
// set, binding 1) to align the sun disc with the in-scene sun direction.
// This costs no extra descriptor binding — the scene set is already bound
// for the main pass.
//
// When IBL lands, the procedural function in sky.frag can be replaced by a
// cubemap sample and the same fragment becomes the IBL specular probe
// source. The pipeline stays the same shape.

struct SkyData {
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       pipeline       = VK_NULL_HANDLE;
};

void createSkyPipeline(SkyData& sky, VkDevice device, VkRenderPass renderPass,
                       VkExtent2D extent, VkDescriptorSetLayout sceneSetLayout,
                       const std::string& vertPath, const std::string& fragPath);

void destroySkyPipeline(VkDevice device, SkyData& sky);

void recordSkyPass(VkCommandBuffer cmd, const SkyData& sky, VkDescriptorSet sceneSet,
                   VkExtent2D extent);
