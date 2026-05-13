#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

#include "buffer.h"

// Hierarchical Z-buffer for occlusion culling.
//
// Single R32_SFLOAT image with a full mip chain; each finer→coarser step is a
// 2×2 maximum reduction. Mip 0 is filled from the previous main pass's depth
// attachment (depth values are already in [0,1] post-projection). Subsequent
// mips reduce from the previous mip.
//
// The cull compute shader samples this image with textureLod() to compare an
// AABB's projected screen rect against the conservative max depth at the
// appropriate mip — if the AABB is strictly behind the HZB, it's occluded.
//
// The HZB lives in VK_IMAGE_LAYOUT_GENERAL throughout — it's both written as
// a storage image (per-mip image2D) during reduction and sampled as a combined
// image sampler during cull. Avoids per-mip layout transitions.
//
// NOTE: GENERAL disables some driver read-only optimisations that
// SHADER_READ_ONLY_OPTIMAL would enable. In *this* engine the HZB is only
// ever read by compute shaders (reduce + cull), never by fragment / vertex
// stages, so the practical win from a SHADER_READ_ONLY_OPTIMAL transition
// would be marginal on most drivers. If the HZB ever gets sampled from a
// fragment shader (e.g. a soft-particle pass), revisit: transition each mip
// to SHADER_READ_ONLY_OPTIMAL after its write barrier in recordHzbReduce(),
// and update gpu_cull's descriptor write to declare the same layout.

struct HzbData {
    AllocatedImage              image;
    VkImageView                 fullView   = VK_NULL_HANDLE; // all mips, for sampling
    std::vector<VkImageView>    mipViews;                    // single-mip storage views
    VkSampler                   sampler    = VK_NULL_HANDLE;
    VkExtent2D                  extent{};
    uint32_t                    mipCount   = 0;

    // Reduction pipeline + per-mip descriptor sets.
    // Set 0 holds: binding 0 = sampler2D (source mip; for mip 0 this is the
    // depth attachment), binding 1 = image2D (dest mip).
    VkDescriptorSetLayout        reduceSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout             reducePipelineLayout = VK_NULL_HANDLE;
    VkPipeline                   reducePipeline       = VK_NULL_HANDLE;
    VkDescriptorPool             reduceDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> reduceSets;  // one per mip
};

struct HzbReducePushConstants {
    int32_t dstWidth;
    int32_t dstHeight;
    int32_t isFirstMip;  // 1 → src is depth attachment; 0 → src is HZB mip i-1
    int32_t _pad0;
};

// Creates the HZB image, mip views, sampler, reduction pipeline, descriptor
// sets, and clears the image to depth=1.0 so the first frame culls nothing.
// `depthView` and `depthSampler` describe the offscreen depth attachment that
// will source mip 0 every frame.
void createHzb(HzbData& hzb,
               VkPhysicalDevice physicalDevice, VkDevice device,
               VkCommandPool commandPool, VkQueue queue,
               VkExtent2D extent,
               VkImageView depthView, VkSampler depthSampler,
               const std::string& reduceShaderPath);

void destroyHzb(VkDevice device, HzbData& hzb);

// Record the full mip chain reduction. Caller must guarantee:
//   * Source depth attachment is in DEPTH_STENCIL_READ_ONLY_OPTIMAL (or
//     SHADER_READ_ONLY_OPTIMAL) and writes have been made visible to the
//     compute stage.
//   * Caller will issue a compute→compute barrier after this if the cull
//     shader reads the HZB in the same submission (typically the HZB is
//     used by NEXT frame's cull, so the queue submission boundary acts as
//     a barrier).
void recordHzbReduce(VkCommandBuffer cmd, HzbData& hzb);
