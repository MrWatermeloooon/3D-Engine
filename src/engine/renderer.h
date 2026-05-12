#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

#include <entt/entt.hpp>

#include "swapchain.h"
#include "mesh.h"
#include "descriptors.h"
#include "shadow.h"
#include "postfx.h"
#include "frustum.h"
#include "instancing.h"
#include "skeletal.h"
#include "gpu_cull.h"
#include "hzb.h"

class ResourceManager;

struct RendererData {
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    uint32_t currentFrame = 0;
};

void createCommandPool(RendererData& data, VkDevice device, uint32_t graphicsFamily);
void allocateCommandBuffers(RendererData& data, VkDevice device);

struct DrawFrameInfo {
    VkPipeline       mainPipeline;
    VkPipelineLayout mainLayout;
    VkPipeline       shadowPipeline;
    VkPipelineLayout shadowLayout;

    DescriptorData*  descriptors;
    ShadowData*      shadow;
    OffscreenTarget* offscreen;
    BloomChain*      bloom;
    SSAOTarget*      ssao;
    CompositeData*   composite;
    LdrTarget*       ldr;
    PostFXPipelines* postfx;
    PostFXSettings*  settings;

    UniformBufferObject* ubo;
    CascadeUBO*          cascadeUbo;
    CullParamsUBO*       cullParams;       // built by engine.cpp every frame
    CandidateBuffer*     candidates;
    BatchHeaderBuffer*   batchHeaders;
    InstanceBuffer*      mainInstances;
    InstanceBuffer*      shadowInstances;
    IndirectBuffer*      indirect;
    GpuCullData*         gpuCull;
    HzbData*             hzb;

    // Skeletal animation
    SkinnedMesh*         skinnedMesh        = nullptr;
    VkPipeline           skinnedPipeline    = VK_NULL_HANDLE;
    VkPipelineLayout     skinnedLayout      = VK_NULL_HANDLE;
    VkDescriptorSet      boneDescriptorSet  = VK_NULL_HANDLE;

    class JobSystem*     jobSystem          = nullptr;

    // Per-frame stats (written by drawFrame, read by UI)
    int* visibleEntities = nullptr;
    int* totalEntities   = nullptr;

    entt::registry*  registry;
    ResourceManager* resources;
};

bool drawFrame(RendererData& renderer, SwapchainData& swapchain,
               VkDevice device, VkQueue graphicsQueue, VkQueue presentQueue,
               const DrawFrameInfo& info,
               bool& framebufferResized);
