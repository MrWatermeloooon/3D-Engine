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
    PostFXPipelines* postfx;
    PostFXSettings*  settings;

    UniformBufferObject* ubo;
    CascadeUBO*          cascadeUbo;

    entt::registry*  registry;
    ResourceManager* resources;
};

bool drawFrame(RendererData& renderer, SwapchainData& swapchain,
               VkDevice device, VkQueue graphicsQueue, VkQueue presentQueue,
               const DrawFrameInfo& info,
               bool& framebufferResized);
