#pragma once

#include "window.h"
#include "vulkan_init.h"
#include "swapchain.h"
#include "pipeline.h"
#include "renderer.h"
#include "descriptors.h"
#include "texture.h"
#include "depth.h"
#include "mesh.h"
#include "camera.h"
#include "resource_manager.h"
#include "scene.h"
#include "debug_ui.h"
#include "lights.h"
#include "shadow.h"
#include "postfx.h"
#include "instancing.h"
#include "skeletal.h"
#include "jobs.h"

class Engine {
public:
    void init();
    void run();
    void cleanup();

    ShadowData&     shadow()   { return m_shadow; }
    PostFXSettings& postFX()   { return m_postFXSettings; }
    int  visibleEntities() const { return m_visibleEntities; }
    int  totalEntities()   const { return m_totalEntities; }

private:
    void recreateSwapchain();
    void processInput(double deltaTime);

    Window*          m_window = nullptr;
    VulkanContext    m_vk{};
    SwapchainData    m_swapchain{};
    PipelineData     m_pipeline{};
    PipelineData     m_shadowPipeline{};
    RendererData     m_renderer{};
    DescriptorData   m_descriptors{};
    Camera           m_camera;
    ResourceManager  m_resources;
    Scene            m_scene;
    DebugUI          m_debugUI;
    LightBufferData  m_lightBuffers;
    ShadowData       m_shadow;
    InstanceBuffer   m_instances;
    IndirectBuffer   m_indirect;

    // Skeletal animation
    SkinnedMesh                  m_skinnedMesh;
    PipelineData                 m_skinnedPipeline;
    VkDescriptorSetLayout        m_boneSetLayout = VK_NULL_HANDLE;
    std::vector<AllocatedBuffer> m_boneUbos;
    std::vector<void*>           m_boneMapped;
    std::vector<VkDescriptorSet> m_boneDescSets;

    // Post-FX
    OffscreenTarget  m_offscreen;
    BloomChain       m_bloom;
    SSAOTarget       m_ssao;
    CompositeData    m_composite;
    LdrTarget        m_ldr;
    PostFXPipelines  m_postFX;
    PostFXSettings   m_postFXSettings{};

    VkFormat         m_depthFormat = VK_FORMAT_UNDEFINED;

    // Job system
    JobSystem        m_jobs;

    // Per-frame stats
    int    m_visibleEntities = 0;
    int    m_totalEntities   = 0;

    double m_lastFrameTime = 0.0;
    double m_lastMouseX    = 0.0;
    double m_lastMouseY    = 0.0;
    bool   m_firstMouse    = true;
    bool   m_tabWasDown    = false;

    float  m_hotReloadTimer = 0.0f;
};
