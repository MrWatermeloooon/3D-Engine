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
#include "gpu_cull.h"
#include "hzb.h"
#include "raytracing.h"
#include "sky.h"
#include "ibl.h"
#include "profiler.h"

#include <memory>
#include <unordered_map>

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

    std::unique_ptr<Window> m_window;
    VulkanContext    m_vk{};
    SwapchainData    m_swapchain{};
    PipelineData     m_pipeline{};
    PipelineData     m_shadowPipeline{};
    SkyData          m_sky{};
    IblData          m_ibl{};
    IblBakeParams    m_iblParams{};
    bool             m_iblRebuildRequested = true;
    std::string      m_pendingGltfLoad;
    RendererData     m_renderer{};
    DescriptorData   m_descriptors{};
    Camera           m_camera;
    ResourceManager  m_resources;
    Scene            m_scene;
    DebugUI          m_debugUI;
    LightBufferData  m_lightBuffers;
    ShadowData       m_shadow;
    // GPU-driven culling: CPU writes candidates+headers, compute fills the two
    // instance buffers and the indirect command buffer.
    CandidateBuffer    m_candidates;
    BatchHeaderBuffer  m_batchHeaders;
    InstanceBuffer     m_mainInstances;     // compute writes visible-only
    InstanceBuffer     m_shadowInstances;   // compute writes all (no cull)
    IndirectBuffer     m_indirect;
    GpuCullData        m_gpuCull;
    HzbData            m_hzb;

    // Ray tracing (Phase 1a: scaffolding only — no GPU work yet).
    RtScene            m_rtScene;
    RtSettings         m_rtSettings{};

    // Motion-vector + jitter scaffolding kept for future TAA / upscaling
    // integration. Currently inert (no upscaling, no jitter applied to
    // the projection). The shader still computes and writes motion vectors
    // into a second offscreen attachment, which costs negligible bandwidth.
    glm::mat4          m_prevViewProjMotion{1.0f}; // un-jittered, for motion vectors

public:
    RtSettings&    rtSettings()        { return m_rtSettings; }
    IblBakeParams& iblParams()         { return m_iblParams; }
    bool&          iblRebuildRequest() { return m_iblRebuildRequested; }

private:

    // Previous frame's view-proj — fed to compute cull for HZB-space projection.
    // Identity on the very first frame; the HZB is cleared to depth=1.0 so
    // nothing is culled until real data exists.
    glm::mat4          m_prevViewProj{1.0f};
    bool               m_hasPrevVP = false;

    // Per-entity previous world matrix for motion vectors. Updated at the end
    // of each successful frame. Entities that disappear are pruned by checking
    // registry.valid() on read.
    std::unordered_map<entt::entity, glm::mat4> m_prevTransforms;

    // Skeletal animation
    SkinnedMesh                  m_skinnedMesh;
    PipelineData                 m_skinnedPipeline;
    PipelineData                 m_skinnedShadowPipeline;
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

    // Per-frame profiler (GPU timestamps + CPU scopes).
    Profiler         m_profiler;

public:
    const Profiler&  profiler() const { return m_profiler; }
private:

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
