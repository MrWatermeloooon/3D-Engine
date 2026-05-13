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
#include "dlss.h"

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

    // Pick the offscreen render extent — m_swapchain.extent when DLSS is off,
    // the NGX-reported optimal input size when on. Stores into m_renderExtent.
    // If DLSS is on but NGX fails, falls back to native and forces dlssEnabled
    // off so callers don't try to create a feature.
    void computeRenderExtent();

    // After the offscreen/upscale targets exist and m_renderExtent matches the
    // user's wanted DLSS state, (re)build the NGX feature. Submits a one-shot
    // command buffer (NGX records init work into it). Updates m_dlssActive*.
    void rebuildDlssFeatureIfNeeded();

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

    // DLSS (Phase 4). `m_haltonIndex` cycles per frame to drive sub-pixel
    // jitter; `m_currViewProj` / `m_dlssPrevViewProj` feed mesh.vert's motion
    // vector calculation (separate from m_prevViewProj which is used by GPU
    // cull's HZB projection — same value, kept distinct for clarity).
    DlssSettings       m_dlssSettings{};
    uint32_t           m_haltonIndex = 1;
    glm::mat4          m_dlssPrevViewProj{1.0f};

    // Effective offscreen render resolution. Equal to m_swapchain.extent when
    // DLSS is off; the optimal-input size reported by NGX when DLSS is on.
    VkExtent2D         m_renderExtent{0, 0};

    // Tracks the (enabled, quality) snapshot the current DLSS feature was
    // created for. Differs from m_dlssSettings → trigger a recreate at the
    // top of the next frame.
    bool               m_dlssActiveEnabled = false;
    DlssQuality        m_dlssActiveQuality = DlssQuality::Off;
    bool               m_dlssResetHistory  = true;

    // Full-res target that DLSS writes its upscaled output into. Resides at
    // m_swapchain.extent. FXAA samples from here when the active DLSS
    // feature is alive.
    UpscaleTarget      m_upscale;

public:
    RtSettings&   rtSettings()   { return m_rtSettings; }
    DlssSettings& dlssSettings() { return m_dlssSettings; }

private:

    // Previous frame's view-proj — fed to compute cull for HZB-space projection.
    // Identity on the very first frame; the HZB is cleared to depth=1.0 so
    // nothing is culled until real data exists.
    glm::mat4          m_prevViewProj{1.0f};
    bool               m_hasPrevVP = false;

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
