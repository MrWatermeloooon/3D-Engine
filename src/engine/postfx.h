#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>

#include "buffer.h"
#include "config.h"

using engine_config::BLOOM_MIP_COUNT;
using engine_config::SSAO_KERNEL_SIZE;

// ── HDR offscreen target (main pass output) ─────────────────────────────────

struct OffscreenTarget {
    AllocatedImage colorImage;
    VkImageView    colorView    = VK_NULL_HANDLE;
    // Motion vectors — written as 2nd color attachment by mesh.frag. NDC-space
    // (prev - curr) in [-1, 1] units. Kept for future TAA/upscalers.
    AllocatedImage motionImage;
    VkImageView    motionView   = VK_NULL_HANDLE;
    AllocatedImage depthImage;
    VkImageView    depthView    = VK_NULL_HANDLE;    // depth aspect (attachment + sampling)
    VkRenderPass   renderPass     = VK_NULL_HANDLE;  // pass-A: CLEAR + initial UNDEFINED
    VkRenderPass   renderPassLate = VK_NULL_HANDLE;  // pass-B: LOAD + initial READ_ONLY (Phase 2.2)
    VkFramebuffer  framebuffer  = VK_NULL_HANDLE;
    VkSampler      sampler      = VK_NULL_HANDLE;    // for sampling color in post-fx
    VkSampler      depthSampler = VK_NULL_HANDLE;    // for sampling depth in SSAO
    VkExtent2D     extent{};
    VkFormat       colorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat       motionFormat = VK_FORMAT_R16G16_SFLOAT;
    VkFormat       depthFormat  = VK_FORMAT_UNDEFINED;
};

// ── Bloom mip chain ─────────────────────────────────────────────────────────

struct BloomChain {
    AllocatedImage image;
    std::array<VkImageView,    BLOOM_MIP_COUNT> mipViews{};
    std::array<VkFramebuffer,  BLOOM_MIP_COUNT> framebuffers{};
    std::array<VkExtent2D,     BLOOM_MIP_COUNT> extents{};
    VkRenderPass downsamplePass = VK_NULL_HANDLE;   // load=DontCare, store, finalLayout=ShaderRead
    VkRenderPass upsamplePass   = VK_NULL_HANDLE;   // load=Load,    store, finalLayout=ShaderRead
    VkSampler    sampler        = VK_NULL_HANDLE;

    // Descriptor sets for downsample (mip i reads from mip i-1, except 0 reads from HDR)
    std::array<VkDescriptorSet, BLOOM_MIP_COUNT> downsampleSets{};
    // Descriptor sets for upsample (mip i reads from mip i+1)
    std::array<VkDescriptorSet, BLOOM_MIP_COUNT> upsampleSets{};
};

// ── SSAO ────────────────────────────────────────────────────────────────────

struct SSAOTarget {
    AllocatedImage image;          // R8 occlusion
    VkImageView    view        = VK_NULL_HANDLE;
    VkRenderPass   renderPass  = VK_NULL_HANDLE;
    VkFramebuffer  framebuffer = VK_NULL_HANDLE;
    VkSampler      sampler     = VK_NULL_HANDLE;
    VkExtent2D     extent{};

    AllocatedImage noiseImage;
    VkImageView    noiseView   = VK_NULL_HANDLE;
    VkSampler      noiseSampler= VK_NULL_HANDLE;

    // Per-frame UBO (kernel + projection)
    std::vector<AllocatedBuffer> ubos;
    std::vector<void*>           uboMapped;

    std::vector<VkDescriptorSet> descriptorSets;
};

// ── Composite ───────────────────────────────────────────────────────────────

struct CompositeData {
    std::vector<AllocatedBuffer> ubos;
    std::vector<void*>           uboMapped;
    std::vector<VkDescriptorSet> descriptorSets;
};

// ── LDR target (output of composite, input to FXAA) ─────────────────────────

struct LdrTarget {
    AllocatedImage image;
    VkImageView    view        = VK_NULL_HANDLE;
    VkRenderPass   renderPass  = VK_NULL_HANDLE;
    VkFramebuffer  framebuffer = VK_NULL_HANDLE;
    VkSampler      sampler     = VK_NULL_HANDLE;
    VkExtent2D     extent{};
    VkFormat       format      = VK_FORMAT_R8G8B8A8_UNORM;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE; // used by FXAA pass
};

// ── Settings (UI-driven) ────────────────────────────────────────────────────

struct PostFXSettings {
    // Tone mapping / grading
    float exposure          = 0.0f;
    float saturation        = 1.05f;
    float contrast          = 1.05f;
    float gamma             = 2.2f;
    int   tonemapMode       = 0; // 0=ACES, 1=Reinhard, 2=Off
    glm::vec3 colorBalance  = glm::vec3(0.0f);

    // Bloom
    bool  bloomEnabled      = true;
    float bloomStrength     = 0.04f;
    // Threshold raised above 1.0 so an idle scene (lit only by IBL ≤ 1.0
    // SDR-range) doesn't bloom every chrome highlight. Crank back toward 1.0
    // for a hazier, more cinematic look when you have an explicit sun.
    float bloomThreshold    = 1.5f;
    float bloomFilterRadius = 1.0f;

    // SSAO
    bool  ssaoEnabled       = true;
    float ssaoStrength      = 0.5f;
    float ssaoRadius        = 0.5f;
    float ssaoBias          = 0.025f;
    float ssaoIntensity     = 1.5f;

    // Vignette
    float vignetteIntensity = 0.6f;
    float vignetteFalloff   = 0.15f;

    // Depth of field (CoC-based, computed inside the composite pass — see
    // composite.frag). Disabled by default so the screenshot look matches
    // the pre-DoF builds; enable to introduce focus.
    bool  dofEnabled       = false;
    float dofFocusDistance = 5.0f;   // metres of view-space depth in focus
    float dofFocusRange    = 1.5f;   // depth half-range that stays sharp
    float dofBokehRadius   = 4.0f;   // max CoC radius in pixels
    // Camera near/far for the depth linearisation in the shader. Set by the
    // engine to match the current camera each frame.
    float nearClip         = 0.1f;
    float farClip          = 200.0f;

    // FXAA
    bool  fxaaEnabled = true;

    // Variable rate shading (Phase 4).
    //   0 = off (always shade at 1×1 — same as no VRS)
    //   1 = auto / per-LOD (LOD0=1×1, LOD1=2×1, LOD2=2×2, LOD3=4×4)
    //   2 = forced 2×2
    //   3 = forced 4×4
    // Has no effect if the device doesn't support VK_KHR_fragment_shading_rate.
    int   vrsMode = 1;

    // Debug
    int   debugView = 0; // 0=final, 1=HDR, 2=Bloom, 3=SSAO, 5=Lit (no postfx)
};

// ── Pipelines ───────────────────────────────────────────────────────────────

struct PostFXPipelines {
    VkPipelineLayout bloomLayout = VK_NULL_HANDLE;
    VkPipeline       bloomDownsample = VK_NULL_HANDLE;
    VkPipeline       bloomUpsample   = VK_NULL_HANDLE;

    VkPipelineLayout ssaoLayout = VK_NULL_HANDLE;
    VkPipeline       ssao       = VK_NULL_HANDLE;

    VkPipelineLayout compositeLayout = VK_NULL_HANDLE;
    VkPipeline       composite       = VK_NULL_HANDLE;

    VkPipelineLayout fxaaLayout = VK_NULL_HANDLE;
    VkPipeline       fxaa       = VK_NULL_HANDLE;

    VkDescriptorSetLayout bloomSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoSetLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout fxaaSetLayout      = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
};

// ── Module API ──────────────────────────────────────────────────────────────

void createOffscreenTarget(OffscreenTarget& t, VkPhysicalDevice physicalDevice, VkDevice device,
                           VkExtent2D extent, VkFormat depthFormat);
void destroyOffscreenTarget(VkDevice device, OffscreenTarget& t);

void createBloomChain(BloomChain& b, VkPhysicalDevice physicalDevice, VkDevice device,
                      VkExtent2D mainExtent);
void destroyBloomChain(VkDevice device, BloomChain& b);

void createSSAO(SSAOTarget& s, VkPhysicalDevice physicalDevice, VkDevice device,
                VkCommandPool commandPool, VkQueue queue,
                VkExtent2D mainExtent, uint32_t framesInFlight);
void destroySSAO(VkDevice device, SSAOTarget& s);

void createCompositeData(CompositeData& c, VkPhysicalDevice physicalDevice, VkDevice device,
                         uint32_t framesInFlight);
void destroyCompositeData(VkDevice device, CompositeData& c);

void createLdrTarget(LdrTarget& t, VkPhysicalDevice physicalDevice, VkDevice device,
                     VkExtent2D extent);
void destroyLdrTarget(VkDevice device, LdrTarget& t);

void createPostFXPipelines(PostFXPipelines& p, VkDevice device, BloomChain& bloom,
                           SSAOTarget& ssao, CompositeData& composite, LdrTarget& ldr,
                           OffscreenTarget& offscreen, VkRenderPass swapchainPass,
                           uint32_t framesInFlight);
void destroyPostFXPipelines(VkDevice device, PostFXPipelines& p);

// Per-frame updates
void updateSSAOUbo(SSAOTarget& s, uint32_t frameIndex, const glm::mat4& projection,
                   const glm::mat4& invProjection, VkExtent2D mainExtent,
                   const PostFXSettings& settings);
void updateCompositeUbo(CompositeData& c, uint32_t frameIndex, const PostFXSettings& settings);

// Recording helpers
void recordSSAO(VkCommandBuffer cmd, SSAOTarget& s, PostFXPipelines& p, uint32_t frameIndex);
void recordBloom(VkCommandBuffer cmd, BloomChain& b, PostFXPipelines& p,
                 const PostFXSettings& settings);
