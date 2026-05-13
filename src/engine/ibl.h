#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <vector>
#include <string>

// Image-based lighting bake. Produces three GPU resources on init:
//   * envCube       — unfiltered environment cubemap. Sampled by sky.frag for
//                     the background and used as the *source* of the
//                     prefilter / irradiance bakes.
//   * prefilterCube — GGX-prefiltered specular cubemap with N mips, one mip
//                     per roughness band (mip 0 = sharpest = roughness 0).
//   * irradianceCube— diffuse-irradiance cubemap (lambert convolution).
//   * brdfLut       — split-sum BRDF LUT (R = scale, G = bias).
//
// The runtime samples (irradiance, prefilter, brdfLut) — the env cube is only
// kept so a UI rebuild can re-run the convolution without going back to the
// source.

struct IblData {
    // Environment cubemap (mip 0 = unfiltered). RGBA16F, 1 mip.
    VkImage         envImage     = VK_NULL_HANDLE;
    VmaAllocation   envAlloc     = VK_NULL_HANDLE;
    VkImageView     envView      = VK_NULL_HANDLE;
    uint32_t        envFaceSize  = 512;

    // Prefiltered specular cubemap. RGBA16F, N mips.
    VkImage         prefilterImage   = VK_NULL_HANDLE;
    VmaAllocation   prefilterAlloc   = VK_NULL_HANDLE;
    VkImageView     prefilterView    = VK_NULL_HANDLE;          // full chain (sampling)
    std::vector<VkImageView> prefilterMipViews;                  // per-mip writes
    // 512 matches envFaceSize so mip 0 is a 1:1 copy of env — the sky (which
    // samples this map at lod 0) stays as crisp as the env source rather than
    // losing half its resolution to a 256² downsample.
    uint32_t        prefilterFaceSize = 512;
    uint32_t        prefilterMipCount = 7;

    // Diffuse-irradiance cubemap. RGBA16F, 1 mip.
    VkImage         irradianceImage   = VK_NULL_HANDLE;
    VmaAllocation   irradianceAlloc   = VK_NULL_HANDLE;
    VkImageView     irradianceView    = VK_NULL_HANDLE;
    uint32_t        irradianceFaceSize = 32;

    // Split-sum BRDF LUT (RG16F, 2D, 1 mip).
    VkImage         brdfLutImage  = VK_NULL_HANDLE;
    VmaAllocation   brdfLutAlloc  = VK_NULL_HANDLE;
    VkImageView     brdfLutView   = VK_NULL_HANDLE;
    uint32_t        brdfLutSize   = 256;

    // Shared sampler used by mesh.frag / sky.frag for cubemap reads.
    VkSampler       cubeSampler   = VK_NULL_HANDLE;
    // Linear-clamp 2D sampler for the BRDF LUT.
    VkSampler       lutSampler    = VK_NULL_HANDLE;
};

// Procedural sky bake parameters. Match these to your in-scene sun for a
// coherent look between geometry and skybox.
struct IblBakeParams {
    glm::vec3 sunDir{ 0.4f, 0.8f, 0.4f };  // toward the sun
    float     sunIntensity = 1.5f;
    glm::vec3 zenithColor { 0.18f, 0.30f, 0.55f };
    glm::vec3 horizonColor{ 0.78f, 0.78f, 0.72f };
    glm::vec3 groundColor { 0.10f, 0.09f, 0.08f };
    // Multiplies the procedural sky output (or loaded HDR) before it's stored
    // in the env cubemap. 1.0 keeps the sky background at SDR brightness;
    // increase for HDR-style backgrounds. Surface-light contribution from IBL
    // is scaled SEPARATELY in mesh.frag, so this is purely a sky-display
    // knob — adjust without worrying about over-lighting the scene.
    float     intensity    = 1.0f;
    // When non-empty, load this HDR equirect file and bake from it instead of
    // the procedural sky.
    std::string hdrPath;
};

void createIbl(IblData& ibl, VkPhysicalDevice physicalDevice, VkDevice device);
void destroyIbl(VkDevice device, IblData& ibl);

// Rebuild env + irradiance + prefilter + BRDF LUT. Safe to call any time the
// IBL bindings aren't in use by the GPU (i.e. after a vkDeviceWaitIdle). For
// startup, call right after createIbl.
void bakeIbl(IblData& ibl, VkPhysicalDevice physicalDevice, VkDevice device,
             VkCommandPool commandPool, VkQueue queue,
             const IblBakeParams& params);
