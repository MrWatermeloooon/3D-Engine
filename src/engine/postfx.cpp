#include "postfx.h"
#include "depth.h"
#include "pipeline.h"
#include "../utils/vk_check.h"

#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>

// ──────────────────────────────────────────────────────────────────────────
// Generic helpers
// ──────────────────────────────────────────────────────────────────────────

static VkSampler makeLinearClampSampler(VkDevice device) {
    VkSamplerCreateInfo s{};
    s.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter    = VK_FILTER_LINEAR;
    s.minFilter    = VK_FILTER_LINEAR;
    s.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.maxLod       = 1.0f;
    VkSampler out;
    VK_CHECK(vkCreateSampler(device, &s, nullptr, &out));
    return out;
}

static VkSampler makeNoiseSampler(VkDevice device) {
    VkSamplerCreateInfo s{};
    s.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter    = VK_FILTER_NEAREST;
    s.minFilter    = VK_FILTER_NEAREST;
    s.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSampler out;
    VK_CHECK(vkCreateSampler(device, &s, nullptr, &out));
    return out;
}

static VkRenderPass makeColorOnlyPass(VkDevice device, VkFormat fmt,
                                      VkAttachmentLoadOp loadOp,
                                      VkImageLayout initialLayout) {
    VkAttachmentDescription a{};
    a.format         = fmt;
    a.samples        = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp         = loadOp;
    a.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    a.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    a.initialLayout  = initialLayout;
    a.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &ref;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments    = &a;
    rp.subpassCount    = 1;
    rp.pSubpasses      = &sub;
    rp.dependencyCount = 2;
    rp.pDependencies   = deps;
    VkRenderPass out;
    VK_CHECK(vkCreateRenderPass(device, &rp, nullptr, &out));
    return out;
}

// ──────────────────────────────────────────────────────────────────────────
// Offscreen HDR target (main pass output)
// ──────────────────────────────────────────────────────────────────────────

void createOffscreenTarget(OffscreenTarget& t, VkPhysicalDevice physicalDevice, VkDevice device,
                           VkExtent2D extent, VkFormat depthFormat)
{
    t.extent      = extent;
    t.depthFormat = depthFormat;

    // Color: HDR R16G16B16A16_SFLOAT
    t.colorImage = createImage(physicalDevice, device, extent.width, extent.height, 1,
        t.colorFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    t.colorView = createImageView(device, t.colorImage.image, t.colorFormat,
                                  VK_IMAGE_ASPECT_COLOR_BIT, 1);

    // Motion vectors: RG16F. Color attachment + sampleable by DLSS.
    t.motionImage = createImage(physicalDevice, device, extent.width, extent.height, 1,
        t.motionFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    t.motionView = createImageView(device, t.motionImage.image, t.motionFormat,
                                   VK_IMAGE_ASPECT_COLOR_BIT, 1);

    // Depth: depthFormat, sampleable
    t.depthImage = createImage(physicalDevice, device, extent.width, extent.height, 1,
        depthFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    t.depthView = createImageView(device, t.depthImage.image, depthFormat,
                                  VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    // Render pass: 2 color attachments (color + motion) → SHADER_READ,
    // depth → DEPTH_STENCIL_READ_ONLY.
    VkAttachmentDescription atts[3]{};
    atts[0].format         = t.colorFormat;
    atts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    atts[1].format         = t.motionFormat;
    atts[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[1].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    atts[2].format         = depthFormat;
    atts[2].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[2].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRefs[2]{};
    colorRefs[0].attachment = 0; colorRefs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorRefs[1].attachment = 1; colorRefs[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthRef{}; depthRef.attachment = 2; depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 2;
    sub.pColorAttachments       = colorRefs;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 3;
    rp.pAttachments    = atts;
    rp.subpassCount    = 1;
    rp.pSubpasses      = &sub;
    rp.dependencyCount = 2;
    rp.pDependencies   = deps;
    VK_CHECK(vkCreateRenderPass(device, &rp, nullptr, &t.renderPass));

    VkImageView attsView[] = { t.colorView, t.motionView, t.depthView };
    VkFramebufferCreateInfo fb{};
    fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass      = t.renderPass;
    fb.attachmentCount = 3;
    fb.pAttachments    = attsView;
    fb.width           = extent.width;
    fb.height          = extent.height;
    fb.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(device, &fb, nullptr, &t.framebuffer));

    t.sampler      = makeLinearClampSampler(device);
    t.depthSampler = makeLinearClampSampler(device);
}

void destroyOffscreenTarget(VkDevice device, OffscreenTarget& t) {
    if (t.depthSampler) vkDestroySampler(device, t.depthSampler, nullptr);
    if (t.sampler)      vkDestroySampler(device, t.sampler, nullptr);
    if (t.framebuffer)  vkDestroyFramebuffer(device, t.framebuffer, nullptr);
    if (t.renderPass)   vkDestroyRenderPass(device, t.renderPass, nullptr);
    if (t.colorView)    vkDestroyImageView(device, t.colorView, nullptr);
    if (t.motionView)   vkDestroyImageView(device, t.motionView, nullptr);
    if (t.depthView)    vkDestroyImageView(device, t.depthView, nullptr);
    if (t.colorImage.image) {
        vkDestroyImage(device, t.colorImage.image, nullptr);
        vkFreeMemory(device, t.colorImage.memory, nullptr);
    }
    if (t.motionImage.image) {
        vkDestroyImage(device, t.motionImage.image, nullptr);
        vkFreeMemory(device, t.motionImage.memory, nullptr);
    }
    if (t.depthImage.image) {
        vkDestroyImage(device, t.depthImage.image, nullptr);
        vkFreeMemory(device, t.depthImage.memory, nullptr);
    }
    t = {};
}

// ──────────────────────────────────────────────────────────────────────────
// Bloom chain
// ──────────────────────────────────────────────────────────────────────────

void createBloomChain(BloomChain& b, VkPhysicalDevice physicalDevice, VkDevice device,
                      VkExtent2D mainExtent)
{
    constexpr VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;

    // mip 0 is half resolution of mainExtent
    uint32_t w = std::max(1u, mainExtent.width  / 2);
    uint32_t h = std::max(1u, mainExtent.height / 2);
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i) {
        b.extents[i] = { w, h };
        w = std::max(1u, w / 2);
        h = std::max(1u, h / 2);
    }

    // Single image with N mips
    VkImageCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType     = VK_IMAGE_TYPE_2D;
    info.format        = fmt;
    info.extent        = { b.extents[0].width, b.extents[0].height, 1 };
    info.mipLevels     = BLOOM_MIP_COUNT;
    info.arrayLayers   = 1;
    info.samples       = VK_SAMPLE_COUNT_1_BIT;
    info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    info.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &info, nullptr, &b.image.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, b.image.image, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &alloc, nullptr, &b.image.memory));
    VK_CHECK(vkBindImageMemory(device, b.image.image, b.image.memory, 0));

    // Per-mip views
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i) {
        VkImageViewCreateInfo v{};
        v.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image    = b.image.image;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D;
        v.format   = fmt;
        v.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        v.subresourceRange.baseMipLevel   = i;
        v.subresourceRange.levelCount     = 1;
        v.subresourceRange.baseArrayLayer = 0;
        v.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &v, nullptr, &b.mipViews[i]));
    }

    // Render passes
    b.downsamplePass = makeColorOnlyPass(device, fmt, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                         VK_IMAGE_LAYOUT_UNDEFINED);
    b.upsamplePass   = makeColorOnlyPass(device, fmt, VK_ATTACHMENT_LOAD_OP_LOAD,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Framebuffers (one per mip — same renderpass-compatible structure)
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i) {
        VkFramebufferCreateInfo fb{};
        fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass      = b.downsamplePass; // compatible with upsample pass too (same attachments)
        fb.attachmentCount = 1;
        fb.pAttachments    = &b.mipViews[i];
        fb.width           = b.extents[i].width;
        fb.height          = b.extents[i].height;
        fb.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(device, &fb, nullptr, &b.framebuffers[i]));
    }

    b.sampler = makeLinearClampSampler(device);
}

void destroyBloomChain(VkDevice device, BloomChain& b) {
    if (b.sampler) vkDestroySampler(device, b.sampler, nullptr);
    for (auto fb : b.framebuffers) if (fb) vkDestroyFramebuffer(device, fb, nullptr);
    if (b.upsamplePass)   vkDestroyRenderPass(device, b.upsamplePass, nullptr);
    if (b.downsamplePass) vkDestroyRenderPass(device, b.downsamplePass, nullptr);
    for (auto v : b.mipViews) if (v) vkDestroyImageView(device, v, nullptr);
    if (b.image.image)  vkDestroyImage(device, b.image.image, nullptr);
    if (b.image.memory) vkFreeMemory(device, b.image.memory, nullptr);
    b = {};
}

// ──────────────────────────────────────────────────────────────────────────
// SSAO (target + noise + UBO)
// ──────────────────────────────────────────────────────────────────────────

struct alignas(16) SSAOUboCpu {
    glm::mat4 projection;
    glm::mat4 invProjection;
    glm::vec4 samples[SSAO_KERNEL_SIZE];
    glm::vec2 noiseScale;
    float radius;
    float bias;
    float intensity;
    float pad0, pad1, pad2;
};

static void uploadNoise(SSAOTarget& s, VkPhysicalDevice physicalDevice, VkDevice device,
                        VkCommandPool commandPool, VkQueue queue)
{
    constexpr int N = 4;
    constexpr VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    uint16_t noise[N * N * 4];
    for (int i = 0; i < N * N; ++i) {
        glm::vec3 v(dist(rng), dist(rng), 0.0f);
        v = glm::normalize(v);
        // half-float storage: convert via glm::packHalf would be ideal; but here R16G16B16A16_SFLOAT
        // requires actual half values. Use __builtin_convertvector? Cleanly: use vk format with float storage.
        // Simpler: use R32G32B32A32_SFLOAT for the 4x4 noise — tiny memory cost, avoids half conversion.
        (void)v;
    }
    // Switch to float32 noise — simpler, only 256 bytes
    constexpr VkFormat fmt32 = VK_FORMAT_R32G32B32A32_SFLOAT;
    float pixels[N * N * 4];
    std::mt19937 rng2(0xC0FFEE);
    std::uniform_real_distribution<float> dist2(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        glm::vec3 v(dist2(rng2), dist2(rng2), 0.0f);
        v = glm::normalize(v);
        pixels[i*4+0] = v.x; pixels[i*4+1] = v.y; pixels[i*4+2] = v.z; pixels[i*4+3] = 0.0f;
    }

    s.noiseImage = createImage(physicalDevice, device, N, N, 1, fmt32,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    s.noiseView = createImageView(device, s.noiseImage.image, fmt32,
                                  VK_IMAGE_ASPECT_COLOR_BIT, 1);

    auto staging = createBuffer(physicalDevice, device, sizeof(pixels),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped;
    vkMapMemory(device, staging.memory, 0, sizeof(pixels), 0, &mapped);
    std::memcpy(mapped, pixels, sizeof(pixels));
    vkUnmapMemory(device, staging.memory);

    transitionImageLayout(device, commandPool, queue, s.noiseImage.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);

    auto cmd = beginSingleTimeCommands(device, commandPool);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { N, N, 1 };
    vkCmdCopyBufferToImage(cmd, staging.buffer, s.noiseImage.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(device, commandPool, queue, cmd);

    transitionImageLayout(device, commandPool, queue, s.noiseImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    destroyBuffer(device, staging);
    s.noiseSampler = makeNoiseSampler(device);
}

void createSSAO(SSAOTarget& s, VkPhysicalDevice physicalDevice, VkDevice device,
                VkCommandPool commandPool, VkQueue queue,
                VkExtent2D mainExtent, uint32_t framesInFlight)
{
    s.extent = { std::max(1u, mainExtent.width / 2), std::max(1u, mainExtent.height / 2) };

    constexpr VkFormat fmt = VK_FORMAT_R8_UNORM;

    s.image = createImage(physicalDevice, device, s.extent.width, s.extent.height, 1,
        fmt, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    s.view = createImageView(device, s.image.image, fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1);

    s.renderPass = makeColorOnlyPass(device, fmt, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                     VK_IMAGE_LAYOUT_UNDEFINED);

    VkFramebufferCreateInfo fb{};
    fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass      = s.renderPass;
    fb.attachmentCount = 1;
    fb.pAttachments    = &s.view;
    fb.width           = s.extent.width;
    fb.height          = s.extent.height;
    fb.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(device, &fb, nullptr, &s.framebuffer));

    s.sampler = makeLinearClampSampler(device);

    uploadNoise(s, physicalDevice, device, commandPool, queue);

    // Per-frame UBO
    VkDeviceSize size = sizeof(SSAOUboCpu);
    s.ubos.resize(framesInFlight);
    s.uboMapped.resize(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        s.ubos[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, s.ubos[i].memory, 0, size, 0, &s.uboMapped[i]);
    }
}

void destroySSAO(VkDevice device, SSAOTarget& s) {
    for (auto& b : s.ubos) destroyBuffer(device, b);
    s.ubos.clear();
    s.uboMapped.clear();
    if (s.noiseSampler) vkDestroySampler(device, s.noiseSampler, nullptr);
    if (s.noiseView)    vkDestroyImageView(device, s.noiseView, nullptr);
    if (s.noiseImage.image)  { vkDestroyImage(device, s.noiseImage.image, nullptr);
                                vkFreeMemory(device, s.noiseImage.memory, nullptr); }
    if (s.sampler)     vkDestroySampler(device, s.sampler, nullptr);
    if (s.framebuffer) vkDestroyFramebuffer(device, s.framebuffer, nullptr);
    if (s.renderPass)  vkDestroyRenderPass(device, s.renderPass, nullptr);
    if (s.view)        vkDestroyImageView(device, s.view, nullptr);
    if (s.image.image) { vkDestroyImage(device, s.image.image, nullptr);
                          vkFreeMemory(device, s.image.memory, nullptr); }
    s = {};
}

// ──────────────────────────────────────────────────────────────────────────
// Composite UBO
// ──────────────────────────────────────────────────────────────────────────

struct CompositeUboCpu {
    float exposure;
    float bloomStrength;
    float ssaoStrength;
    float vignetteIntensity;
    float vignetteFalloff;
    float saturation;
    float contrast;
    float gamma;
    glm::vec4 colorBalance;
    int   tonemapMode;
    int   enableSSAO;
    int   enableBloom;
    int   debugView;
};

void createCompositeData(CompositeData& c, VkPhysicalDevice physicalDevice, VkDevice device,
                         uint32_t framesInFlight)
{
    VkDeviceSize size = sizeof(CompositeUboCpu);
    c.ubos.resize(framesInFlight);
    c.uboMapped.resize(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        c.ubos[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, c.ubos[i].memory, 0, size, 0, &c.uboMapped[i]);
    }
}

void destroyCompositeData(VkDevice device, CompositeData& c) {
    for (auto& b : c.ubos) destroyBuffer(device, b);
    c.ubos.clear();
    c.uboMapped.clear();
}

// ──────────────────────────────────────────────────────────────────────────
// LDR target (composite writes here; FXAA reads from here)
// ──────────────────────────────────────────────────────────────────────────

void createLdrTarget(LdrTarget& t, VkPhysicalDevice physicalDevice, VkDevice device,
                     VkExtent2D extent)
{
    t.extent = extent;
    t.image = createImage(physicalDevice, device, extent.width, extent.height, 1,
        t.format, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    t.view = createImageView(device, t.image.image, t.format, VK_IMAGE_ASPECT_COLOR_BIT, 1);

    t.renderPass = makeColorOnlyPass(device, t.format, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                     VK_IMAGE_LAYOUT_UNDEFINED);

    VkFramebufferCreateInfo fb{};
    fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass      = t.renderPass;
    fb.attachmentCount = 1;
    fb.pAttachments    = &t.view;
    fb.width           = extent.width;
    fb.height          = extent.height;
    fb.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(device, &fb, nullptr, &t.framebuffer));

    t.sampler = makeLinearClampSampler(device);
}

void destroyLdrTarget(VkDevice device, LdrTarget& t) {
    if (t.sampler)     vkDestroySampler(device, t.sampler, nullptr);
    if (t.framebuffer) vkDestroyFramebuffer(device, t.framebuffer, nullptr);
    if (t.renderPass)  vkDestroyRenderPass(device, t.renderPass, nullptr);
    if (t.view)        vkDestroyImageView(device, t.view, nullptr);
    if (t.image.image) {
        vkDestroyImage(device, t.image.image, nullptr);
        vkFreeMemory(device, t.image.memory, nullptr);
    }
    t = {};
}

void updateCompositeUbo(CompositeData& c, uint32_t frameIndex, const PostFXSettings& s) {
    CompositeUboCpu u{};
    u.exposure          = s.exposure;
    u.bloomStrength     = s.bloomStrength;
    u.ssaoStrength      = s.ssaoStrength;
    u.vignetteIntensity = s.vignetteIntensity;
    u.vignetteFalloff   = s.vignetteFalloff;
    u.saturation        = s.saturation;
    u.contrast          = s.contrast;
    u.gamma             = s.gamma;
    u.colorBalance      = glm::vec4(s.colorBalance, 1.0f);
    u.tonemapMode       = s.tonemapMode;
    u.enableSSAO        = s.ssaoEnabled  ? 1 : 0;
    u.enableBloom       = s.bloomEnabled ? 1 : 0;
    u.debugView         = s.debugView;
    std::memcpy(c.uboMapped[frameIndex], &u, sizeof(u));
}

// ──────────────────────────────────────────────────────────────────────────
// SSAO UBO update
// ──────────────────────────────────────────────────────────────────────────

void updateSSAOUbo(SSAOTarget& s, uint32_t frameIndex, const glm::mat4& projection,
                   const glm::mat4& invProjection, VkExtent2D mainExtent,
                   const PostFXSettings& settings)
{
    static SSAOUboCpu u{};
    static bool kernelGenerated = false;
    if (!kernelGenerated) {
        std::mt19937 rng(0xBEEF);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < SSAO_KERNEL_SIZE; ++i) {
            glm::vec3 sample(dist(rng) * 2.0f - 1.0f,
                             dist(rng) * 2.0f - 1.0f,
                             dist(rng));
            sample = glm::normalize(sample);
            sample *= dist(rng);
            float scale = float(i) / float(SSAO_KERNEL_SIZE);
            scale = 0.1f + (scale * scale) * 0.9f; // accelerate towards center
            sample *= scale;
            u.samples[i] = glm::vec4(sample, 0.0f);
        }
        kernelGenerated = true;
    }

    u.projection    = projection;
    u.invProjection = invProjection;
    u.noiseScale    = glm::vec2(static_cast<float>(mainExtent.width)  / 8.0f,
                                static_cast<float>(mainExtent.height) / 8.0f);
    u.radius        = settings.ssaoRadius;
    u.bias          = settings.ssaoBias;
    u.intensity     = settings.ssaoIntensity;
    std::memcpy(s.uboMapped[frameIndex], &u, sizeof(u));
}

// ──────────────────────────────────────────────────────────────────────────
// Pipelines + descriptor sets
// ──────────────────────────────────────────────────────────────────────────

static VkDescriptorSetLayout makeDescriptorSetLayout(VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings)
{
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.pBindings    = bindings.data();
    VkDescriptorSetLayout out;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &out));
    return out;
}

struct BloomPC {
    glm::vec2 texelSize;
    float threshold;
    float pad;
};
struct BloomUpsamplePC {
    glm::vec2 texelSize;
    float filterRadius;
    float pad;
};

static VkPipeline makeFullscreenPipeline(VkDevice device, VkRenderPass renderPass,
                                         VkPipelineLayout layout,
                                         const std::string& fragPath,
                                         bool additiveBlend = false,
                                         VkExtent2D viewportExtent = { 0, 0 })
{
    auto vs = loadShaderModule(device, "shaders/fullscreen.vert.spv");
    auto fs = loadShaderModule(device, fragPath);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                         | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (additiveBlend) {
        blend.blendEnable         = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.colorBlendOp        = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend.alphaBlendOp        = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &blend;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dyn;

    VkGraphicsPipelineCreateInfo info{};
    info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount          = 2;
    info.pStages             = stages;
    info.pVertexInputState   = &vi;
    info.pInputAssemblyState = &ia;
    info.pViewportState      = &vp;
    info.pRasterizationState = &rs;
    info.pMultisampleState   = &ms;
    info.pDepthStencilState  = &ds;
    info.pColorBlendState    = &cb;
    info.pDynamicState       = &dynState;
    info.layout              = layout;
    info.renderPass          = renderPass;
    info.subpass             = 0;

    VkPipeline out;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &out));

    vkDestroyShaderModule(device, fs, nullptr);
    vkDestroyShaderModule(device, vs, nullptr);
    (void)viewportExtent;
    return out;
}

void createPostFXPipelines(PostFXPipelines& p, VkDevice device, BloomChain& bloom,
                           SSAOTarget& ssao, CompositeData& composite, LdrTarget& ldr,
                           OffscreenTarget& offscreen, VkRenderPass swapchainPass,
                           uint32_t framesInFlight)
{
    uint32_t bloomSets    = BLOOM_MIP_COUNT * 2;
    uint32_t totalSets    = bloomSets + framesInFlight /*ssao*/ + framesInFlight /*composite*/ + 1 /*fxaa*/;

    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          bloomSets + framesInFlight * 2 /*ssao*/ + framesInFlight * 3 /*comp*/ + 1 /*fxaa*/ },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         framesInFlight + framesInFlight },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = totalSets;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes    = sizes;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &p.descriptorPool));

    // ── Bloom set layout: 1 sampler ─────────────────────────────────────
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        p.bloomSetLayout = makeDescriptorSetLayout(device, { b });
    }

    // Bloom layouts: push constants for params
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(BloomPC);

    VkPipelineLayoutCreateInfo lci{};
    lci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount         = 1;
    lci.pSetLayouts            = &p.bloomSetLayout;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &p.bloomLayout));

    p.bloomDownsample = makeFullscreenPipeline(device, bloom.downsamplePass, p.bloomLayout,
                                               "shaders/bloom_downsample.frag.spv");
    p.bloomUpsample   = makeFullscreenPipeline(device, bloom.upsamplePass,   p.bloomLayout,
                                               "shaders/bloom_upsample.frag.spv", true);

    // ── SSAO set layout: depth (0), noise (1), UBO (2) ──────────────────
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(3);
        bindings[0] = {}; bindings[0].binding = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1; bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1] = bindings[0]; bindings[1].binding = 1;
        bindings[2] = {}; bindings[2].binding = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[2].descriptorCount = 1; bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        p.ssaoSetLayout = makeDescriptorSetLayout(device, bindings);
    }
    {
        VkPipelineLayoutCreateInfo lci2{};
        lci2.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci2.setLayoutCount         = 1;
        lci2.pSetLayouts            = &p.ssaoSetLayout;
        VK_CHECK(vkCreatePipelineLayout(device, &lci2, nullptr, &p.ssaoLayout));
    }
    p.ssao = makeFullscreenPipeline(device, ssao.renderPass, p.ssaoLayout,
                                    "shaders/ssao.frag.spv");

    // ── Composite set layout: hdr (0), bloom (1), ssao (2), UBO (3) ─────
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(4);
        for (int i = 0; i < 3; ++i) {
            bindings[i] = {}; bindings[i].binding = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        bindings[3] = {}; bindings[3].binding = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[3].descriptorCount = 1; bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        p.compositeSetLayout = makeDescriptorSetLayout(device, bindings);
    }
    {
        VkPipelineLayoutCreateInfo lci3{};
        lci3.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci3.setLayoutCount         = 1;
        lci3.pSetLayouts            = &p.compositeSetLayout;
        VK_CHECK(vkCreatePipelineLayout(device, &lci3, nullptr, &p.compositeLayout));
    }
    // Composite targets the LDR render pass (not the swapchain directly anymore)
    p.composite = makeFullscreenPipeline(device, ldr.renderPass, p.compositeLayout,
                                         "shaders/composite.frag.spv");

    // ── FXAA: 1 set with 1 sampler (LDR input) ──────────────────────────
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        p.fxaaSetLayout = makeDescriptorSetLayout(device, { b });
    }
    {
        // Push constant: vec4 (texelSize.xy, enable, _)
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(glm::vec4);

        VkPipelineLayoutCreateInfo lci{};
        lci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount         = 1;
        lci.pSetLayouts            = &p.fxaaSetLayout;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges    = &pcr;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &p.fxaaLayout));
    }
    p.fxaa = makeFullscreenPipeline(device, swapchainPass, p.fxaaLayout,
                                    "shaders/fxaa.frag.spv");

    // ── Allocate descriptor sets ────────────────────────────────────────
    auto allocSet = [&](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = p.descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &layout;
        VkDescriptorSet set;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &set));
        return set;
    };

    // Bloom downsample sets: mip 0 reads HDR, mip i (>0) reads mip i-1
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i) {
        bloom.downsampleSets[i] = allocSet(p.bloomSetLayout);
        VkDescriptorImageInfo img{};
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img.imageView   = (i == 0) ? offscreen.colorView : bloom.mipViews[i - 1];
        img.sampler     = bloom.sampler;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = bloom.downsampleSets[i]; w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1; w.pImageInfo = &img;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
    // Bloom upsample sets: mip i reads mip i+1 (used for mip < N-1)
    for (uint32_t i = 0; i + 1 < BLOOM_MIP_COUNT; ++i) {
        bloom.upsampleSets[i] = allocSet(p.bloomSetLayout);
        VkDescriptorImageInfo img{};
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img.imageView   = bloom.mipViews[i + 1];
        img.sampler     = bloom.sampler;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = bloom.upsampleSets[i]; w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1; w.pImageInfo = &img;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    // SSAO sets (per-frame UBO)
    ssao.descriptorSets.resize(framesInFlight);
    for (uint32_t f = 0; f < framesInFlight; ++f) {
        ssao.descriptorSets[f] = allocSet(p.ssaoSetLayout);
        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthInfo.imageView   = offscreen.depthView;
        depthInfo.sampler     = offscreen.depthSampler;

        VkDescriptorImageInfo noiseInfo{};
        noiseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        noiseInfo.imageView   = ssao.noiseView;
        noiseInfo.sampler     = ssao.noiseSampler;

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = ssao.ubos[f].buffer;
        bufInfo.range  = sizeof(SSAOUboCpu);

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = ssao.descriptorSets[f]; writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1; writes[0].pImageInfo = &depthInfo;
        writes[1] = writes[0]; writes[1].dstBinding = 1; writes[1].pImageInfo = &noiseInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = ssao.descriptorSets[f]; writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1; writes[2].pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }

    // FXAA descriptor set (reads from LDR target)
    {
        ldr.descriptorSet = allocSet(p.fxaaSetLayout);
        VkDescriptorImageInfo info{};
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info.imageView   = ldr.view;
        info.sampler     = ldr.sampler;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ldr.descriptorSet; w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1; w.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    // Composite sets (per-frame UBO)
    composite.descriptorSets.resize(framesInFlight);
    for (uint32_t f = 0; f < framesInFlight; ++f) {
        composite.descriptorSets[f] = allocSet(p.compositeSetLayout);

        VkDescriptorImageInfo hdrInfo{};
        hdrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hdrInfo.imageView   = offscreen.colorView;
        hdrInfo.sampler     = offscreen.sampler;

        VkDescriptorImageInfo bloomInfo{};
        bloomInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bloomInfo.imageView   = bloom.mipViews[0];
        bloomInfo.sampler     = bloom.sampler;

        VkDescriptorImageInfo ssaoInfo{};
        ssaoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ssaoInfo.imageView   = ssao.view;
        ssaoInfo.sampler     = ssao.sampler;

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = composite.ubos[f].buffer;
        bufInfo.range  = sizeof(CompositeUboCpu);

        VkWriteDescriptorSet writes[4]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = composite.descriptorSets[f]; writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1; writes[0].pImageInfo = &hdrInfo;
        writes[1] = writes[0]; writes[1].dstBinding = 1; writes[1].pImageInfo = &bloomInfo;
        writes[2] = writes[0]; writes[2].dstBinding = 2; writes[2].pImageInfo = &ssaoInfo;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = composite.descriptorSets[f]; writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3].descriptorCount = 1; writes[3].pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }
}

void destroyPostFXPipelines(VkDevice device, PostFXPipelines& p) {
    if (p.fxaa)              vkDestroyPipeline(device, p.fxaa, nullptr);
    if (p.fxaaLayout)        vkDestroyPipelineLayout(device, p.fxaaLayout, nullptr);
    if (p.fxaaSetLayout)     vkDestroyDescriptorSetLayout(device, p.fxaaSetLayout, nullptr);
    if (p.composite)         vkDestroyPipeline(device, p.composite, nullptr);
    if (p.compositeLayout)   vkDestroyPipelineLayout(device, p.compositeLayout, nullptr);
    if (p.compositeSetLayout)vkDestroyDescriptorSetLayout(device, p.compositeSetLayout, nullptr);
    if (p.ssao)              vkDestroyPipeline(device, p.ssao, nullptr);
    if (p.ssaoLayout)        vkDestroyPipelineLayout(device, p.ssaoLayout, nullptr);
    if (p.ssaoSetLayout)     vkDestroyDescriptorSetLayout(device, p.ssaoSetLayout, nullptr);
    if (p.bloomDownsample)   vkDestroyPipeline(device, p.bloomDownsample, nullptr);
    if (p.bloomUpsample)     vkDestroyPipeline(device, p.bloomUpsample, nullptr);
    if (p.bloomLayout)       vkDestroyPipelineLayout(device, p.bloomLayout, nullptr);
    if (p.bloomSetLayout)    vkDestroyDescriptorSetLayout(device, p.bloomSetLayout, nullptr);
    if (p.descriptorPool)    vkDestroyDescriptorPool(device, p.descriptorPool, nullptr);
    p = {};
}

// ──────────────────────────────────────────────────────────────────────────
// Recording helpers
// ──────────────────────────────────────────────────────────────────────────

static void setFullViewportScissor(VkCommandBuffer cmd, VkExtent2D extent) {
    VkViewport vp{};
    vp.width = static_cast<float>(extent.width);
    vp.height = static_cast<float>(extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{}; sc.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

void recordSSAO(VkCommandBuffer cmd, SSAOTarget& s, PostFXPipelines& p, uint32_t frameIndex) {
    VkRenderPassBeginInfo rp{};
    rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass        = s.renderPass;
    rp.framebuffer       = s.framebuffer;
    rp.renderArea.extent = s.extent;

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.ssao);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.ssaoLayout,
                            0, 1, &s.descriptorSets[frameIndex], 0, nullptr);
    setFullViewportScissor(cmd, s.extent);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void recordBloom(VkCommandBuffer cmd, BloomChain& b, PostFXPipelines& p,
                 const PostFXSettings& settings)
{
    // ── Downsample: HDR → mip0 → mip1 → ... ─────────────────────────────
    for (uint32_t i = 0; i < BLOOM_MIP_COUNT; ++i) {
        VkRenderPassBeginInfo rp{};
        rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass        = b.downsamplePass;
        rp.framebuffer       = b.framebuffers[i];
        rp.renderArea.extent = b.extents[i];

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.bloomDownsample);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.bloomLayout,
                                0, 1, &b.downsampleSets[i], 0, nullptr);

        BloomPC pc{};
        // texelSize = 1 / sourceSize (source is the HDR for mip 0, otherwise the previous mip)
        VkExtent2D src = (i == 0) ? VkExtent2D{ b.extents[0].width  * 2,
                                                b.extents[0].height * 2 }
                                   : b.extents[i - 1];
        pc.texelSize = glm::vec2(1.0f / src.width, 1.0f / src.height);
        pc.threshold = (i == 0) ? settings.bloomThreshold : 0.0f;
        vkCmdPushConstants(cmd, p.bloomLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);

        setFullViewportScissor(cmd, b.extents[i]);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

    // ── Upsample (additive): from smallest back to mip 0 ────────────────
    for (int i = static_cast<int>(BLOOM_MIP_COUNT) - 2; i >= 0; --i) {
        VkRenderPassBeginInfo rp{};
        rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass        = b.upsamplePass;
        rp.framebuffer       = b.framebuffers[i];
        rp.renderArea.extent = b.extents[i];

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.bloomUpsample);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.bloomLayout,
                                0, 1, &b.upsampleSets[i], 0, nullptr);

        BloomUpsamplePC pc{};
        pc.texelSize    = glm::vec2(1.0f / b.extents[i + 1].width,
                                    1.0f / b.extents[i + 1].height);
        pc.filterRadius = settings.bloomFilterRadius;
        vkCmdPushConstants(cmd, p.bloomLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);

        setFullViewportScissor(cmd, b.extents[i]);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }
}
