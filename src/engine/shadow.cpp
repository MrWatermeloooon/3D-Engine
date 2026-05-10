#include "shadow.h"
#include "../utils/vk_check.h"

#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cstring>
#include <limits>

static void createShadowImage(ShadowData& s, VkPhysicalDevice physicalDevice, VkDevice device) {
    VkImageCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType     = VK_IMAGE_TYPE_2D;
    info.format        = SHADOW_FORMAT;
    info.extent        = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
    info.mipLevels     = 1;
    info.arrayLayers   = SHADOW_CASCADE_COUNT;
    info.samples       = VK_SAMPLE_COUNT_1_BIT;
    info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    info.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(device, &info, nullptr, &s.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, s.image, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &alloc, nullptr, &s.memory));
    VK_CHECK(vkBindImageMemory(device, s.image, s.memory, 0));

    // Array view (sampled in main pass)
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = s.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format   = SHADOW_FORMAT;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = SHADOW_CASCADE_COUNT;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &s.arrayView));

    // Per-layer views (used as framebuffer attachments)
    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.subresourceRange.baseArrayLayer = i;
        viewInfo.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &s.layerViews[i]));
    }
}

static void createShadowRenderPass(ShadowData& s, VkDevice device) {
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format         = SHADOW_FORMAT;
    depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 0;
    depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 0;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments    = &depthAttachment;
    rp.subpassCount    = 1;
    rp.pSubpasses      = &subpass;
    rp.dependencyCount = 2;
    rp.pDependencies   = deps;

    VK_CHECK(vkCreateRenderPass(device, &rp, nullptr, &s.renderPass));
}

static void createShadowFramebuffers(ShadowData& s, VkDevice device) {
    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        VkFramebufferCreateInfo fb{};
        fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass      = s.renderPass;
        fb.attachmentCount = 1;
        fb.pAttachments    = &s.layerViews[i];
        fb.width           = SHADOW_MAP_SIZE;
        fb.height          = SHADOW_MAP_SIZE;
        fb.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(device, &fb, nullptr, &s.framebuffers[i]));
    }
}

static void createShadowSampler(ShadowData& s, VkDevice device) {
    VkSamplerCreateInfo info{};
    info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter    = VK_FILTER_LINEAR;
    info.minFilter    = VK_FILTER_LINEAR;
    info.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    info.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // outside cascade = lit
    info.compareEnable= VK_TRUE;
    info.compareOp    = VK_COMPARE_OP_LESS;
    info.minLod       = 0.0f;
    info.maxLod       = 1.0f;

    VK_CHECK(vkCreateSampler(device, &info, nullptr, &s.sampler));
}

static void createCascadeBuffers(ShadowData& s, VkPhysicalDevice physicalDevice,
                                 VkDevice device, uint32_t framesInFlight)
{
    VkDeviceSize size = sizeof(CascadeUBO);
    s.cascadeBuffers.resize(framesInFlight);
    s.cascadeMapped.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        s.cascadeBuffers[i] = createBuffer(physicalDevice, device, size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, s.cascadeBuffers[i].memory, 0, size, 0, &s.cascadeMapped[i]);
    }
}

void createShadowResources(ShadowData& shadow, VkPhysicalDevice physicalDevice,
                           VkDevice device, uint32_t framesInFlight)
{
    createShadowImage(shadow, physicalDevice, device);
    createShadowRenderPass(shadow, device);
    createShadowFramebuffers(shadow, device);
    createShadowSampler(shadow, device);
    createCascadeBuffers(shadow, physicalDevice, device, framesInFlight);
}

void destroyShadowResources(VkDevice device, ShadowData& shadow) {
    for (auto& buf : shadow.cascadeBuffers) destroyBuffer(device, buf);
    shadow.cascadeBuffers.clear();
    shadow.cascadeMapped.clear();

    if (shadow.sampler)    vkDestroySampler(device, shadow.sampler, nullptr);
    for (auto fb : shadow.framebuffers) if (fb) vkDestroyFramebuffer(device, fb, nullptr);
    if (shadow.renderPass) vkDestroyRenderPass(device, shadow.renderPass, nullptr);
    for (auto v : shadow.layerViews) if (v) vkDestroyImageView(device, v, nullptr);
    if (shadow.arrayView)  vkDestroyImageView(device, shadow.arrayView, nullptr);
    if (shadow.image)      vkDestroyImage(device, shadow.image, nullptr);
    if (shadow.memory)     vkFreeMemory(device, shadow.memory, nullptr);

    shadow = {};
}

// ── Cascade computation ─────────────────────────────────────────────────────
// Practical split scheme: lerp between uniform and logarithmic splits.

void computeCascades(ShadowData& shadow, uint32_t frameIndex,
                     const glm::mat4& cameraView, const glm::mat4& cameraProj,
                     float nearClip, float farClip,
                     const glm::vec3& lightDir,
                     CascadeUBO& outUbo)
{
    float clipRange = farClip - nearClip;
    float minZ = nearClip;
    float maxZ = nearClip + clipRange;
    float range = maxZ - minZ;
    float ratio = maxZ / minZ;

    float cascadeSplits[SHADOW_CASCADE_COUNT];
    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        float p = (i + 1) / static_cast<float>(SHADOW_CASCADE_COUNT);
        float log = minZ * std::pow(ratio, p);
        float uniform = minZ + range * p;
        float d = shadow.cascadeSplitLambda * (log - uniform) + uniform;
        cascadeSplits[i] = (d - nearClip) / clipRange; // normalized [0,1]
    }

    glm::mat4 invCam = glm::inverse(cameraProj * cameraView);
    glm::vec3 ldir = glm::normalize(lightDir);

    // Pick a stable up axis that isn't parallel to the light direction.
    glm::vec3 up = (std::abs(ldir.y) > 0.99f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

    float lastSplitDist = 0.0f;
    for (uint32_t c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
        float splitDist = cascadeSplits[c];

        glm::vec3 frustumCorners[8] = {
            {-1.0f,  1.0f, 0.0f}, { 1.0f,  1.0f, 0.0f},
            { 1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f},
            {-1.0f,  1.0f, 1.0f}, { 1.0f,  1.0f, 1.0f},
            { 1.0f, -1.0f, 1.0f}, {-1.0f, -1.0f, 1.0f},
        };

        for (auto& fc : frustumCorners) {
            glm::vec4 inv = invCam * glm::vec4(fc, 1.0f);
            fc = glm::vec3(inv) / inv.w;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            glm::vec3 dist = frustumCorners[i + 4] - frustumCorners[i];
            frustumCorners[i + 4] = frustumCorners[i] + dist * splitDist;
            frustumCorners[i]     = frustumCorners[i] + dist * lastSplitDist;
        }

        // Bounding sphere → rotation-stable bounds
        glm::vec3 center{0.0f};
        for (auto& fc : frustumCorners) center += fc;
        center /= 8.0f;

        float radius = 0.0f;
        for (auto& fc : frustumCorners)
            radius = std::max(radius, glm::length(fc - center));
        radius = std::ceil(radius * 16.0f) / 16.0f;

        // Texel-snap the sphere center in light space → translation-stable bounds.
        // Without this, shadows shimmer as the camera moves.
        glm::mat4 lightViewUnsnapped = glm::lookAt(glm::vec3(0.0f), ldir, up);
        glm::vec3 centerLS = glm::vec3(lightViewUnsnapped * glm::vec4(center, 1.0f));
        float worldUnitsPerTexel = (radius * 2.0f) / static_cast<float>(SHADOW_MAP_SIZE);
        centerLS.x = std::floor(centerLS.x / worldUnitsPerTexel) * worldUnitsPerTexel;
        centerLS.y = std::floor(centerLS.y / worldUnitsPerTexel) * worldUnitsPerTexel;
        glm::vec3 snappedCenter = glm::vec3(glm::inverse(lightViewUnsnapped) * glm::vec4(centerLS, 1.0f));

        glm::mat4 lightView = glm::lookAt(snappedCenter - ldir * radius, snappedCenter, up);
        glm::mat4 lightProj = glm::ortho(-radius, radius, -radius, radius,
                                         0.0f, radius * 2.0f);

        outUbo.lightViewProj[c] = lightProj * lightView;
        outUbo.splitsViewSpace[c] = -(nearClip + splitDist * clipRange);

        lastSplitDist = cascadeSplits[c];
    }

    std::memcpy(shadow.cascadeMapped[frameIndex], &outUbo, sizeof(CascadeUBO));
}
