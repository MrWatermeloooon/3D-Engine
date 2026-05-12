#include "hzb.h"
#include "pipeline.h"
#include "../utils/vk_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

static uint32_t computeMipCount(VkExtent2D extent) {
    uint32_t maxDim = std::max(extent.width, extent.height);
    return static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(maxDim)))) + 1;
}

static VkExtent2D mipExtent(VkExtent2D base, uint32_t level) {
    return {
        std::max(1u, base.width  >> level),
        std::max(1u, base.height >> level),
    };
}

void createHzb(HzbData& hzb,
               VkPhysicalDevice physicalDevice, VkDevice device,
               VkCommandPool commandPool, VkQueue queue,
               VkExtent2D extent,
               VkImageView depthView, VkSampler depthSampler,
               const std::string& reduceShaderPath)
{
    hzb.extent   = extent;
    hzb.mipCount = computeMipCount(extent);

    // ── Image ───────────────────────────────────────────────────────────────
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R32_SFLOAT;
    ici.extent        = { extent.width, extent.height, 1 };
    ici.mipLevels     = hzb.mipCount;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_STORAGE_BIT
                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &ici, nullptr, &hzb.image.image));

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(device, hzb.image.image, &memReq);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &hzb.image.memory));
    VK_CHECK(vkBindImageMemory(device, hzb.image.image, hzb.image.memory, 0));

    // ── Views ───────────────────────────────────────────────────────────────
    // Full chain view (for sampling in cull).
    {
        VkImageViewCreateInfo ivi{};
        ivi.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivi.image                       = hzb.image.image;
        ivi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format                      = VK_FORMAT_R32_SFLOAT;
        ivi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivi.subresourceRange.baseMipLevel   = 0;
        ivi.subresourceRange.levelCount     = hzb.mipCount;
        ivi.subresourceRange.baseArrayLayer = 0;
        ivi.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &ivi, nullptr, &hzb.fullView));
    }

    // Per-mip single-level views (for storage writes during reduction).
    hzb.mipViews.resize(hzb.mipCount);
    for (uint32_t i = 0; i < hzb.mipCount; ++i) {
        VkImageViewCreateInfo ivi{};
        ivi.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivi.image                       = hzb.image.image;
        ivi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format                      = VK_FORMAT_R32_SFLOAT;
        ivi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivi.subresourceRange.baseMipLevel   = i;
        ivi.subresourceRange.levelCount     = 1;
        ivi.subresourceRange.baseArrayLayer = 0;
        ivi.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &ivi, nullptr, &hzb.mipViews[i]));
    }

    // ── Sampler ─────────────────────────────────────────────────────────────
    {
        VkSamplerCreateInfo si{};
        si.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter     = VK_FILTER_NEAREST;
        si.minFilter     = VK_FILTER_NEAREST;
        si.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod        = 0.0f;
        si.maxLod        = static_cast<float>(hzb.mipCount);
        VK_CHECK(vkCreateSampler(device, &si, nullptr, &hzb.sampler));
    }

    // ── Reduction descriptor set layout (binding 0 = sampler2D, 1 = image2D) ─
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl{};
    dsl.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = static_cast<uint32_t>(bindings.size());
    dsl.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl, nullptr, &hzb.reduceSetLayout));

    // ── Pipeline layout (1 set + push constants) ───────────────────────────
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(HzbReducePushConstants);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &hzb.reduceSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &hzb.reducePipelineLayout));

    // ── Pipeline ────────────────────────────────────────────────────────────
    {
        VkShaderModule shader = loadShaderModule(device, reduceShaderPath);
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpi{};
        cpi.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage  = stage;
        cpi.layout = hzb.reducePipelineLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr,
                                          &hzb.reducePipeline));
        vkDestroyShaderModule(device, shader, nullptr);
    }

    // ── Descriptor pool + per-mip sets ─────────────────────────────────────
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = hzb.mipCount;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = hzb.mipCount;

    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets       = hzb.mipCount;
    dpi.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    dpi.pPoolSizes    = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device, &dpi, nullptr, &hzb.reduceDescriptorPool));

    std::vector<VkDescriptorSetLayout> layouts(hzb.mipCount, hzb.reduceSetLayout);
    VkDescriptorSetAllocateInfo dai{};
    dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool     = hzb.reduceDescriptorPool;
    dai.descriptorSetCount = hzb.mipCount;
    dai.pSetLayouts        = layouts.data();
    hzb.reduceSets.resize(hzb.mipCount);
    VK_CHECK(vkAllocateDescriptorSets(device, &dai, hzb.reduceSets.data()));

    // Write each descriptor set. Mip 0 sources from depth attachment; mip i>0
    // sources from HZB mip i-1.
    for (uint32_t i = 0; i < hzb.mipCount; ++i) {
        VkDescriptorImageInfo src{};
        src.imageLayout = (i == 0)
            ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            : VK_IMAGE_LAYOUT_GENERAL;
        src.imageView   = (i == 0) ? depthView    : hzb.mipViews[i - 1];
        src.sampler     = (i == 0) ? depthSampler : hzb.sampler;

        VkDescriptorImageInfo dst{};
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        dst.imageView   = hzb.mipViews[i];

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet          = hzb.reduceSets[i];
        w[0].dstBinding      = 0;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].descriptorCount = 1;
        w[0].pImageInfo      = &src;
        w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet          = hzb.reduceSets[i];
        w[1].dstBinding      = 1;
        w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].descriptorCount = 1;
        w[1].pImageInfo      = &dst;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
    }

    // ── Initialize image to GENERAL layout, depth=1.0 ──────────────────────
    // First-frame cull samples this before any reduction has happened; clearing
    // to max depth means "nothing occludes anything", which is the safe default.
    {
        VkCommandBuffer setup = beginSingleTimeCommands(device, commandPool);

        VkImageMemoryBarrier toTransfer{};
        toTransfer.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toTransfer.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.image            = hzb.image.image;
        toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransfer.subresourceRange.levelCount = hzb.mipCount;
        toTransfer.subresourceRange.layerCount = 1;
        toTransfer.srcAccessMask    = 0;
        toTransfer.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(setup,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toTransfer);

        VkClearColorValue clearVal{};
        clearVal.float32[0] = 1.0f;
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = hzb.mipCount;
        range.layerCount = 1;
        vkCmdClearColorImage(setup, hzb.image.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearVal, 1, &range);

        VkImageMemoryBarrier toGeneral = toTransfer;
        toGeneral.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(setup,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toGeneral);

        endSingleTimeCommands(device, commandPool, queue, setup);
    }

    std::cout << "[VulkanEngine] HZB created (" << extent.width << "×" << extent.height
              << ", " << hzb.mipCount << " mips)\n";
}

void destroyHzb(VkDevice device, HzbData& hzb) {
    if (hzb.reduceDescriptorPool) {
        vkDestroyDescriptorPool(device, hzb.reduceDescriptorPool, nullptr);
        hzb.reduceDescriptorPool = VK_NULL_HANDLE;
    }
    hzb.reduceSets.clear();
    if (hzb.reducePipeline)        vkDestroyPipeline(device, hzb.reducePipeline, nullptr);
    if (hzb.reducePipelineLayout)  vkDestroyPipelineLayout(device, hzb.reducePipelineLayout, nullptr);
    if (hzb.reduceSetLayout)       vkDestroyDescriptorSetLayout(device, hzb.reduceSetLayout, nullptr);
    if (hzb.sampler)               vkDestroySampler(device, hzb.sampler, nullptr);
    for (auto v : hzb.mipViews)    if (v) vkDestroyImageView(device, v, nullptr);
    hzb.mipViews.clear();
    if (hzb.fullView)              vkDestroyImageView(device, hzb.fullView, nullptr);
    if (hzb.image.image)           vkDestroyImage(device, hzb.image.image, nullptr);
    if (hzb.image.memory)          vkFreeMemory(device, hzb.image.memory, nullptr);
    hzb = {};
}

void recordHzbReduce(VkCommandBuffer cmd, HzbData& hzb) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzb.reducePipeline);

    for (uint32_t i = 0; i < hzb.mipCount; ++i) {
        // Bind this mip's descriptor set.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                hzb.reducePipelineLayout,
                                0, 1, &hzb.reduceSets[i], 0, nullptr);

        VkExtent2D ext = mipExtent(hzb.extent, i);
        HzbReducePushConstants pc{};
        pc.dstWidth   = static_cast<int32_t>(ext.width);
        pc.dstHeight  = static_cast<int32_t>(ext.height);
        pc.isFirstMip = (i == 0) ? 1 : 0;
        vkCmdPushConstants(cmd, hzb.reducePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t gx = (ext.width  + 7) / 8;
        uint32_t gy = (ext.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);

        // Between mips: storage-write-to-sampled-read on the just-written mip.
        // After the last mip, the caller is responsible for any onward sync
        // (typically just queue submission boundary before next frame's cull).
        if (i + 1 < hzb.mipCount) {
            VkImageMemoryBarrier b{};
            b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
            b.image            = hzb.image.image;
            b.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.baseMipLevel   = i;
            b.subresourceRange.levelCount     = 1;
            b.subresourceRange.baseArrayLayer = 0;
            b.subresourceRange.layerCount     = 1;
            b.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        }
    }
}
