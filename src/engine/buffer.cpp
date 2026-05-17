#include "buffer.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <stdexcept>

// findMemoryType is kept for the few sites (manual descriptor-set allocators)
// that still query memory-type indices. VMA does its own selection internally.
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

AllocatedBuffer createBuffer(VkPhysicalDevice /*physicalDevice*/, VkDevice /*device*/,
                             VkDeviceSize size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags properties) {
    AllocatedBuffer result;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = size;
    bufferInfo.usage       = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};

    const bool wantsHostVisible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (wantsHostVisible) {
        // Persistent CPU-write target: instance buffers, candidate stream,
        // UBOs, staging. AUTO_PREFER_HOST lands these in BAR/host memory;
        // SEQUENTIAL_WRITE_BIT lets VMA pick a write-combined memory type
        // (faster on PCIe upload) when one is available. MAPPED_BIT keeps the
        // pointer alive across frames so we never touch vkMapMemory.
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        // Pure GPU-side buffer: BLAS/TLAS storage, scratch, device-local
        // staging targets, indirect drawer counters. No host map.
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }

    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(gVmaAllocator, &bufferInfo, &aci,
                             &result.buffer, &result.allocation, &info));
    // VMA places the persistent map (when requested) at allocationInfo.pMappedData.
    // For non-host-visible allocations this is null.
    result.mapped = info.pMappedData;
    return result;
}

void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    auto cmd = beginSingleTimeCommands(device, commandPool);

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    endSingleTimeCommands(device, commandPool, queue, cmd);
}

void destroyBuffer(VkDevice /*device*/, AllocatedBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(gVmaAllocator, buf.buffer, buf.allocation);
        buf.buffer     = VK_NULL_HANDLE;
        buf.allocation = VK_NULL_HANDLE;
        buf.mapped     = nullptr;
    }
}

AllocatedImage createImage(VkPhysicalDevice /*physicalDevice*/, VkDevice /*device*/,
                           uint32_t width, uint32_t height, uint32_t mipLevels,
                           VkFormat format, VkImageTiling tiling,
                           VkImageUsageFlags usage, VkMemoryPropertyFlags properties) {
    AllocatedImage result;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width  = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth  = 1;
    imageInfo.mipLevels     = mipLevels;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = format;
    imageInfo.tiling        = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    // All images in this engine are GPU-side (color/depth attachments, shadow
    // maps, HZB, SSAO noise/output). Host-visible images don't appear.
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    (void)properties; // legacy hint preserved in signature; VMA ignores

    VK_CHECK(vmaCreateImage(gVmaAllocator, &imageInfo, &aci,
                            &result.image, &result.allocation, nullptr));
    return result;
}

void destroyImage(VkDevice /*device*/, AllocatedImage& img) {
    if (img.image != VK_NULL_HANDLE) {
        vmaDestroyImage(gVmaAllocator, img.image, img.allocation);
        img.image      = VK_NULL_HANDLE;
        img.allocation = VK_NULL_HANDLE;
    }
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange.aspectMask     = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VkImageView view;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &view));
    return view;
}

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool        = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    return cmd;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue queue, VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit with a per-call fence and wait on that fence specifically.
    // Avoids vkQueueWaitIdle which stalls the entire queue (problematic during
    // hot-reload / runtime uploads while frame rendering is in flight).
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                           VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                           uint32_t mipLevels) {
    auto cmd = beginSingleTimeCommands(device, commandPool);

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::runtime_error("Unsupported layout transition");
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(device, commandPool, queue, cmd);
}
