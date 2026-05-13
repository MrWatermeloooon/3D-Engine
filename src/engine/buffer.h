#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

// VMA-backed buffer/image wrappers.
//
// Allocation is owned by `allocation` (an opaque VMA handle). The legacy
// `VkDeviceMemory memory` field has been removed — callers that used to do
// `vkMapMemory(device, buf.memory, …)` should now read the persistently-mapped
// pointer from `mapped` (populated automatically when a host-visible buffer is
// created via `createBuffer`).
//
// `destroyBuffer` / `destroyImage` go through VMA, which knows the right
// memory pool to return the allocation to. No separate `vkFreeMemory` is
// needed (or possible — there is no exposed VkDeviceMemory to free).

struct AllocatedBuffer {
    VkBuffer       buffer     = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
    void*          mapped     = nullptr;   // non-null iff host-visible and persistently mapped
};

struct AllocatedImage {
    VkImage        image      = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
};

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties);

// `properties` is interpreted to pick the right VMA usage profile:
//   * DEVICE_LOCAL_BIT (no HOST_VISIBLE)        → GPU-only buffer
//   * HOST_VISIBLE_BIT (with/without COHERENT)  → CPU-writeable, persistently mapped
//                                                 (mapped pointer placed in result.mapped)
//
// Buffers requested with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
// automatically get VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT applied by VMA.
// `physicalDevice` is kept in the signature for API stability and ignored by
// the new implementation.
AllocatedBuffer createBuffer(VkPhysicalDevice physicalDevice, VkDevice device,
                             VkDeviceSize size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags properties);

void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                VkBuffer src, VkBuffer dst, VkDeviceSize size);

void destroyBuffer(VkDevice device, AllocatedBuffer& buf);

AllocatedImage createImage(VkPhysicalDevice physicalDevice, VkDevice device,
                           uint32_t width, uint32_t height, uint32_t mipLevels,
                           VkFormat format, VkImageTiling tiling,
                           VkImageUsageFlags usage, VkMemoryPropertyFlags properties);

void destroyImage(VkDevice device, AllocatedImage& img);

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspectFlags, uint32_t mipLevels);

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);
void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue queue, VkCommandBuffer cmd);

void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                           VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                           uint32_t mipLevels);
