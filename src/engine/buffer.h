#pragma once

#include <vulkan/vulkan.h>

struct AllocatedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct AllocatedImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties);

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

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                            VkImageAspectFlags aspectFlags, uint32_t mipLevels);

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);
void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue queue, VkCommandBuffer cmd);

void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue,
                           VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                           uint32_t mipLevels);
