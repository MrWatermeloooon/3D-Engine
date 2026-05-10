#pragma once

#include <vulkan/vulkan.h>

struct DepthData {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);
DepthData createDepthResources(VkPhysicalDevice physicalDevice, VkDevice device, VkExtent2D extent);
void destroyDepthResources(VkDevice device, DepthData& depth);
