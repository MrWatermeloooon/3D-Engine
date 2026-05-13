#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <string>

struct TextureData {
    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView   imageView  = VK_NULL_HANDLE;
    VkSampler     sampler    = VK_NULL_HANDLE;
    uint32_t      mipLevels  = 1;
};

TextureData createTextureFromFile(VkPhysicalDevice physicalDevice, VkDevice device,
                                  VkCommandPool commandPool, VkQueue queue,
                                  const std::string& filepath);

TextureData createDefaultTexture(VkPhysicalDevice physicalDevice, VkDevice device,
                                 VkCommandPool commandPool, VkQueue queue);

void destroyTexture(VkDevice device, TextureData& tex);
