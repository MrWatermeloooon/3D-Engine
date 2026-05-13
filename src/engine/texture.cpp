#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "texture.h"
#include "buffer.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>

static void generateMipmaps(VkPhysicalDevice physicalDevice, VkDevice device,
                            VkCommandPool commandPool, VkQueue queue,
                            VkImage image, VkFormat format,
                            int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
    VkFormatProperties formatProps;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);

    if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("Texture image format does not support linear blitting");
    }

    auto cmd = beginSingleTimeCommands(device, commandPool);

    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image                           = image;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.subresourceRange.levelCount     = 1;

    int32_t mipWidth  = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1,
                               mipHeight > 1 ? mipHeight / 2 : 1, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };

        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        if (mipWidth > 1)  mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(device, commandPool, queue, cmd);
}

static TextureData createTextureFromPixels(VkPhysicalDevice physicalDevice, VkDevice device,
                                           VkCommandPool commandPool, VkQueue queue,
                                           const unsigned char* pixels,
                                           int width, int height) {
    TextureData tex;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;
    tex.mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height)))) + 1;

    auto staging = createBuffer(physicalDevice, device, imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // VMA persistently maps host-visible allocations; read the pointer from
    // the AllocatedBuffer's `mapped` field.
    memcpy(staging.mapped, pixels, static_cast<size_t>(imageSize));

    auto img = createImage(physicalDevice, device,
        static_cast<uint32_t>(width), static_cast<uint32_t>(height), tex.mipLevels,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tex.image      = img.image;
    tex.allocation = img.allocation;

    transitionImageLayout(device, commandPool, queue, tex.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex.mipLevels);

    {
        auto cmd = beginSingleTimeCommands(device, commandPool);

        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { static_cast<uint32_t>(width),
                               static_cast<uint32_t>(height), 1 };

        vkCmdCopyBufferToImage(cmd, staging.buffer, tex.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(device, commandPool, queue, cmd);
    }

    destroyBuffer(device, staging);

    generateMipmaps(physicalDevice, device, commandPool, queue,
        tex.image, VK_FORMAT_R8G8B8A8_SRGB, width, height, tex.mipLevels);

    tex.imageView = createImageView(device, tex.image, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_ASPECT_COLOR_BIT, tex.mipLevels);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter        = VK_FILTER_LINEAR;
    samplerInfo.minFilter        = VK_FILTER_LINEAR;
    samplerInfo.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy    = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod           = 0.0f;
    samplerInfo.maxLod           = static_cast<float>(tex.mipLevels);

    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &tex.sampler));

    return tex;
}

TextureData createTextureFromFile(VkPhysicalDevice physicalDevice, VkDevice device,
                                  VkCommandPool commandPool, VkQueue queue,
                                  const std::string& filepath) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load texture: " + filepath);
    }

    auto tex = createTextureFromPixels(physicalDevice, device, commandPool, queue,
                                       pixels, width, height);
    stbi_image_free(pixels);
    return tex;
}

TextureData createDefaultTexture(VkPhysicalDevice physicalDevice, VkDevice device,
                                 VkCommandPool commandPool, VkQueue queue) {
    // 1x1 plain white. Multiplying by 1.0 means the material colorTint shows
    // through cleanly — what the user picks in the slider is what they see.
    constexpr int size = 1;
    unsigned char pixels[size * size * 4] = { 255, 255, 255, 255 };
    return createTextureFromPixels(physicalDevice, device, commandPool, queue,
                                   pixels, size, size);
}

void destroyTexture(VkDevice device, TextureData& tex) {
    if (tex.sampler   != VK_NULL_HANDLE) vkDestroySampler(device, tex.sampler, nullptr);
    if (tex.imageView != VK_NULL_HANDLE) vkDestroyImageView(device, tex.imageView, nullptr);
    if (tex.image     != VK_NULL_HANDLE) vmaDestroyImage(gVmaAllocator, tex.image, tex.allocation);
    tex = {};
}
