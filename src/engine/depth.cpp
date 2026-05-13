#include "depth.h"
#include "buffer.h"
#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <stdexcept>
#include <vector>

static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                                    const std::vector<VkFormat>& candidates,
                                    VkImageTiling tiling,
                                    VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == VK_IMAGE_TILING_OPTIMAL &&
            (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("Failed to find supported depth format");
}

VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
    return findSupportedFormat(physicalDevice,
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

DepthData createDepthResources(VkPhysicalDevice physicalDevice, VkDevice device,
                               VkExtent2D extent) {
    DepthData depth;
    depth.format = findDepthFormat(physicalDevice);

    auto img = createImage(physicalDevice, device,
                           extent.width, extent.height, 1,
                           depth.format, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    depth.image      = img.image;
    depth.allocation = img.allocation;
    depth.imageView = createImageView(device, depth.image, depth.format,
                                      VK_IMAGE_ASPECT_DEPTH_BIT, 1);
    return depth;
}

void destroyDepthResources(VkDevice device, DepthData& depth) {
    if (depth.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, depth.imageView, nullptr);
        depth.imageView = VK_NULL_HANDLE;
    }
    if (depth.image != VK_NULL_HANDLE) {
        vmaDestroyImage(gVmaAllocator, depth.image, depth.allocation);
        depth.image      = VK_NULL_HANDLE;
        depth.allocation = VK_NULL_HANDLE;
    }
}
