#include "swapchain.h"
#include "../utils/vk_check.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>

// ── Support query ───────────────────────────────────────────────────────────

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &fmtCount, nullptr);
    if (fmtCount) {
        details.formats.resize(fmtCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &fmtCount, details.formats.data());
    }

    uint32_t modeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, nullptr);
    if (modeCount) {
        details.presentModes.resize(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, details.presentModes.data());
    }
    return details;
}

// ── Format / present mode / extent selection ────────────────────────────────

static VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats[0];
}

static VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D windowExtent) {
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return caps.currentExtent;
    }
    windowExtent.width  = std::clamp(windowExtent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    windowExtent.height = std::clamp(windowExtent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return windowExtent;
}

// ── Swapchain ───────────────────────────────────────────────────────────────

void createSwapchain(SwapchainData& data, VkPhysicalDevice physicalDevice,
                     VkDevice device, VkSurfaceKHR surface,
                     uint32_t graphicsFamily, uint32_t presentFamily,
                     VkExtent2D windowExtent)
{
    auto support = querySwapchainSupport(physicalDevice, surface);
    auto fmt     = chooseSurfaceFormat(support.formats);
    auto mode    = choosePresentMode(support.presentModes);
    auto extent  = chooseExtent(support.capabilities, windowExtent);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = fmt.format;
    ci.imageColorSpace  = fmt.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t families[] = { graphicsFamily, presentFamily };
    if (graphicsFamily != presentFamily) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = families;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform   = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = mode;
    ci.clipped        = VK_TRUE;
    ci.oldSwapchain   = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device, &ci, nullptr, &data.swapchain));

    vkGetSwapchainImagesKHR(device, data.swapchain, &imageCount, nullptr);
    data.images.resize(imageCount);
    vkGetSwapchainImagesKHR(device, data.swapchain, &imageCount, data.images.data());

    data.imageFormat = fmt.format;
    data.extent      = extent;

    std::cout << "[VulkanEngine] Swapchain created (" << extent.width << "x" << extent.height
              << ", " << imageCount << " images)\n";
}

// ── Image views ─────────────────────────────────────────────────────────────

void createImageViews(SwapchainData& data, VkDevice device) {
    data.imageViews.resize(data.images.size());

    for (size_t i = 0; i < data.images.size(); i++) {
        VkImageViewCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image    = data.images[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format   = data.imageFormat;
        ci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.baseMipLevel   = 0;
        ci.subresourceRange.levelCount     = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount     = 1;

        VK_CHECK(vkCreateImageView(device, &ci, nullptr, &data.imageViews[i]));
    }
}

// ── Render pass ─────────────────────────────────────────────────────────────

void createRenderPass(SwapchainData& data, VkDevice device) {
    // Color-only render pass — used for composite + ImGui (post-fx writes the final image).
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format         = data.imageFormat;
    colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0;
    dependency.srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                             | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments    = &colorAttachment;
    rpInfo.subpassCount    = 1;
    rpInfo.pSubpasses      = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies   = &dependency;

    VK_CHECK(vkCreateRenderPass(device, &rpInfo, nullptr, &data.renderPass));
}

// ── Framebuffers ────────────────────────────────────────────────────────────

void createFramebuffers(SwapchainData& data, VkDevice device) {
    data.framebuffers.resize(data.imageViews.size());

    for (size_t i = 0; i < data.imageViews.size(); i++) {
        VkFramebufferCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass      = data.renderPass;
        ci.attachmentCount = 1;
        ci.pAttachments    = &data.imageViews[i];
        ci.width           = data.extent.width;
        ci.height          = data.extent.height;
        ci.layers          = 1;

        VK_CHECK(vkCreateFramebuffer(device, &ci, nullptr, &data.framebuffers[i]));
    }
}

// ── Sync objects ────────────────────────────────────────────────────────────

void createSyncObjects(SwapchainData& data, VkDevice device) {
    data.imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    data.inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &data.imageAvailableSemaphores[i]));
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &data.inFlightFences[i]));
    }
}

void createPresentSemaphores(SwapchainData& data, VkDevice device) {
    data.renderFinishedSemaphores.resize(data.images.size());

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (size_t i = 0; i < data.images.size(); i++) {
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &data.renderFinishedSemaphores[i]));
    }
}

void destroyPresentSemaphores(SwapchainData& data, VkDevice device) {
    for (auto s : data.renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
    data.renderFinishedSemaphores.clear();
}

// ── Cleanup (partial — for swapchain recreation) ────────────────────────────

void cleanupSwapchain(SwapchainData& data, VkDevice device) {
    destroyPresentSemaphores(data, device);
    for (auto fb : data.framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    for (auto iv : data.imageViews)   vkDestroyImageView(device, iv, nullptr);
    vkDestroySwapchainKHR(device, data.swapchain, nullptr);

    data.framebuffers.clear();
    data.imageViews.clear();
    data.images.clear();
}
