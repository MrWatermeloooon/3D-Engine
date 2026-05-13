#pragma once

#include <vulkan/vulkan.h>
#include <vector>

#include "config.h"

using engine_config::MAX_FRAMES_IN_FLIGHT;

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct SwapchainData {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;
    std::vector<VkFramebuffer> framebuffers;
    VkExtent2D extent{};
    VkFormat imageFormat{};
    VkRenderPass renderPass = VK_NULL_HANDLE;

    // Sync objects
    std::vector<VkSemaphore> imageAvailableSemaphores; // per frame in flight
    std::vector<VkSemaphore> renderFinishedSemaphores; // per swapchain image
    std::vector<VkFence> inFlightFences;               // per frame in flight
};

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

void createSwapchain(SwapchainData& data, VkPhysicalDevice physicalDevice,
                     VkDevice device, VkSurfaceKHR surface,
                     uint32_t graphicsFamily, uint32_t presentFamily,
                     VkExtent2D windowExtent);

void createImageViews(SwapchainData& data, VkDevice device);
void createRenderPass(SwapchainData& data, VkDevice device);  // color-only (composite + UI)
void createFramebuffers(SwapchainData& data, VkDevice device);
void createSyncObjects(SwapchainData& data, VkDevice device);
void createPresentSemaphores(SwapchainData& data, VkDevice device);
void destroyPresentSemaphores(SwapchainData& data, VkDevice device);

// Destroys present semaphores, framebuffers, image views, and swapchain
void cleanupSwapchain(SwapchainData& data, VkDevice device);
