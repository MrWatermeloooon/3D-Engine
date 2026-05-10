#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <optional>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    QueueFamilyIndices queueFamilies;
};

// Initialization
VkInstance createInstance(const std::vector<const char*>& requiredExtensions);
VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance);
VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const QueueFamilyIndices& indices);

// Cleanup
void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger);

// Helpers
bool checkValidationLayerSupport();

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif

inline const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

inline const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
