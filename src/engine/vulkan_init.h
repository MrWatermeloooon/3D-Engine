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
VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const QueueFamilyIndices& indices,
                             const std::vector<const char*>& extraExtensions = {});

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

// Variable-rate shading: probed at device creation. If the GPU exposes
// VK_KHR_fragment_shading_rate AND the pipelineFragmentShadingRate feature,
// we enable both, light up VRS_SUPPORTED, and load the setter. Otherwise
// all VRS code paths gracefully no-op.
extern bool                                  VRS_SUPPORTED;
extern PFN_vkCmdSetFragmentShadingRateKHR    VRS_SetRate;

// Ray tracing: probed at device creation. If the GPU exposes
// VK_KHR_acceleration_structure + VK_KHR_ray_query + VK_KHR_deferred_host_operations
// AND the relevant features, RT_SUPPORTED is set and the AS function pointers
// are loaded. Otherwise all RT code paths gracefully no-op (engine falls back
// to the existing CSM shadow path).
extern bool RT_SUPPORTED;
extern PFN_vkCreateAccelerationStructureKHR                 RT_CreateAS;
extern PFN_vkDestroyAccelerationStructureKHR                RT_DestroyAS;
extern PFN_vkGetAccelerationStructureBuildSizesKHR          RT_GetASBuildSizes;
extern PFN_vkGetAccelerationStructureDeviceAddressKHR       RT_GetASDeviceAddress;
extern PFN_vkCmdBuildAccelerationStructuresKHR              RT_CmdBuildAS;
extern PFN_vkCmdWriteAccelerationStructuresPropertiesKHR    RT_CmdWriteASProps;
extern PFN_vkGetBufferDeviceAddressKHR                      RT_GetBufferDeviceAddress;
