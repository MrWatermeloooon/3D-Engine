#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
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

// VMA allocator. Single instance, created after the logical device and
// destroyed before it. All buffer/image allocations route through this rather
// than vkAllocateMemory — VMA sub-allocates from large pages, dodging the
// 4096-allocation driver limit that bites when many small buffers are created
// (texture-heavy scenes, hot-reload churn).
extern VmaAllocator gVmaAllocator;
void createVmaAllocator(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
void destroyVmaAllocator();

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

// Optional device features probed at logical-device creation. Grouped in a
// struct so the values can be reset cleanly between sequential engine
// instances (unit tests, editor + game) — they used to be raw module-level
// globals which the second device's probe would silently overwrite on top of
// the first. createLogicalDevice() calls resetDeviceCapabilities() before
// probing so stale state from a previous engine never leaks through.
//
// The legacy VRS_SUPPORTED / RT_SUPPORTED / function-pointer globals are kept
// as references into this struct so existing call sites compile unchanged.
struct DeviceCapabilities {
    // Variable-rate shading
    bool                                  vrsSupported = false;
    PFN_vkCmdSetFragmentShadingRateKHR    vrsSetRate   = nullptr;

    // Ray tracing
    bool rtSupported = false;
    PFN_vkCreateAccelerationStructureKHR                 rtCreateAS              = nullptr;
    PFN_vkDestroyAccelerationStructureKHR                rtDestroyAS             = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR          rtGetASBuildSizes       = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR       rtGetASDeviceAddress    = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR              rtCmdBuildAS            = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR    rtCmdWriteASProps       = nullptr;
    PFN_vkGetBufferDeviceAddressKHR                      rtGetBufferDeviceAddress = nullptr;
    // VkPhysicalDeviceAccelerationStructurePropertiesKHR::
    // minAccelerationStructureScratchOffsetAlignment. AS build scratch device
    // addresses must be a multiple of this; VMA buffer base addresses are not
    // guaranteed to satisfy it. 1 = "no constraint" when RT is unsupported.
    uint32_t rtScratchAlignment = 1;
};

extern DeviceCapabilities gDeviceCaps;
void resetDeviceCapabilities();

// Legacy aliases — references bind to the struct fields so any existing
// `if (VRS_SUPPORTED)` / `if (RT_SUPPORTED)` use continues to work and stays
// authoritative with the new owner.
inline bool&                                  VRS_SUPPORTED         = gDeviceCaps.vrsSupported;
inline PFN_vkCmdSetFragmentShadingRateKHR&    VRS_SetRate           = gDeviceCaps.vrsSetRate;
inline bool&                                  RT_SUPPORTED          = gDeviceCaps.rtSupported;
inline PFN_vkCreateAccelerationStructureKHR&                 RT_CreateAS              = gDeviceCaps.rtCreateAS;
inline PFN_vkDestroyAccelerationStructureKHR&                RT_DestroyAS             = gDeviceCaps.rtDestroyAS;
inline PFN_vkGetAccelerationStructureBuildSizesKHR&          RT_GetASBuildSizes       = gDeviceCaps.rtGetASBuildSizes;
inline PFN_vkGetAccelerationStructureDeviceAddressKHR&       RT_GetASDeviceAddress    = gDeviceCaps.rtGetASDeviceAddress;
inline PFN_vkCmdBuildAccelerationStructuresKHR&              RT_CmdBuildAS            = gDeviceCaps.rtCmdBuildAS;
inline PFN_vkCmdWriteAccelerationStructuresPropertiesKHR&    RT_CmdWriteASProps       = gDeviceCaps.rtCmdWriteASProps;
inline PFN_vkGetBufferDeviceAddressKHR&                      RT_GetBufferDeviceAddress = gDeviceCaps.rtGetBufferDeviceAddress;
inline uint32_t&                                            RT_SCRATCH_ALIGNMENT      = gDeviceCaps.rtScratchAlignment;
