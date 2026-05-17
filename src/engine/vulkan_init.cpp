#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <iostream>
#include <set>
#include <string>
#include <cstring>

// Global VMA allocator. Created in createVmaAllocator() after device creation,
// destroyed in destroyVmaAllocator() before the device. All AllocatedBuffer /
// AllocatedImage instances live inside this allocator's pages.
VmaAllocator gVmaAllocator = VK_NULL_HANDLE;

void createVmaAllocator(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device) {
    VmaAllocatorCreateInfo info{};
    info.instance         = instance;
    info.physicalDevice   = physicalDevice;
    info.device           = device;
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    // Enable BUFFER_DEVICE_ADDRESS allocation flag pass-through. Without this
    // VMA refuses to allocate buffers requested with
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT (the RT-related buffers).
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    VK_CHECK(vmaCreateAllocator(&info, &gVmaAllocator));
}

void destroyVmaAllocator() {
    if (gVmaAllocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(gVmaAllocator);
        gVmaAllocator = VK_NULL_HANDLE;
    }
}

// ── Debug messenger callback ────────────────────────────────────────────────

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/)
{
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << pCallbackData->pMessage << "\n";
    }
    return VK_FALSE;
}

static VkDebugUtilsMessengerCreateInfoEXT makeDebugCreateInfo() {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    return info;
}

// ── Validation layers ───────────────────────────────────────────────────────

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> available(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, available.data());

    for (const char* name : VALIDATION_LAYERS) {
        bool found = false;
        for (const auto& layer : available) {
            if (std::strcmp(name, layer.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

// ── Instance ────────────────────────────────────────────────────────────────

VkInstance createInstance(const std::vector<const char*>& requiredExtensions) {
    if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Vulkan Engine";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName        = "VulkanEngine";
    appInfo.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    // Extensions: GLFW-required + debug utils
    std::vector<const char*> extensions(requiredExtensions);
    if (ENABLE_VALIDATION_LAYERS) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Validation layers + debug messenger for vkCreateInstance/vkDestroyInstance
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    if (ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        debugInfo        = makeDebugCreateInfo();
        createInfo.pNext = &debugInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext             = nullptr;
    }

    VkInstance instance;
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
    std::cout << "[VulkanEngine] Vulkan instance created\n";
    return instance;
}

// ── Debug messenger ─────────────────────────────────────────────────────────

VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance) {
    if (!ENABLE_VALIDATION_LAYERS) return VK_NULL_HANDLE;

    auto info = makeDebugCreateInfo();
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

    VkDebugUtilsMessengerEXT messenger;
    if (!func || func(instance, &info, nullptr, &messenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to set up debug messenger");
    }
    return messenger;
}

void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger) {
    if (!ENABLE_VALIDATION_LAYERS || messenger == VK_NULL_HANDLE) return;
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func) func(instance, messenger, nullptr);
}

// ── Queue family discovery ──────────────────────────────────────────────────

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport)
            indices.presentFamily = i;

        if (indices.isComplete()) break;
    }
    return indices;
}

// ── Physical device selection ───────────────────────────────────────────────

static bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());

    std::set<std::string> required(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
    for (const auto& ext : available) required.erase(ext.extensionName);
    return required.empty();
}

static int rateDevice(VkPhysicalDevice device, VkSurfaceKHR surface) {
    auto indices = findQueueFamilies(device, surface);
    if (!indices.isComplete()) return 0;
    if (!checkDeviceExtensionSupport(device)) return 0;

    // Must have surface formats + present modes
    uint32_t fmtCount, modeCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &fmtCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, nullptr);
    if (fmtCount == 0 || modeCount == 0) return 0;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    int score = 1;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
    return score;
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("No GPUs with Vulkan support");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int bestScore = 0;
    for (auto& d : devices) {
        int s = rateDevice(d, surface);
        if (s > bestScore) { bestScore = s; best = d; }
    }
    if (best == VK_NULL_HANDLE) throw std::runtime_error("No suitable GPU found");

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(best, &props);
    std::cout << "[VulkanEngine] GPU: " << props.deviceName << "\n";
    return best;
}

// ── Logical device ──────────────────────────────────────────────────────────

DeviceCapabilities gDeviceCaps{};

void resetDeviceCapabilities() { gDeviceCaps = DeviceCapabilities{}; }

// Small helper to chain feature structs without manually rewiring pointers
// every time a new optional feature is added. Sets head.pNext = next and
// returns next so calls can be daisy-chained:
//   chain(chain(&features2, &vk12), &rqFeat);
template <typename Head, typename Next>
static Next* chainPNext(Head* head, Next* next) {
    next->pNext = head->pNext;
    head->pNext = next;
    return next;
}

// Returns true if the device exposes VK_KHR_fragment_shading_rate AND the
// pipelineFragmentShadingRate feature. Attachment-based VRS (texel-rate image)
// would also need attachmentFragmentShadingRate + a render-pass-2 migration;
// we deliberately stop at pipeline/dynamic rate to keep render passes simple.
// Probe RT support. Three extensions must be present AND both features
// (accelerationStructure + rayQuery) must report VK_TRUE.
//
// We intentionally use ray *queries* (inline tracing from existing shaders)
// rather than ray-tracing *pipelines* (rgen/rmiss/rchit + SBT). Ray queries
// require strictly less machinery — no separate pipeline, no shader binding
// table — and integrate directly into mesh.frag and any future GI compute
// pass.
struct RtProbe {
    bool extPresent   = false;
    bool asFeature    = false;
    bool rqFeature    = false;
    bool bdaFeature   = false; // bufferDeviceAddress, comes via VK12 features
};

static RtProbe probeRtSupport(VkPhysicalDevice physicalDevice) {
    RtProbe out{};

    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> avail(count);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, avail.data());

    bool hasAS = false, hasRQ = false, hasDHO = false;
    for (const auto& e : avail) {
        std::string n = e.extensionName;
        if (n == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)    hasAS  = true;
        if (n == VK_KHR_RAY_QUERY_EXTENSION_NAME)                 hasRQ  = true;
        if (n == VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)  hasDHO = true;
    }
    out.extPresent = hasAS && hasRQ && hasDHO;
    if (!out.extPresent) {
        std::cout << "[VulkanEngine] RT probe: missing extension(s) —"
                  << (hasAS  ? "" : " VK_KHR_acceleration_structure")
                  << (hasRQ  ? "" : " VK_KHR_ray_query")
                  << (hasDHO ? "" : " VK_KHR_deferred_host_operations")
                  << "\n";
        return out;
    }

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{};
    asFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{};
    rqFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    asFeat.pNext = &rqFeat;

    VkPhysicalDeviceVulkan12Features vk12{};
    vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    rqFeat.pNext = &vk12;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &asFeat;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &f2);

    out.asFeature  = asFeat.accelerationStructure == VK_TRUE;
    out.rqFeature  = rqFeat.rayQuery == VK_TRUE;
    out.bdaFeature = vk12.bufferDeviceAddress == VK_TRUE;

    if (!(out.asFeature && out.rqFeature && out.bdaFeature)) {
        std::cout << "[VulkanEngine] RT probe: extensions present but feature(s) off —"
                  << " accelerationStructure="  << out.asFeature
                  << " rayQuery="               << out.rqFeature
                  << " bufferDeviceAddress="    << out.bdaFeature
                  << "\n";
    }
    return out;
}

static bool deviceSupportsVRS(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> avail(count);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, avail.data());
    bool extPresent = false;
    for (const auto& e : avail) {
        if (std::string(e.extensionName) == VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME) {
            extPresent = true; break;
        }
    }
    if (!extPresent) return false;

    VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrsFeat{};
    vrsFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &vrsFeat;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &f2);
    return vrsFeat.pipelineFragmentShadingRate == VK_TRUE;
}

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const QueueFamilyIndices& indices,
                             const std::vector<const char*>& extraExtensions) {
    // Wipe any state left over from a previous engine instance — without this,
    // a second engine's probe results overwrite the first's on top.
    resetDeviceCapabilities();

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::set<uint32_t> uniqueFamilies = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    float priority = 1.0f;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    // Vulkan 1.2 descriptor-indexing features for bindless rendering
    VkPhysicalDeviceVulkan12Features vk12{};
    vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12.descriptorIndexing                              = VK_TRUE;
    vk12.descriptorBindingPartiallyBound                 = VK_TRUE;
    vk12.descriptorBindingSampledImageUpdateAfterBind    = VK_TRUE;
    vk12.descriptorBindingVariableDescriptorCount        = VK_TRUE;
    vk12.runtimeDescriptorArray                          = VK_TRUE;
    vk12.shaderSampledImageArrayNonUniformIndexing       = VK_TRUE;
    // mesh.frag uses GL_EXT_scalar_block_layout + uint64_t (for buffer-
    // reference pointer reconstruction) — both must be opted into at device
    // creation or shader module creation fires validation errors.
    vk12.scalarBlockLayout                               = VK_TRUE;

    // Optional VRS feature struct, chained only if the device supports it.
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrsFeat{};
    vrsFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
    vrsFeat.pipelineFragmentShadingRate = VK_TRUE;

    const bool vrsSupported = deviceSupportsVRS(physicalDevice);

    // Optional RT feature structs, chained only if the device supports them.
    // Acceleration structures need bufferDeviceAddress on VK_KHR_buffer_device_address
    // (a core 1.2 feature) — that's part of the vk12 struct.
    RtProbe rtProbe = probeRtSupport(physicalDevice);
    const bool rtSupported = rtProbe.extPresent && rtProbe.asFeature
                          && rtProbe.rqFeature && rtProbe.bdaFeature;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{};
    asFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeat.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{};
    rqFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqFeat.rayQuery = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vk12;
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.features.shaderInt64       = VK_TRUE; // mesh.frag uint64_t

    // Chain optional features. The chain looks like:
    //   features2 → vk12 → [vrsFeat] → [asFeat → rqFeat]
    // chainPNext() preserves whatever the head already pointed at, so adding
    // a new feature in the middle later just needs another line — no manual
    // tail-pointer bookkeeping.
    if (vrsSupported) chainPNext(&vk12, &vrsFeat);
    if (rtSupported) {
        // Enable bufferDeviceAddress on the existing vk12 struct (core 1.2).
        vk12.bufferDeviceAddress = VK_TRUE;
        chainPNext(&vk12, &asFeat);
        chainPNext(&vk12, &rqFeat);
    }

    // Build the per-device extension list, copying base + optionally VRS + RT.
    std::vector<const char*> deviceExts(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
    if (vrsSupported) deviceExts.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
    if (rtSupported) {
        deviceExts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        deviceExts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }
    // Caller-supplied extensions for optional integrations. Dedupe
    // so we don't double-list anything already enabled above — validation
    // would otherwise complain.
    for (const char* e : extraExtensions) {
        bool present = false;
        for (const char* existing : deviceExts) {
            if (std::strcmp(existing, e) == 0) { present = true; break; }
        }
        if (!present) deviceExts.push_back(e);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext                   = &features2;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos       = queueInfos.data();
    // When using pNext->VkPhysicalDeviceFeatures2, pEnabledFeatures must be null.
    createInfo.pEnabledFeatures        = nullptr;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExts.size());
    createInfo.ppEnabledExtensionNames = deviceExts.data();

    // Deprecated but set for older driver compat
    if (ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    VkDevice device;
    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));

    VRS_SUPPORTED = vrsSupported;
    if (VRS_SUPPORTED) {
        VRS_SetRate = reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
            vkGetDeviceProcAddr(device, "vkCmdSetFragmentShadingRateKHR"));
        if (!VRS_SetRate) VRS_SUPPORTED = false;  // function load failed somehow
    }

    RT_SUPPORTED = rtSupported;
    if (RT_SUPPORTED) {
        auto load = [&](const char* name, auto& out) {
            out = reinterpret_cast<std::remove_reference_t<decltype(out)>>(
                vkGetDeviceProcAddr(device, name));
            if (out == nullptr) {
                std::cout << "[VulkanEngine] RT load: missing " << name << "\n";
                return false;
            }
            return true;
        };
        bool ok = true;
        ok &= load("vkCreateAccelerationStructureKHR",                RT_CreateAS);
        ok &= load("vkDestroyAccelerationStructureKHR",               RT_DestroyAS);
        ok &= load("vkGetAccelerationStructureBuildSizesKHR",         RT_GetASBuildSizes);
        ok &= load("vkGetAccelerationStructureDeviceAddressKHR",      RT_GetASDeviceAddress);
        ok &= load("vkCmdBuildAccelerationStructuresKHR",             RT_CmdBuildAS);
        ok &= load("vkCmdWriteAccelerationStructuresPropertiesKHR",   RT_CmdWriteASProps);
        // GetBufferDeviceAddress is core 1.2, but also exposed under the KHR
        // alias; prefer the core symbol and fall back to the KHR alias.
        RT_GetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
            vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress"));
        if (!RT_GetBufferDeviceAddress) {
            RT_GetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
                vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
        }
        if (!RT_GetBufferDeviceAddress) {
            std::cout << "[VulkanEngine] RT load: missing vkGetBufferDeviceAddress[KHR]\n";
            ok = false;
        }
        if (!ok) RT_SUPPORTED = false;

        if (RT_SUPPORTED) {
            VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
            asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &asProps;
            vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
            RT_SCRATCH_ALIGNMENT = asProps.minAccelerationStructureScratchOffsetAlignment
                                       ? asProps.minAccelerationStructureScratchOffsetAlignment
                                       : 1u;
        }
    }

    std::cout << "[VulkanEngine] Logical device created"
              << (VRS_SUPPORTED ? " (VRS supported)" : " (VRS unavailable)")
              << (RT_SUPPORTED  ? " (RT supported)"  : " (RT unavailable)")
              << "\n";
    return device;
}
