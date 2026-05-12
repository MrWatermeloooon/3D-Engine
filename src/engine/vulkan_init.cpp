#include "vulkan_init.h"
#include "../utils/vk_check.h"

#include <iostream>
#include <set>
#include <string>
#include <cstring>

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

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const QueueFamilyIndices& indices) {
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

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vk12;
    features2.features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext                   = &features2;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos       = queueInfos.data();
    // When using pNext->VkPhysicalDeviceFeatures2, pEnabledFeatures must be null.
    createInfo.pEnabledFeatures        = nullptr;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
    createInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

    // Deprecated but set for older driver compat
    if (ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    VkDevice device;
    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));
    std::cout << "[VulkanEngine] Logical device created\n";
    return device;
}
