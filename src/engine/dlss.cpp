#include "dlss.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_defs.h>

#include <iostream>
#include <filesystem>
#include <string>

namespace {
    bool                  s_initDone        = false;
    bool                  s_available       = false;
    NVSDK_NGX_Parameter*  s_params          = nullptr;

    // Stable IDs. Any nonzero project ID works for development. NVIDIA only
    // requires a real project ID for release-signed DLLs (the rel/ DLL); the
    // dev DLL we ship accepts any.
    // NGX requires the project ID to be a UUID (8-4-4-4-12 hex digits).
    // Any valid UUID works for development; this one is just a random GUID.
    constexpr const char* kProjectId       = "5b7e8f4a-2c9d-4e3f-a1b6-c8d7e9f0a1b2";
    constexpr const char* kEngineVersion   = "0.1.0";

    // Lifetime-stable storage for extension name strings.
    std::vector<std::string> s_instanceExtStorage;
    std::vector<std::string> s_deviceExtStorage;
    std::vector<const char*> s_instanceExtPtrs;
    std::vector<const char*> s_deviceExtPtrs;

    // NGX is extremely chatty at verbose level. We let it write its own log
    // file (dlss_logs/nvsdk_ngx.log) for diagnostics but silence the console
    // mirror — no callback. Crank the log level back up to VERBOSE if you
    // need to debug an init failure.
    void fillCommonInfo(NVSDK_NGX_FeatureCommonInfo& info,
                       const wchar_t** paths, uint32_t numPaths,
                       NVSDK_NGX_LoggingInfo& /*outLogging*/)
    {
        info.PathListInfo.Path        = paths;
        info.PathListInfo.Length      = numPaths;
        // Leaving info.LoggingInfo at default ({nullptr, OFF, false}) means
        // NGX uses its built-in file logger only — no console spam.
    }

    std::wstring exeDirW() {
        return std::filesystem::current_path().wstring();
    }

    NVSDK_NGX_FeatureDiscoveryInfo makeDiscoveryInfo(
        const NVSDK_NGX_FeatureCommonInfo* commonInfo,
        const wchar_t* appDataPath)
    {
        NVSDK_NGX_FeatureDiscoveryInfo info{};
        info.SDKVersion          = NVSDK_NGX_Version_API;
        info.FeatureID           = NVSDK_NGX_Feature_SuperSampling;
        info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
        info.Identifier.v.ProjectDesc.ProjectId     = kProjectId;
        info.Identifier.v.ProjectDesc.EngineType    = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
        info.Identifier.v.ProjectDesc.EngineVersion = kEngineVersion;
        info.ApplicationDataPath = appDataPath;
        info.FeatureInfo         = commonInfo;
        return info;
    }
}

std::vector<const char*> dlssRequiredInstanceExtensions() {
    s_instanceExtStorage.clear();
    s_instanceExtPtrs.clear();

    static std::wstring sExeDir = exeDirW();
    static std::wstring sAppData = (std::filesystem::current_path() / L"dlss_logs").wstring();
    const wchar_t* paths[] = { sExeDir.c_str() };
    NVSDK_NGX_FeatureCommonInfo commonInfo{};
    NVSDK_NGX_LoggingInfo       logging{};
    fillCommonInfo(commonInfo, paths, 1, logging);

    NVSDK_NGX_FeatureDiscoveryInfo info = makeDiscoveryInfo(&commonInfo, sAppData.c_str());

    uint32_t count = 0;
    VkExtensionProperties* props = nullptr;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
        &info, &count, &props);
    if (NVSDK_NGX_FAILED(r)) {
        std::cerr << "[DLSS] GetFeatureInstanceExtensionRequirements failed: 0x"
                  << std::hex << (uint32_t)r << std::dec << "\n";
        return {};
    }
    for (uint32_t i = 0; i < count; ++i) {
        s_instanceExtStorage.emplace_back(props[i].extensionName);
    }
    s_instanceExtPtrs.reserve(s_instanceExtStorage.size());
    for (auto& s : s_instanceExtStorage) s_instanceExtPtrs.push_back(s.c_str());
    return s_instanceExtPtrs;
}

std::vector<const char*> dlssRequiredDeviceExtensions(VkInstance instance,
                                                      VkPhysicalDevice physicalDevice) {
    s_deviceExtStorage.clear();
    s_deviceExtPtrs.clear();

    static std::wstring sExeDir = exeDirW();
    static std::wstring sAppData = (std::filesystem::current_path() / L"dlss_logs").wstring();
    const wchar_t* paths[] = { sExeDir.c_str() };
    NVSDK_NGX_FeatureCommonInfo commonInfo{};
    NVSDK_NGX_LoggingInfo       logging{};
    fillCommonInfo(commonInfo, paths, 1, logging);

    NVSDK_NGX_FeatureDiscoveryInfo info = makeDiscoveryInfo(&commonInfo, sAppData.c_str());

    uint32_t count = 0;
    VkExtensionProperties* props = nullptr;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance, physicalDevice, &info, &count, &props);
    if (NVSDK_NGX_FAILED(r)) {
        std::cerr << "[DLSS] GetFeatureDeviceExtensionRequirements failed: 0x"
                  << std::hex << (uint32_t)r << std::dec << "\n";
        return {};
    }
    for (uint32_t i = 0; i < count; ++i) {
        s_deviceExtStorage.emplace_back(props[i].extensionName);
    }
    s_deviceExtPtrs.reserve(s_deviceExtStorage.size());
    for (auto& s : s_deviceExtStorage) s_deviceExtPtrs.push_back(s.c_str());
    return s_deviceExtPtrs;
}

void dlssInit(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device) {
    if (s_initDone) return;
    s_initDone = true;

    std::filesystem::path logDir = std::filesystem::current_path() / L"dlss_logs";
    std::filesystem::create_directories(logDir);

    static std::wstring sExeDirInit = exeDirW();
    const wchar_t* dllPaths[] = { sExeDirInit.c_str() };
    NVSDK_NGX_FeatureCommonInfo featureInfo{};
    NVSDK_NGX_LoggingInfo       logging{};
    fillCommonInfo(featureInfo, dllPaths, 1, logging);

    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId,
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,
        kEngineVersion,
        logDir.c_str(),
        instance,
        physicalDevice,
        device,
        nullptr, nullptr,
        &featureInfo);
    if (NVSDK_NGX_FAILED(r)) {
        std::cerr << "[DLSS] NGX init failed: 0x" << std::hex
                  << (uint32_t)r << std::dec;
        if ((uint32_t)r == 0xBAD00005) {
            std::cerr << " (FeatureNotSupported — check driver version /"
                         " required Vulkan extensions enabled at device creation).";
        }
        std::cerr << "\n";
        return;
    }

    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&s_params);
    if (NVSDK_NGX_FAILED(r) || s_params == nullptr) {
        std::cerr << "[DLSS] GetCapabilityParameters failed: 0x" << std::hex
                  << (uint32_t)r << std::dec << "\n";
        NVSDK_NGX_VULKAN_Shutdown1(device);
        return;
    }

    int dlssAvail = 0;
    NVSDK_NGX_Result q = s_params->Get(
        NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvail);
    if (NVSDK_NGX_FAILED(q) || !dlssAvail) {
        std::cerr << "[DLSS] DLSS Super Sampling not available on this GPU\n";
        NVSDK_NGX_VULKAN_DestroyParameters(s_params);
        s_params = nullptr;
        NVSDK_NGX_VULKAN_Shutdown1(device);
        return;
    }

    s_available = true;
    std::cout << "[DLSS] NGX initialised, DLSS Super Sampling available\n";
}

bool dlssAvailable() { return s_available; }

namespace {
    // Van der Corput base-N. Used for the Halton sequence below.
    float halton(uint32_t i, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        while (i > 0) {
            f /= float(base);
            r += f * float(i % base);
            i /= base;
        }
        return r;
    }
}

glm::vec2 haltonJitter(uint32_t index) {
    // Halton(2, 3) with index >= 1. Returns offset in [-0.5, +0.5].
    return glm::vec2(halton(index, 2) - 0.5f,
                     halton(index, 3) - 0.5f);
}

void dlssShutdown(VkDevice device) {
    if (!s_initDone) return;
    if (s_params) {
        NVSDK_NGX_VULKAN_DestroyParameters(s_params);
        s_params = nullptr;
    }
    if (s_available) {
        NVSDK_NGX_VULKAN_Shutdown1(device);
    }
    s_available = false;
    s_initDone  = false;
}
