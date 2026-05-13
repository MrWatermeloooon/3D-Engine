#include "dlss.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers.h>      // NGX_DLSS_GET_OPTIMAL_SETTINGS (graphics-agnostic)
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_defs.h>

#include <iostream>
#include <filesystem>
#include <string>

namespace {
    bool                  s_initDone        = false;
    bool                  s_available       = false;
    NVSDK_NGX_Parameter*  s_params          = nullptr;

    // Active DLSS feature. Created on demand by dlssCreateFeature; destroyed
    // by dlssReleaseFeature or dlssShutdown. nullptr when no feature is alive.
    NVSDK_NGX_Handle*     s_feature         = nullptr;
    VkExtent2D            s_featureIn       = {0, 0};
    VkExtent2D            s_featureOut      = {0, 0};
    VkDevice              s_device          = VK_NULL_HANDLE;  // captured at init for shutdown helpers

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
    s_device    = device;
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

// ── Feature lifecycle + evaluate ────────────────────────────────────────────

DlssRenderSize dlssGetOptimalRenderSize(uint32_t targetWidth,
                                        uint32_t targetHeight,
                                        DlssQuality quality) {
    DlssRenderSize out{targetWidth, targetHeight, false};
    if (!s_available || !s_params) return out;

    NVSDK_NGX_PerfQuality_Value qv =
        static_cast<NVSDK_NGX_PerfQuality_Value>(static_cast<int>(quality));

    unsigned int optW = 0, optH = 0;
    unsigned int maxW = 0, maxH = 0;
    unsigned int minW = 0, minH = 0;
    float        sharpness = 0.0f;

    NVSDK_NGX_Result r = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        s_params, targetWidth, targetHeight, qv,
        &optW, &optH, &maxW, &maxH, &minW, &minH, &sharpness);

    if (NVSDK_NGX_FAILED(r) || optW == 0 || optH == 0) {
        std::cerr << "[DLSS] GetOptimalSettings failed: 0x"
                  << std::hex << (uint32_t)r << std::dec << "\n";
        return out;
    }

    out.inWidth  = optW;
    out.inHeight = optH;
    out.ok       = true;
    return out;
}

bool dlssCreateFeature(VkCommandBuffer cmd,
                       uint32_t inWidth,  uint32_t inHeight,
                       uint32_t outWidth, uint32_t outHeight,
                       DlssQuality quality) {
    if (!s_available || !s_params) return false;

    // Tear down any prior feature first — quality-change path also lands here.
    if (s_feature) {
        // We're recreating mid-session. Caller must have flushed the GPU
        // before getting here. Use s_device captured at init time.
        if (s_device) vkDeviceWaitIdle(s_device);
        NVSDK_NGX_VULKAN_ReleaseFeature(s_feature);
        s_feature = nullptr;
    }

    NVSDK_NGX_DLSS_Create_Params params{};
    params.Feature.InWidth        = inWidth;
    params.Feature.InHeight       = inHeight;
    params.Feature.InTargetWidth  = outWidth;
    params.Feature.InTargetHeight = outHeight;
    params.Feature.InPerfQualityValue =
        static_cast<NVSDK_NGX_PerfQuality_Value>(static_cast<int>(quality));
    // Input is LDR sRGB-ish from the composite pass — DO NOT flag IsHDR.
    // MVLowRes: motion vectors are at input resolution (not target).
    // DepthInverted: our depth is Vulkan-style 0..1 with near=0 (we use
    // LESS compare, near plane → small depth), so NOT inverted.
    // AutoExposure: we have no exposure texture and our scene is roughly
    // mid-grey, let DLSS estimate from the color stream.
    int createFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
                    | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    params.InFeatureCreateFlags    = createFlags;
    params.InEnableOutputSubrects  = false;

    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(
        cmd, /*creationNodeMask*/1, /*visibilityNodeMask*/1,
        &s_feature, s_params, &params);

    if (NVSDK_NGX_FAILED(r) || s_feature == nullptr) {
        std::cerr << "[DLSS] CreateFeature_DLSS failed: 0x"
                  << std::hex << (uint32_t)r << std::dec << "\n";
        s_feature   = nullptr;
        s_featureIn = s_featureOut = {0, 0};
        return false;
    }

    s_featureIn  = { inWidth,  inHeight  };
    s_featureOut = { outWidth, outHeight };
    std::cout << "[DLSS] Feature created: " << inWidth << "x" << inHeight
              << " -> " << outWidth << "x" << outHeight << "\n";
    return true;
}

void dlssReleaseFeature(VkDevice device) {
    if (!s_feature) return;
    if (device) vkDeviceWaitIdle(device);
    NVSDK_NGX_VULKAN_ReleaseFeature(s_feature);
    s_feature    = nullptr;
    s_featureIn  = {0, 0};
    s_featureOut = {0, 0};
}

bool       dlssFeatureReady()        { return s_feature != nullptr; }
VkExtent2D dlssFeatureInputExtent()  { return s_featureIn; }
VkExtent2D dlssFeatureOutputExtent() { return s_featureOut; }

void dlssEvaluate(VkCommandBuffer cmd,
                  VkImage colorImage,  VkImageView colorView,  VkFormat colorFormat,
                  VkImage depthImage,  VkImageView depthView,  VkFormat depthFormat,
                  VkImage motionImage, VkImageView motionView, VkFormat motionFormat,
                  VkImage outputImage, VkImageView outputView, VkFormat outputFormat,
                  uint32_t inWidth, uint32_t inHeight,
                  uint32_t outWidth, uint32_t outHeight,
                  float jitterX, float jitterY,
                  float motionScaleX, float motionScaleY,
                  bool reset)
{
    if (!s_feature || !s_params) return;
    (void)outWidth; (void)outHeight; // currently informational; NGX reads from create-time config

    auto makeView = [](VkImageView view, VkImage image, VkFormat fmt,
                       uint32_t w, uint32_t h, VkImageAspectFlags aspect, bool rw)
    {
        VkImageSubresourceRange sr{};
        sr.aspectMask     = aspect;
        sr.baseMipLevel   = 0;
        sr.levelCount     = 1;
        sr.baseArrayLayer = 0;
        sr.layerCount     = 1;
        return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, sr, fmt, w, h, rw);
    };

    NVSDK_NGX_Resource_VK colorR  = makeView(colorView,  colorImage,  colorFormat,
                                              inWidth, inHeight, VK_IMAGE_ASPECT_COLOR_BIT, false);
    NVSDK_NGX_Resource_VK depthR  = makeView(depthView,  depthImage,  depthFormat,
                                              inWidth, inHeight, VK_IMAGE_ASPECT_DEPTH_BIT, false);
    NVSDK_NGX_Resource_VK motionR = makeView(motionView, motionImage, motionFormat,
                                              inWidth, inHeight, VK_IMAGE_ASPECT_COLOR_BIT, false);
    NVSDK_NGX_Resource_VK outputR = makeView(outputView, outputImage, outputFormat,
                                              s_featureOut.width, s_featureOut.height,
                                              VK_IMAGE_ASPECT_COLOR_BIT, true);

    NVSDK_NGX_VK_DLSS_Eval_Params eval{};
    eval.Feature.pInColor   = &colorR;
    eval.Feature.pInOutput  = &outputR;
    eval.Feature.InSharpness = 0.0f;
    eval.pInDepth           = &depthR;
    eval.pInMotionVectors   = &motionR;
    eval.InJitterOffsetX    = jitterX;
    eval.InJitterOffsetY    = jitterY;
    eval.InRenderSubrectDimensions.Width  = inWidth;
    eval.InRenderSubrectDimensions.Height = inHeight;
    eval.InReset            = reset ? 1 : 0;
    eval.InMVScaleX         = motionScaleX;
    eval.InMVScaleY         = motionScaleY;

    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, s_feature, s_params, &eval);
    if (NVSDK_NGX_FAILED(r)) {
        std::cerr << "[DLSS] EvaluateFeature failed: 0x"
                  << std::hex << (uint32_t)r << std::dec << "\n";
    }
}

void dlssShutdown(VkDevice device) {
    if (!s_initDone) return;
    if (s_feature) {
        if (device) vkDeviceWaitIdle(device);
        NVSDK_NGX_VULKAN_ReleaseFeature(s_feature);
        s_feature = nullptr;
        s_featureIn = s_featureOut = {0, 0};
    }
    if (s_params) {
        NVSDK_NGX_VULKAN_DestroyParameters(s_params);
        s_params = nullptr;
    }
    if (s_available) {
        NVSDK_NGX_VULKAN_Shutdown1(device);
    }
    s_available = false;
    s_initDone  = false;
    s_device    = VK_NULL_HANDLE;
}
