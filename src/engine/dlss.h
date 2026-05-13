#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

// DLSS Super Resolution integration via NVIDIA NGX.
//
// Phase 4a (this header): init / capability query / shutdown only. No
// feature creation, no evaluation yet — just proves the NGX SDK links and
// loads the runtime DLL. Subsequent phases add jitter, motion vectors,
// the actual upscale pass, and UI.
//
// All entry points safely no-op when DLSS_AVAILABLE is false.

// Quality presets exposed to the UI. Numeric values match
// NVSDK_NGX_PerfQuality_Value so they can be cast directly.
enum class DlssQuality : int {
    Off              = -1,
    DLAA             = 4,  // matches NVSDK_NGX_PerfQuality_Value_DLAA
    Quality          = 2,  // matches NVSDK_NGX_PerfQuality_Value_MaxQuality
    Balanced         = 1,  // matches NVSDK_NGX_PerfQuality_Value_Balanced
    Performance      = 0,  // matches NVSDK_NGX_PerfQuality_Value_MaxPerf
    UltraPerformance = 3,  // matches NVSDK_NGX_PerfQuality_Value_UltraPerformance
};

// User-controlled DLSS toggles. Mirrors RtSettings layout for consistency.
// `enabled` is the master switch; if false, the engine renders natively.
// `quality` chooses the render-resolution multiplier (DLAA = 1.0, Quality =
// 0.667, Balanced = 0.58, Performance = 0.5, UltraPerformance = 0.333).
struct DlssSettings {
    bool        enabled = false;          // user toggles via the DLSS panel
    DlssQuality quality = DlssQuality::Quality;
    bool        jitterEnabled = true;     // sub-pixel jitter on the camera proj
};

// Halton(2,3) low-discrepancy sub-pixel jitter, returned as offset in
// [-0.5, +0.5] for a single pixel cell. `index` cycles 1..N (skip 0 — it's
// the origin). Standard DLSS recommendation: 8-frame Halton cycle, scaled by
// render-scale^2 so denser cycles for higher upscale ratios.
glm::vec2 haltonJitter(uint32_t index);

// Three-phase init. NGX needs specific Vulkan extensions to be enabled at
// instance and device creation. Get them with the two query functions before
// creating the corresponding Vulkan object, then call dlssInit() after the
// VkDevice exists. Without those extensions NGX init returns
// FeatureNotSupported (0xBAD00005) — that was our previous symptom.
//
// Returned pointers are stable for the lifetime of the process (backed by
// internal std::strings in dlss.cpp). Empty lists on failure.

std::vector<const char*> dlssRequiredInstanceExtensions();
std::vector<const char*> dlssRequiredDeviceExtensions(VkInstance instance,
                                                      VkPhysicalDevice physicalDevice);

// Init NGX against an already-created device. Sets `DLSS_AVAILABLE` if the
// driver, GPU, and DLL all check out. Idempotent — safe to call multiple
// times; only the first invocation does work.
void dlssInit(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);

// True iff the runtime is alive and reported a hardware DLSS capability.
bool dlssAvailable();

// Query NGX for the optimal *input* render size given a final upscale target
// (typically the swapchain extent) and quality preset. Returns (inWidth,
// inHeight) on success. On failure, returns (targetW, targetH) so callers can
// always render at native if NGX rejects the combo. DLAA always returns
// target = input.
//
// Must be called after dlssInit returned successfully.
struct DlssRenderSize {
    uint32_t inWidth;
    uint32_t inHeight;
    bool     ok;        // false → quality not supported at this target size
};
DlssRenderSize dlssGetOptimalRenderSize(uint32_t targetWidth,
                                        uint32_t targetHeight,
                                        DlssQuality quality);

// Create the DLSS feature for the given (input, target) pair + quality.
// Records an internal feature handle; subsequent calls to dlssEvaluate use it.
// The create call records GPU work on `cmd` (DLSS needs to upload weights),
// so callers must submit `cmd` and wait/idle before destroying the cmd buffer.
// Releases any previously-created feature first — safe to call repeatedly.
bool dlssCreateFeature(VkCommandBuffer cmd,
                       uint32_t inWidth,  uint32_t inHeight,
                       uint32_t outWidth, uint32_t outHeight,
                       DlssQuality quality);

// Destroy the active DLSS feature. Waits for the device first — callers must
// not have any in-flight frames referencing it. Safe to call when no feature
// is active.
void dlssReleaseFeature(VkDevice device);

// True iff a DLSS feature has been created and is ready to evaluate.
bool dlssFeatureReady();

// Inputs/outputs the active feature was created with. {0,0} if none.
VkExtent2D dlssFeatureInputExtent();
VkExtent2D dlssFeatureOutputExtent();

// Record a DLSS evaluate into `cmd`. All resources must already be in the
// expected layouts:
//   - colorView, depthView, motionView in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
//   - outputView                       in VK_IMAGE_LAYOUT_GENERAL
// The caller is responsible for barriers before and after this call.
//
// jitterX/Y are in *input pixel* space (the same value the engine applied to
// the camera projection, multiplied by inputExtent / 2.0 from NDC).
//
// motionScaleX/Y let NGX convert motion vectors to input-pixel units. Our
// mesh.frag writes NDC-space (prev - curr); to get input-pixel deltas we need
// MV * (inWidth/2, inHeight/2) — but with a sign flip on Y because Vulkan
// NDC y points down vs. DLSS' input-pixel y. Caller passes the final scales.
//
// reset=true on the first frame after feature creation, scene cut, or any
// camera teleport, to flush DLSS' history buffer.
void dlssEvaluate(VkCommandBuffer cmd,
                  VkImage colorImage,  VkImageView colorView,  VkFormat colorFormat,
                  VkImage depthImage,  VkImageView depthView,  VkFormat depthFormat,
                  VkImage motionImage, VkImageView motionView, VkFormat motionFormat,
                  VkImage outputImage, VkImageView outputView, VkFormat outputFormat,
                  uint32_t inWidth, uint32_t inHeight,
                  uint32_t outWidth, uint32_t outHeight,
                  float jitterX, float jitterY,
                  float motionScaleX, float motionScaleY,
                  bool reset);

// Release NGX. Safe to call even if init failed or never ran.
void dlssShutdown(VkDevice device);
