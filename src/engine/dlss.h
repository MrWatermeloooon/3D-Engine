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
    bool        enabled = false;          // off by default until upscale is wired in 4c
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

// Release NGX. Safe to call even if init failed or never ran.
void dlssShutdown(VkDevice device);
