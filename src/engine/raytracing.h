#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

#include "buffer.h"

// Ray tracing infrastructure.
//
// Phase 1a (this header): empty scaffolding — struct declarations + lifecycle
// stubs only. No BLAS/TLAS build, no descriptor writes. The goal is just to
// get the types reachable from engine.cpp without touching the rendering path.
//
// Phase 1b will fill `RtMeshGeometry` (one BLAS per Mesh) and `RtScene::tlas`
// (rebuilt per frame from the GPU-cull instance buffer).
// Phase 1c will add a descriptor binding for the TLAS in mesh.frag and replace
// the CSM PCF lookup with a `rayQueryEXT` traced toward the sun.

// User-controlled RT toggles. `enabled` gates the whole RT pipeline regardless
// of hardware capability: even with RT_SUPPORTED == true, the engine falls
// back to the existing CSM path when `enabled` is false. The individual
// feature flags will be wired up by phases 1c → 3.
struct RtSettings {
    bool  enabled            = true;
    bool  shadows            = true;   // Phase 1c — replaces CSM PCF lookup
    bool  reflections        = true;   // Phase 2 — adds reflections on metals
    bool  gi                 = true;   // Phase 3 — one-bounce indirect diffuse
    bool  sunOnly            = true;   // limit RT shadows to the first directional light (CSM parity).
                                       // Uncheck to let point/spot lights also cast — they can't with CSM.
    float shadowSoftness     = 0.005f; // light angular radius in radians (sun ≈ 0.0046)
    int   shadowSamples      = 16;     // rays per fragment per light — higher = smoother
    int   reflectionSamples   = 8;      // rays per fragment for reflection (1..16)
    float reflectionMaxDist   = 100.0f; // tMax for reflection rays
    float reflectionIntensity = 1.0f;   // 0..2 multiplier on indirect specular
    int   giSamples           = 12;     // rays per fragment for GI (1..16)
    float giMaxDist           = 30.0f;  // tMax for GI rays
    float giIntensity         = 0.15f;  // mix factor: 0 = smooth ambient, 1 = pure (noisy) GI
};

struct RtMeshGeometry {
    // BLAS handle + the device-local buffer that backs it. Owned by the
    // ResourceManager alongside the Mesh once built.
    VkAccelerationStructureKHR accel        = VK_NULL_HANDLE;
    AllocatedBuffer            accelBuffer  = {};
    VkDeviceAddress            deviceAddress = 0;
};

struct Mesh; // fwd

// Per-instance shading material + geometry pointers exposed to ray queries.
// Indexed by `gl_RayQueryInstanceCustomIndexEXT` inside the shader. The
// vertex/index device addresses let the hit handler interpolate the surface
// normal via the standard `(idxBuf, primIndex, barycentrics)` lookup. std430
// layout = vec4 + vec4 + uvec2 + uvec2 = 48 bytes.
struct RtInstanceMaterial {
    glm::vec4 color;       // rgb = albedo, a unused
    glm::vec4 params;      // x = metallic, y = roughness, zw reserved
    glm::uvec2 vertexAddr; // packUint2x32 → buffer_reference Vertices
    glm::uvec2 indexAddr;  // packUint2x32 → buffer_reference Indices
};

struct RtScene {
    // Top-level acceleration structure (per-frame, since instances move).
    // Backed by `tlasBuffer`. `instanceBuffer` holds the array of
    // VkAccelerationStructureInstanceKHR records consumed by the TLAS build.
    // `materialBuffer` holds the parallel `RtInstanceMaterial` array.
    // `scratchBuffer` is the build-scratch space — kept alive across frames
    // and reused as long as its capacity is sufficient.
    std::vector<VkAccelerationStructureKHR> tlas;          // one per frame in flight
    std::vector<AllocatedBuffer>            tlasBuffer;
    std::vector<AllocatedBuffer>            instanceBuffer;
    std::vector<AllocatedBuffer>            materialBuffer;
    std::vector<AllocatedBuffer>            scratchBuffer;
    std::vector<VkDeviceSize>               scratchCapacity;
    std::vector<VkDeviceSize>               tlasCapacity;
    std::vector<VkDeviceSize>               instanceCapacity;
    std::vector<VkDeviceSize>               materialCapacity;
};

// Lifecycle.
void createRtScene(RtScene& scene, VkDevice device, uint32_t framesInFlight);
void destroyRtScene(VkDevice device, RtScene& scene);

void destroyRtMeshGeometry(VkDevice device, RtMeshGeometry& geo);

// Device-address helper. Uses the cached RT_GetBufferDeviceAddress function
// pointer loaded in vulkan_init. Asserts RT_SUPPORTED — callers must guard.
VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer);

// Split a 64-bit address into a glm::uvec2 (low, high) for buffer-reference
// reconstruction in GLSL via `Vertices(addr)` after `packUint2x32`.
inline glm::uvec2 splitAddress(VkDeviceAddress a) {
    return glm::uvec2(static_cast<uint32_t>(a & 0xFFFFFFFFu),
                      static_cast<uint32_t>(a >> 32));
}

// Build the per-mesh BLAS from this mesh's vertex+index buffers. Called once
// from uploadMesh when RT is available. The vertex buffer must include
// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` and the AS-build-input flag.
// Uses a temporary scratch buffer; submits + waits via a single-time command.
void buildMeshBlas(Mesh& mesh, VkPhysicalDevice physicalDevice, VkDevice device,
                   VkCommandPool commandPool, VkQueue queue);

// Re-build the TLAS for `frame` from a flat list of instances. Each entry
// holds a 3×4 row-major transform (Vulkan convention) and a BLAS device
// address. `cmd` must be in a recording state; the build is queued into it.
// A barrier is inserted after the build so subsequent ray queries see the
// finished structure.
struct RtInstanceDesc {
    glm::mat4          transform;    // converted to row-major 3×4 inside the build
    VkDeviceAddress    blasAddress;
    RtInstanceMaterial material;     // shading data accessible at ray hits
};

void buildTlas(RtScene& scene, uint32_t frame,
               VkPhysicalDevice physicalDevice, VkDevice device,
               VkCommandBuffer cmd,
               const std::vector<RtInstanceDesc>& instances);
