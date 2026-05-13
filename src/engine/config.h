#pragma once

#include <cstdint>

// ── Engine-wide configuration constants ─────────────────────────────────────
//
// Single source of truth for limits that the C++ side and shader/UBO side
// must agree on. Previously these were scattered across 5+ headers and the
// shader literal values drifted out of sync silently. New consumers must
// include this header — the legacy headers still define the same names for
// backward compatibility but now source them from here.
//
// When changing any value here, audit the matching GLSL literal:
//   MAX_LIGHTS          → mesh.frag (`const int MAX_LIGHTS = 32;`)
//   SHADOW_CASCADE_COUNT → mesh.frag (`const int CASCADE_COUNT = 4;`)
//   MAX_BONES           → mesh_skinned.vert bone palette size
//   MAX_LOD             → cull.comp (`#define MAX_LOD 4`)

namespace engine_config {

// Render scheduling
inline constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

// Lighting
inline constexpr uint32_t MAX_LIGHTS           = 32;

// Shadows
inline constexpr uint32_t SHADOW_CASCADE_COUNT = 4;
inline constexpr uint32_t SHADOW_MAP_SIZE      = 2048;

// Skeletal animation
inline constexpr uint32_t MAX_BONES            = 64;

// Per-mesh LOD chain
inline constexpr uint32_t MAX_LOD              = 4;

// Post-FX
inline constexpr uint32_t BLOOM_MIP_COUNT      = 4;
inline constexpr uint32_t SSAO_KERNEL_SIZE     = 16;

// GPU-driven culling pools
inline constexpr uint32_t MAX_INSTANCES_PER_FRAME = 16384;
inline constexpr uint32_t MAX_BATCHES_PER_FRAME   = 1024;

} // namespace engine_config
