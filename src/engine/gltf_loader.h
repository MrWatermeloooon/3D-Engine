#pragma once

#include "skeletal.h"
#include <string>

// glTF 2.0 importer.
//
// loadGltfSkinned populates the skeleton + skinned mesh + animations from the
// first skinned primitive found in the file. The returned mesh has its CPU-
// side `vertices`/`indices`/`skeleton`/`animations` filled and `aabbMin/Max`
// computed; the caller must still upload it via `uploadSkinnedMesh` before
// rendering.
//
// Supported attributes: POSITION, NORMAL, TEXCOORD_0, JOINTS_0 (u8 or u16),
// WEIGHTS_0 (f32 or u8/u16 normalised). Up to 4 bone influences per vertex —
// JOINTS_n / WEIGHTS_n for n > 0 is ignored.
//
// Supported animation paths: translation, rotation, scale.
// Supported sampler interpolations: linear, step (snapped to lower keyframe).
// Cubic-spline is parsed as linear with a warning printed to stderr.
//
// Returns `true` on success and `false` if the file failed to parse, no skin
// was found, or the skin had no animated joints. `errorOut` (when non-null)
// is filled with a human-readable error string on failure.
//
// Architectural note: the engine currently keeps a single shared skinned mesh
// resource (`Engine::m_skinnedMesh`), so loading a new glTF replaces the
// previous one. Multi-skin support is a future extension.
bool loadGltfSkinned(const std::string& path, SkinnedMesh& out,
                     std::string* errorOut = nullptr);
