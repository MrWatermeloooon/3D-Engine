#pragma once

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <string>

// Binary scene serialization. Self-contained little-endian format, no deps.
//
// Saves the common gameplay/visual components (name, transform, mesh,
// material, lights, rotator, rigid body, audio source, script). Runtime-only
// handles (Jolt body id, OpenAL buffer/source ids) are NOT written — on load
// they reset to "uncreated" so PhysicsWorld/AudioEngine rebuild them.
//
// Resource references (mesh/texture handles) are stored as their runtime
// integer ids. That round-trips correctly for the built-in cube/texture and
// within a session; meshes/textures imported from files in a different order
// across sessions are a known limitation (path-based resol is future work).
//
// load() destroys all existing entities first (EnTT on_destroy hooks clean up
// physics/audio/script state), then recreates them.
namespace SceneSerializer {
    bool save(const entt::registry& reg, const std::string& path);
    bool load(entt::registry& reg, const std::string& path);

    // In-memory variants (used by the undo system for scene snapshots).
    std::string saveToBuffer(const entt::registry& reg);
    bool        loadFromBuffer(entt::registry& reg, const std::string& buf);
}

// A prefab is a single entity's component set saved as a reusable template,
// using the exact same per-entity codec as the scene serializer. Instantiate
// stamps out an independent copy (runtime physics/audio/script handles reset
// so each instance rebuilds its own), optionally overriding the position.
namespace Prefab {
    bool save(const entt::registry& reg, entt::entity e, const std::string& path);
    entt::entity instantiate(entt::registry& reg, const std::string& path,
                             const glm::vec3* positionOverride = nullptr);
}
