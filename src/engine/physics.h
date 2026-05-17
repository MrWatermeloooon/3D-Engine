#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include <memory>
#include <optional>

struct RaycastHit {
    entt::entity entity = entt::null;
    glm::vec3    point{0.0f};
    glm::vec3    normal{0.0f};
    float        distance = 0.0f;
};

// Thin wrapper over a Jolt PhysicsSystem. Jolt headers are confined to
// physics.cpp (PIMPL) so they do not propagate through engine.h into the
// whole codebase (Jolt's headers are heavy and gate struct layout on build
// defines we want to keep in one translation unit).
//
// ECS integration: entities carrying a RigidBodyComponent get a Jolt body
// lazily via syncNewBodies(); step() advances the simulation on a fixed
// timestep and writes Dynamic/Kinematic poses back into TransformComponent.
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld&)            = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    void init();
    void cleanup();

    // Create Jolt bodies for entities that have a RigidBodyComponent whose
    // bodyId is still the "uncreated" sentinel. Reads the entity's
    // TransformComponent for the initial pose.
    void syncNewBodies(entt::registry& reg);

    // Advance the simulation. Accumulates real time and runs zero or more
    // fixed (1/60 s) sub-steps, then writes Dynamic body poses (and reads
    // Kinematic targets) through TransformComponent. No-op while paused.
    void step(entt::registry& reg, float dt);

    // Detach + destroy the Jolt body backing an entity (call before the
    // entity or its RigidBodyComponent is destroyed).
    void removeBody(entt::registry& reg, entt::entity e);

    std::optional<RaycastHit> raycast(const glm::vec3& origin,
                                      const glm::vec3& dir,
                                      float maxDistance) const;

    void      setGravity(const glm::vec3& g);
    glm::vec3 gravity() const;

    void  setPaused(bool p)     { m_paused = p; }
    bool  paused() const        { return m_paused; }
    void  setTimeScale(float s) { m_timeScale = s; }
    float timeScale() const     { return m_timeScale; }

    int bodyCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    bool  m_paused      = false;
    float m_timeScale   = 1.0f;
    float m_accumulator = 0.0f;
};
