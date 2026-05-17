#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include <memory>
#include <string>
#include <cstdint>

// 3D spatial audio via OpenAL Soft. OpenAL headers are confined to audio.cpp
// (PIMPL) so <AL/*> does not propagate through engine.h.
//
// ECS integration: an entity with an AudioSourceComponent gets an OpenAL
// source lazily. update() positions each source from its TransformComponent
// and the listener from the camera, so panning/attenuation track the scene.
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();
    AudioEngine(const AudioEngine&)            = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    void init();
    void cleanup();
    bool ok() const;

    // Load a 16/8-bit PCM .wav. Returns an opaque buffer handle, or 0 on
    // failure. Cached by path (same file returns the same handle).
    uint32_t loadWav(const std::string& path);

    // Procedurally generate a looping sine tone (mono 16-bit). Lets the demo
    // play positional audio with no asset files on disk.
    uint32_t createTone(float freqHz, float seconds, float amplitude = 0.6f);

    // Per-frame: create OpenAL sources for new AudioSourceComponents, push
    // their TransformComponent position + params, start/stop to match the
    // component's `playing` flag, and update the listener from the camera.
    void update(entt::registry& reg,
                const glm::vec3& listenerPos,
                const glm::vec3& listenerForward,
                const glm::vec3& listenerUp,
                const glm::vec3& listenerVel);

    // Detach + delete the OpenAL source for an entity (before it/its
    // component is destroyed).
    void removeSource(entt::registry& reg, entt::entity e);

    void  setMasterGain(float g);
    float masterGain() const { return m_masterGain; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    float m_masterGain = 1.0f;
};
