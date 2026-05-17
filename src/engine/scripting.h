#pragma once

#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;
class PhysicsWorld;

// Lua scripting via sol2. The sol2/Lua headers are confined to scripting.cpp
// (PIMPL) — they are heavy and we don't want them in engine.h.
//
// A script is a .lua file attached to an entity via ScriptComponent. It may
// define `on_start(self)` (called once) and `on_update(self, dt)` (called
// every frame). `self` is an Entity userdata with transform/name/destroy
// methods. Files hot-reload when their modification time changes; a runtime
// error disables that one script (logged) until the file changes again.
class ScriptEngine {
public:
    ScriptEngine();
    ~ScriptEngine();
    ScriptEngine(const ScriptEngine&)            = delete;
    ScriptEngine& operator=(const ScriptEngine&) = delete;

    // window  : for the input.key() binding
    // physics : for the raycast() binding (may be null)
    // spawnCube: engine-supplied factory so scripting needn't depend on
    //            ResourceManager; returns the new entity.
    void init(GLFWwindow* window, PhysicsWorld* physics,
              std::function<entt::entity(const glm::vec3&)> spawnCube);
    void cleanup();

    void update(entt::registry& reg, float dt, double appTime);

    // entt on_destroy hook — drop a destroyed entity's script state.
    void onDestroy(entt::registry& reg, entt::entity e);

    // UI: rolling log of script prints + errors.
    const std::vector<std::string>& log() const;
    void clearLog();

    // Write a self-contained sample script to `path` (for the demo button).
    static bool writeSampleScript(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
