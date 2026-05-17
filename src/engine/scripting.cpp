#include "scripting.h"
#include "components.h"
#include "physics.h"

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <cstdio>

namespace fs = std::filesystem;

namespace {
    // Lightweight scriptable handle. The registry pointer is the engine's
    // single long-lived registry, so holding it across frames is safe.
    struct LuaEntity {
        entt::registry* reg = nullptr;
        entt::entity     e  = entt::null;

        bool valid() const { return reg && reg->valid(e); }
        TransformComponent* tf() const {
            return valid() ? reg->try_get<TransformComponent>(e) : nullptr;
        }
        glm::vec3 get_position() const { auto* t = tf(); return t ? t->position : glm::vec3(0); }
        glm::vec3 get_rotation() const { auto* t = tf(); return t ? t->rotation : glm::vec3(0); }
        glm::vec3 get_scale()    const { auto* t = tf(); return t ? t->scale    : glm::vec3(1); }
        void set_position(const glm::vec3& v) { if (auto* t = tf()) t->position = v; }
        void set_rotation(const glm::vec3& v) { if (auto* t = tf()) t->rotation = v; }
        void set_scale(const glm::vec3& v)    { if (auto* t = tf()) t->scale    = v; }
        std::string name() const {
            if (valid()) if (auto* n = reg->try_get<NameComponent>(e)) return n->name;
            return {};
        }
        void destroy() { if (valid()) reg->destroy(e); }
    };

    int glfwKeyFor(const std::string& s) {
        if (s.size() == 1) {
            char c = s[0];
            if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
            if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return c;
        }
        if (s == "SPACE")  return GLFW_KEY_SPACE;
        if (s == "SHIFT")  return GLFW_KEY_LEFT_SHIFT;
        if (s == "CTRL")   return GLFW_KEY_LEFT_CONTROL;
        if (s == "ENTER")  return GLFW_KEY_ENTER;
        if (s == "ESC")    return GLFW_KEY_ESCAPE;
        if (s == "UP")     return GLFW_KEY_UP;
        if (s == "DOWN")   return GLFW_KEY_DOWN;
        if (s == "LEFT")   return GLFW_KEY_LEFT;
        if (s == "RIGHT")  return GLFW_KEY_RIGHT;
        return GLFW_KEY_UNKNOWN;
    }
}

struct ScriptEngine::Impl {
    sol::state lua;
    GLFWwindow* window = nullptr;
    PhysicsWorld* physics = nullptr;
    std::function<entt::entity(const glm::vec3&)> spawnCube;
    // Set at the top of each update(); valid while script callbacks run, so
    // the spawn_cube/raycast bindings can wrap entities against it.
    entt::registry* regBindingPtr = nullptr;

    struct PerScript {
        std::string path;
        fs::file_time_type lastWrite{};
        bool started = false;
        bool errored = false;
        sol::environment env;
        sol::protected_function onStart;
        sol::protected_function onUpdate;
    };
    std::unordered_map<std::uint32_t, PerScript> scripts;
    std::vector<std::string> logLines;

    void logMsg(const std::string& s) {
        std::fprintf(stderr, "[Lua] %s\n", s.c_str());
        logLines.push_back(s);
        if (logLines.size() > 200) logLines.erase(logLines.begin());
    }
};

ScriptEngine::ScriptEngine()  = default;
ScriptEngine::~ScriptEngine() { cleanup(); }

void ScriptEngine::init(GLFWwindow* window, PhysicsWorld* physics,
                        std::function<entt::entity(const glm::vec3&)> spawnCube) {
    if (m_impl) return;
    m_impl = std::make_unique<Impl>();
    m_impl->window   = window;
    m_impl->physics  = physics;
    m_impl->spawnCube = std::move(spawnCube);

    sol::state& lua = m_impl->lua;
    lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::string,
                       sol::lib::table, sol::lib::os);

    lua.new_usertype<glm::vec3>("vec3",
        sol::constructors<glm::vec3(), glm::vec3(float, float, float)>(),
        "x", &glm::vec3::x, "y", &glm::vec3::y, "z", &glm::vec3::z,
        "length",     [](const glm::vec3& v) { return glm::length(v); },
        "normalized", [](const glm::vec3& v) {
            float l = glm::length(v); return l > 1e-6f ? v / l : glm::vec3(0); },
        "dot", [](const glm::vec3& a, const glm::vec3& b) { return glm::dot(a, b); },
        sol::meta_function::addition,
            [](const glm::vec3& a, const glm::vec3& b) { return a + b; },
        sol::meta_function::subtraction,
            [](const glm::vec3& a, const glm::vec3& b) { return a - b; },
        sol::meta_function::multiplication,
            [](const glm::vec3& a, float s) { return a * s; });

    lua.new_usertype<LuaEntity>("Entity",
        "valid",        &LuaEntity::valid,
        "get_position", &LuaEntity::get_position,
        "set_position", &LuaEntity::set_position,
        "get_rotation", &LuaEntity::get_rotation,
        "set_rotation", &LuaEntity::set_rotation,
        "get_scale",    &LuaEntity::get_scale,
        "set_scale",    &LuaEntity::set_scale,
        "get_name",     &LuaEntity::name,
        "destroy",      &LuaEntity::destroy);

    lua.set_function("log", [this](const std::string& s) { m_impl->logMsg(s); });

    lua.set_function("spawn_cube", [this](const glm::vec3& p) -> sol::object {
        if (!m_impl->spawnCube) return sol::nil;
        entt::entity e = m_impl->spawnCube(p);
        return sol::make_object(m_impl->lua,
                                LuaEntity{ m_impl->regBindingPtr, e });
    });

    lua.set_function("raycast",
        [this](const glm::vec3& o, const glm::vec3& d, float maxDist) -> sol::object {
            if (!m_impl->physics) return sol::nil;
            auto hit = m_impl->physics->raycast(o, d, maxDist);
            if (!hit) return sol::nil;
            sol::table t = m_impl->lua.create_table();
            t["entity"]   = LuaEntity{ m_impl->regBindingPtr, hit->entity };
            t["point"]    = hit->point;
            t["normal"]   = hit->normal;
            t["distance"] = hit->distance;
            return t;
        });

    sol::table input = lua.create_table();
    input.set_function("key", [this](const std::string& name) {
        if (!m_impl->window) return false;
        int k = glfwKeyFor(name);
        return k != GLFW_KEY_UNKNOWN &&
               glfwGetKey(m_impl->window, k) == GLFW_PRESS;
    });
    lua["input"] = input;

    lua["app"] = lua.create_table();
    lua["app"]["time"] = 0.0;
    lua["app"]["dt"]   = 0.0;
}

void ScriptEngine::cleanup() {
    if (!m_impl) return;
    m_impl->scripts.clear();
    m_impl.reset();
}

void ScriptEngine::onDestroy(entt::registry&, entt::entity e) {
    if (m_impl) m_impl->scripts.erase(entt::to_integral(e));
}

const std::vector<std::string>& ScriptEngine::log() const {
    static const std::vector<std::string> empty;
    return m_impl ? m_impl->logLines : empty;
}
void ScriptEngine::clearLog() { if (m_impl) m_impl->logLines.clear(); }

void ScriptEngine::update(entt::registry& reg, float dt, double appTime) {
    if (!m_impl) return;
    m_impl->regBindingPtr = &reg;
    m_impl->lua["app"]["time"] = appTime;
    m_impl->lua["app"]["dt"]   = dt;

    auto view = reg.view<ScriptComponent>();
    for (auto e : view) {
        auto& sc = view.get<ScriptComponent>(e);
        if (!sc.enabled || sc.path.empty()) continue;

        auto& ps = m_impl->scripts[entt::to_integral(e)];

        // (Re)load when first seen, path changed, or the file changed on disk.
        bool needLoad = (ps.path != sc.path);
        std::error_code ec;
        auto mtime = fs::last_write_time(sc.path, ec);
        if (!needLoad && !ec && mtime != ps.lastWrite) needLoad = true;

        if (needLoad) {
            ps.path     = sc.path;
            ps.started  = false;
            ps.errored  = false;
            ps.env      = sol::environment(m_impl->lua, sol::create,
                                           m_impl->lua.globals());
            auto r = m_impl->lua.safe_script_file(sc.path, ps.env,
                         sol::script_pass_on_error);
            if (!r.valid()) {
                sol::error err = r;
                m_impl->logMsg(std::string("load error: ") + err.what());
                ps.errored = true;
            } else {
                ps.onStart  = ps.env["on_start"];
                ps.onUpdate = ps.env["on_update"];
                if (!ec) ps.lastWrite = mtime;
                m_impl->logMsg("loaded " + sc.path);
            }
        }
        if (ps.errored) continue;

        LuaEntity self{ &reg, e };
        if (!ps.started) {
            ps.started = true;
            if (ps.onStart.valid()) {
                auto r = ps.onStart(self);
                if (!r.valid()) {
                    sol::error err = r;
                    m_impl->logMsg(std::string("on_start: ") + err.what());
                    ps.errored = true;
                    continue;
                }
            }
        }
        if (ps.onUpdate.valid()) {
            auto r = ps.onUpdate(self, dt);
            if (!r.valid()) {
                sol::error err = r;
                m_impl->logMsg(std::string("on_update: ") + err.what());
                ps.errored = true;   // latch off until the file changes
            }
        }
    }
}

bool ScriptEngine::writeSampleScript(const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f <<
R"(-- Sample engine script. Edit & save while running to hot-reload.
local t = 0.0

function on_start(self)
    log("sample script started on '" .. self:get_name() .. "'")
end

function on_update(self, dt)
    t = t + dt
    -- bob up and down
    local p = self:get_position()
    p.y = 1.5 + math.sin(t * 2.0) * 0.6
    self:set_position(p)
    -- spin around Y (wrapped so the euler angle stays bounded)
    local r = self:get_rotation()
    r.y = (r.y + dt * 90.0) % 360.0
    self:set_rotation(r)
end
)";
    return true;
}
