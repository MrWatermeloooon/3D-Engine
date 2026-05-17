#include "serialization.h"
#include "components.h"

#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace {
    constexpr char     kSceneMagic[4]  = { 'V', 'E', 'S', 'C' };
    constexpr char     kPrefabMagic[4] = { 'V', 'E', 'P', 'F' };
    constexpr uint32_t kVersion        = 1;

    enum CompBit : uint32_t {
        C_NAME      = 1u << 0,
        C_TRANSFORM = 1u << 1,
        C_MESH      = 1u << 2,
        C_MATERIAL  = 1u << 3,
        C_DIRLIGHT  = 1u << 4,
        C_POINTLIGHT= 1u << 5,
        C_SPOTLIGHT = 1u << 6,
        C_ROTATOR   = 1u << 7,
        C_RIGIDBODY = 1u << 8,
        C_AUDIO     = 1u << 9,
        C_SCRIPT    = 1u << 10,
    };

    // Buffer-based codec so the same path serves files (scene/prefab) and
    // in-memory snapshots (undo).
    struct Writer {
        std::string buf;
        template <class T> void pod(const T& v) {
            buf.append(reinterpret_cast<const char*>(&v), sizeof(T));
        }
        void raw(const char* p, size_t n) { buf.append(p, n); }
        void str(const std::string& s) {
            pod(static_cast<uint32_t>(s.size()));
            buf.append(s);
        }
    };

    struct Reader {
        const std::string& buf;
        size_t pos = 0;
        bool   ok  = true;
        explicit Reader(const std::string& b) : buf(b) {}
        template <class T> T pod() {
            T v{};
            if (pos + sizeof(T) <= buf.size()) {
                std::memcpy(&v, buf.data() + pos, sizeof(T));
                pos += sizeof(T);
            } else ok = false;
            return v;
        }
        bool raw(char* p, size_t n) {
            if (pos + n > buf.size()) { ok = false; return false; }
            std::memcpy(p, buf.data() + pos, n); pos += n; return true;
        }
        std::string str() {
            uint32_t n = pod<uint32_t>();
            if (!ok || pos + n > buf.size()) { ok = false; return {}; }
            std::string s(buf.data() + pos, n);
            pos += n;
            return s;
        }
    };

    bool readFile(const std::string& path, std::string& out) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        std::streamsize n = f.tellg();
        if (n < 0) return false;
        out.resize(static_cast<size_t>(n));
        f.seekg(0);
        if (n) f.read(&out[0], n);
        return true;
    }
    bool writeFile(const std::string& path, const std::string& data) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        if (!data.empty())
            f.write(data.data(), static_cast<std::streamsize>(data.size()));
        return static_cast<bool>(f);
    }

    // ── Shared per-entity codec (scene serializer AND prefabs use this) ──
    void writeEntity(Writer& w, const entt::registry& reg, entt::entity e) {
        uint32_t mask = 0;
        if (reg.all_of<NameComponent>(e))              mask |= C_NAME;
        if (reg.all_of<TransformComponent>(e))         mask |= C_TRANSFORM;
        if (reg.all_of<MeshComponent>(e))              mask |= C_MESH;
        if (reg.all_of<MaterialComponent>(e))          mask |= C_MATERIAL;
        if (reg.all_of<DirectionalLightComponent>(e))  mask |= C_DIRLIGHT;
        if (reg.all_of<PointLightComponent>(e))        mask |= C_POINTLIGHT;
        if (reg.all_of<SpotLightComponent>(e))         mask |= C_SPOTLIGHT;
        if (reg.all_of<RotatorComponent>(e))           mask |= C_ROTATOR;
        if (reg.all_of<RigidBodyComponent>(e))         mask |= C_RIGIDBODY;
        if (reg.all_of<AudioSourceComponent>(e))       mask |= C_AUDIO;
        if (reg.all_of<ScriptComponent>(e))            mask |= C_SCRIPT;
        w.pod(mask);

        if (mask & C_NAME)  w.str(reg.get<NameComponent>(e).name);
        if (mask & C_TRANSFORM) {
            const auto& t = reg.get<TransformComponent>(e);
            w.pod(t.position); w.pod(t.rotation); w.pod(t.scale);
        }
        if (mask & C_MESH)
            w.pod(reg.get<MeshComponent>(e).handle.id);
        if (mask & C_MATERIAL) {
            const auto& m = reg.get<MaterialComponent>(e);
            w.pod(m.texture.id); w.pod(m.normalTexture.id);
            w.pod(m.heightTexture.id);
            w.pod(m.color); w.pod(m.metallic); w.pod(m.roughness);
            w.pod(m.parallaxScale);
        }
        if (mask & C_DIRLIGHT) {
            const auto& l = reg.get<DirectionalLightComponent>(e);
            w.pod(l.direction); w.pod(l.color); w.pod(l.intensity);
            w.pod(l.castsShadows);
        }
        if (mask & C_POINTLIGHT) {
            const auto& l = reg.get<PointLightComponent>(e);
            w.pod(l.color); w.pod(l.intensity); w.pod(l.range);
        }
        if (mask & C_SPOTLIGHT) {
            const auto& l = reg.get<SpotLightComponent>(e);
            w.pod(l.direction); w.pod(l.color); w.pod(l.intensity);
            w.pod(l.range); w.pod(l.innerConeDeg); w.pod(l.outerConeDeg);
        }
        if (mask & C_ROTATOR) {
            const auto& r = reg.get<RotatorComponent>(e);
            w.pod(r.axis); w.pod(r.speed);
        }
        if (mask & C_RIGIDBODY) {
            const auto& rb = reg.get<RigidBodyComponent>(e);
            w.pod(rb.motion); w.pod(rb.shape); w.pod(rb.autoShapeFromScale);
            w.pod(rb.halfExtent); w.pod(rb.radius); w.pod(rb.halfHeight);
            w.pod(rb.mass); w.pod(rb.friction); w.pod(rb.restitution);
        }
        if (mask & C_AUDIO) {
            const auto& a = reg.get<AudioSourceComponent>(e);
            w.str(a.clip);
            w.pod(a.toneHz); w.pod(a.toneSeconds);
            w.pod(a.gain); w.pod(a.pitch); w.pod(a.loop); w.pod(a.playing);
            w.pod(a.spatial); w.pod(a.refDistance); w.pod(a.maxDistance);
            w.pod(a.rolloff);
        }
        if (mask & C_SCRIPT) {
            const auto& s = reg.get<ScriptComponent>(e);
            w.str(s.path); w.pod(s.enabled);
        }
    }

    entt::entity readEntity(Reader& r, entt::registry& reg) {
        auto e = reg.create();
        uint32_t mask = r.pod<uint32_t>();

        if (mask & C_NAME) {
            NameComponent n; n.name = r.str();
            reg.emplace<NameComponent>(e, n);
        }
        if (mask & C_TRANSFORM) {
            TransformComponent t;
            t.position = r.pod<glm::vec3>();
            t.rotation = r.pod<glm::vec3>();
            t.scale    = r.pod<glm::vec3>();
            reg.emplace<TransformComponent>(e, t);
        }
        if (mask & C_MESH) {
            MeshComponent m; m.handle.id = r.pod<uint32_t>();
            reg.emplace<MeshComponent>(e, m);
        }
        if (mask & C_MATERIAL) {
            MaterialComponent m;
            m.texture.id       = r.pod<uint32_t>();
            m.normalTexture.id = r.pod<uint32_t>();
            m.heightTexture.id = r.pod<uint32_t>();
            m.color         = r.pod<glm::vec4>();
            m.metallic      = r.pod<float>();
            m.roughness     = r.pod<float>();
            m.parallaxScale = r.pod<float>();
            reg.emplace<MaterialComponent>(e, m);
        }
        if (mask & C_DIRLIGHT) {
            DirectionalLightComponent l;
            l.direction    = r.pod<glm::vec3>();
            l.color        = r.pod<glm::vec3>();
            l.intensity    = r.pod<float>();
            l.castsShadows = r.pod<bool>();
            reg.emplace<DirectionalLightComponent>(e, l);
        }
        if (mask & C_POINTLIGHT) {
            PointLightComponent l;
            l.color     = r.pod<glm::vec3>();
            l.intensity = r.pod<float>();
            l.range     = r.pod<float>();
            reg.emplace<PointLightComponent>(e, l);
        }
        if (mask & C_SPOTLIGHT) {
            SpotLightComponent l;
            l.direction    = r.pod<glm::vec3>();
            l.color        = r.pod<glm::vec3>();
            l.intensity    = r.pod<float>();
            l.range        = r.pod<float>();
            l.innerConeDeg = r.pod<float>();
            l.outerConeDeg = r.pod<float>();
            reg.emplace<SpotLightComponent>(e, l);
        }
        if (mask & C_ROTATOR) {
            RotatorComponent c;
            c.axis  = r.pod<glm::vec3>();
            c.speed = r.pod<float>();
            reg.emplace<RotatorComponent>(e, c);
        }
        if (mask & C_RIGIDBODY) {
            RigidBodyComponent rb;
            rb.motion             = r.pod<RigidBodyComponent::Motion>();
            rb.shape              = r.pod<RigidBodyComponent::Shape>();
            rb.autoShapeFromScale = r.pod<bool>();
            rb.halfExtent  = r.pod<glm::vec3>();
            rb.radius      = r.pod<float>();
            rb.halfHeight  = r.pod<float>();
            rb.mass        = r.pod<float>();
            rb.friction    = r.pod<float>();
            rb.restitution = r.pod<float>();
            rb.bodyId = 0xFFFFFFFFu;          // each instance rebuilds its body
            reg.emplace<RigidBodyComponent>(e, rb);
        }
        if (mask & C_AUDIO) {
            AudioSourceComponent a;
            a.clip        = r.str();
            a.toneHz      = r.pod<float>();
            a.toneSeconds = r.pod<float>();
            a.gain        = r.pod<float>();
            a.pitch       = r.pod<float>();
            a.loop        = r.pod<bool>();
            a.playing     = r.pod<bool>();
            a.spatial     = r.pod<bool>();
            a.refDistance = r.pod<float>();
            a.maxDistance = r.pod<float>();
            a.rolloff     = r.pod<float>();
            a.bufferId = 0; a.sourceId = 0;   // each instance rebuilds its source
            reg.emplace<AudioSourceComponent>(e, a);
        }
        if (mask & C_SCRIPT) {
            ScriptComponent s;
            s.path    = r.str();
            s.enabled = r.pod<bool>();
            reg.emplace<ScriptComponent>(e, s);
        }
        return e;
    }
}

std::string SceneSerializer::saveToBuffer(const entt::registry& reg) {
    Writer w;
    w.raw(kSceneMagic, 4);
    w.pod(kVersion);
    // Every entity in this engine is created with a TransformComponent
    // (Scene::createEntity + all spawn paths), so this view enumerates them
    // all without relying on a specific EnTT all-entities iteration API.
    std::vector<entt::entity> ents;
    for (auto e : reg.view<TransformComponent>()) ents.push_back(e);
    w.pod(static_cast<uint32_t>(ents.size()));
    for (auto e : ents) writeEntity(w, reg, e);
    return std::move(w.buf);
}


bool SceneSerializer::loadFromBuffer(entt::registry& reg,
                                     const std::string& data) {
    Reader r(data);
    char magic[4];
    if (!r.raw(magic, 4) ||
        std::string(magic, 4) != std::string(kSceneMagic, 4)) {
        std::fprintf(stderr, "[Scene] load: bad magic\n"); return false;
    }
    uint32_t ver = r.pod<uint32_t>();
    if (ver != kVersion) {
        std::fprintf(stderr, "[Scene] load: version %u unsupported\n", ver);
        return false;
    }
    reg.clear();   // on_destroy hooks free Jolt/OpenAL/script state
    uint32_t count = r.pod<uint32_t>();
    for (uint32_t i = 0; i < count && r.ok; ++i) readEntity(r, reg);
    return true;
}

bool SceneSerializer::save(const entt::registry& reg, const std::string& path) {
    if (!writeFile(path, saveToBuffer(reg))) {
        std::fprintf(stderr, "[Scene] save: cannot write %s\n", path.c_str());
        return false;
    }
    return true;
}

bool SceneSerializer::load(entt::registry& reg, const std::string& path) {
    std::string data;
    if (!readFile(path, data)) {
        std::fprintf(stderr, "[Scene] load: cannot open %s\n", path.c_str());
        return false;
    }
    if (!loadFromBuffer(reg, data)) return false;
    std::fprintf(stderr, "[Scene] loaded from %s\n", path.c_str());
    return true;
}

bool Prefab::save(const entt::registry& reg, entt::entity e,
                  const std::string& path) {
    if (!reg.valid(e)) return false;
    Writer w;
    w.raw(kPrefabMagic, 4);
    w.pod(kVersion);
    writeEntity(w, reg, e);
    if (!writeFile(path, w.buf)) {
        std::fprintf(stderr, "[Prefab] save: cannot write %s\n", path.c_str());
        return false;
    }
    return true;
}

entt::entity Prefab::instantiate(entt::registry& reg, const std::string& path,
                                 const glm::vec3* positionOverride) {
    std::string data;
    if (!readFile(path, data)) {
        std::fprintf(stderr, "[Prefab] instantiate: cannot open %s\n",
                     path.c_str());
        return entt::null;
    }
    Reader r(data);
    char magic[4];
    if (!r.raw(magic, 4) ||
        std::string(magic, 4) != std::string(kPrefabMagic, 4)) {
        std::fprintf(stderr, "[Prefab] bad magic in %s\n", path.c_str());
        return entt::null;
    }
    uint32_t ver = r.pod<uint32_t>();
    if (ver != kVersion) {
        std::fprintf(stderr, "[Prefab] version %u unsupported\n", ver);
        return entt::null;
    }
    entt::entity e = readEntity(r, reg);
    if (positionOverride && reg.all_of<TransformComponent>(e))
        reg.get<TransformComponent>(e).position = *positionOverride;
    std::fprintf(stderr, "[Prefab] instantiated %s\n", path.c_str());
    return e;
}
