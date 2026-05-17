#include "physics.h"
#include "components.h"

// Jolt headers are confined to this translation unit. Jolt.h MUST be first.
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <thread>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstdarg>
#include <cstdio>
#include <cstdint>

// ── Object / broad-phase layers (standard Jolt two-layer setup) ─────────────
namespace Layers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING     = 1;
    static constexpr JPH::ObjectLayer NUM        = 2;
}

namespace BroadPhaseLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr unsigned int NUM = 2;
}

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer l) const override {
        return (l == Layers::NON_MOVING) ? BroadPhaseLayers::NON_MOVING
                                         : BroadPhaseLayers::MOVING;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override { return "Layer"; }
#endif
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer obj, JPH::BroadPhaseLayer bp) const override {
        if (obj == Layers::NON_MOVING) return bp == BroadPhaseLayers::MOVING;
        return true; // MOVING collides with everything
    }
};

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        if (a == Layers::NON_MOVING) return b == Layers::MOVING;
        return true; // MOVING vs anything
    }
};

namespace {
    inline JPH::Vec3 toJ(const glm::vec3& v) { return JPH::Vec3(v.x, v.y, v.z); }
    inline glm::vec3 toG(JPH::Vec3 v) { return glm::vec3(v.GetX(), v.GetY(), v.GetZ()); }

    // TransformComponent stores Euler degrees applied X then Y then Z (see
    // components.h getMatrix). Build the matching quaternion the same way.
    JPH::Quat eulerDegToJQuat(const glm::vec3& deg) {
        glm::mat4 r(1.0f);
        r = glm::rotate(r, glm::radians(deg.x), glm::vec3(1, 0, 0));
        r = glm::rotate(r, glm::radians(deg.y), glm::vec3(0, 1, 0));
        r = glm::rotate(r, glm::radians(deg.z), glm::vec3(0, 0, 1));
        glm::quat q = glm::quat_cast(r);
        return JPH::Quat(q.x, q.y, q.z, q.w);
    }

    glm::vec3 jQuatToEulerDeg(JPH::Quat q) {
        glm::quat g(q.GetW(), q.GetX(), q.GetY(), q.GetZ());
        glm::mat4 m = glm::mat4_cast(g);
        float ex, ey, ez;
        glm::extractEulerAngleXYZ(m, ex, ey, ez); // inverse of the X*Y*Z build
        return glm::degrees(glm::vec3(ex, ey, ez));
    }

    void TraceImpl(const char* fmt, ...) {
        va_list list; va_start(list, fmt);
        char buf[1024]; std::vsnprintf(buf, sizeof(buf), fmt, list);
        va_end(list);
        std::fprintf(stderr, "[Jolt] %s\n", buf);
    }

#ifdef JPH_ENABLE_ASSERTS
    // Log Jolt assertions instead of breaking/aborting. Returning false tells
    // Jolt not to trigger a breakpoint — without a custom handler a fired
    // JPH_ASSERT calls abort() (the "Debug Error! abort() has been called"
    // dialog) with no debugger attached. Non-fatal asserts are the right
    // default for a dev build; the message still reaches the console.
    bool AssertFailedImpl(const char* expr, const char* msg,
                          const char* file, JPH::uint line) {
        std::fprintf(stderr, "[Jolt] ASSERT %s:%u: (%s) %s\n",
                     file ? file : "?", line, expr ? expr : "?",
                     msg ? msg : "");
        return false;
    }
#endif

    bool finite3(const glm::vec3& v) {
        return !glm::any(glm::isnan(v)) && !glm::any(glm::isinf(v));
    }

    // True if the live transform diverged from what physics last wrote (i.e.
    // an editor/script/UI edit). Generous epsilons so the euler round-trip
    // jitter on write-back never reads as an external move.
    bool movedExternally(const glm::vec3& pos, const glm::vec3& rot,
                         const std::pair<glm::vec3, glm::vec3>& last) {
        return glm::distance(pos, last.first)  > 1e-3f ||
               glm::distance(rot, last.second) > 1e-2f;
    }
}

// ── PIMPL ───────────────────────────────────────────────────────────────────
struct PhysicsWorld::Impl {
    std::unique_ptr<JPH::TempAllocatorImpl>     temp;
    std::unique_ptr<JPH::JobSystemThreadPool>   jobs;
    BPLayerInterfaceImpl                        bpLayer;
    ObjectVsBroadPhaseLayerFilterImpl           objVsBp;
    ObjectLayerPairFilterImpl                   objVsObj;
    JPH::PhysicsSystem                          system;
    std::vector<JPH::BodyID>                    bodies;
    // Last pose physics wrote to each entity's TransformComponent. If the live
    // transform differs from this, something outside physics (editor gizmo,
    // script, UI) moved it — we teleport the Jolt body to match instead of
    // letting the next step overwrite the edit.
    std::unordered_map<std::uint32_t, std::pair<glm::vec3, glm::vec3>> lastPose;
    bool                                        initialized = false;
};

PhysicsWorld::PhysicsWorld()  = default;
PhysicsWorld::~PhysicsWorld() { cleanup(); }

void PhysicsWorld::init() {
    if (m_impl) return;
    m_impl = std::make_unique<Impl>();

    JPH::RegisterDefaultAllocator();
    JPH::Trace = TraceImpl;
#ifdef JPH_ENABLE_ASSERTS
    JPH::AssertFailed = AssertFailedImpl;
#endif
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    // Per-step scratch arena. Jolt sizes its working buffers from the
    // configured max bodies / pairs / contacts below, so even a near-empty
    // scene needs ~17 MB per Update with these limits. 64 MB gives ample
    // headroom (incl. the "spawn 1000 cubes" stress button) for a one-time
    // reservation; overflowing it returns null inside Update and aborts.
    m_impl->temp = std::make_unique<JPH::TempAllocatorImpl>(64 * 1024 * 1024);
    unsigned hw  = std::thread::hardware_concurrency();
    int threads  = (hw > 1) ? static_cast<int>(hw - 1) : 1;
    m_impl->jobs = std::make_unique<JPH::JobSystemThreadPool>(
        JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, threads);

    constexpr JPH::uint kMaxBodies          = 65536;
    constexpr JPH::uint kNumBodyMutexes     = 0;
    constexpr JPH::uint kMaxBodyPairs       = 65536;
    constexpr JPH::uint kMaxContactConstr   = 20480;
    m_impl->system.Init(kMaxBodies, kNumBodyMutexes, kMaxBodyPairs, kMaxContactConstr,
                        m_impl->bpLayer, m_impl->objVsBp, m_impl->objVsObj);
    m_impl->system.SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));
    m_impl->initialized = true;
}

void PhysicsWorld::cleanup() {
    if (!m_impl) return;
    const bool wasInit = m_impl->initialized;

    if (wasInit) {
        JPH::BodyInterface& bi = m_impl->system.GetBodyInterface();
        for (JPH::BodyID id : m_impl->bodies) {
            if (!id.IsInvalid()) { bi.RemoveBody(id); bi.DestroyBody(id); }
        }
        m_impl->bodies.clear();
    }

    // Destroy the PhysicsSystem (and jobs/temp/layer interfaces, in correct
    // reverse-declaration order) BEFORE unregistering Jolt types / deleting
    // the Factory. Doing it the other way round tears down the type/RTTI
    // registry while shapes & the system still reference it — that fires a
    // Jolt assert and aborts the process.
    m_impl.reset();

    if (wasInit) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
}

void PhysicsWorld::syncNewBodies(entt::registry& reg) {
    if (!m_impl) return;
    JPH::BodyInterface& bi = m_impl->system.GetBodyInterface();

    bool createdAny = false;
    auto view = reg.view<RigidBodyComponent, TransformComponent>();
    for (auto e : view) {
        auto& rb = view.get<RigidBodyComponent>(e);
        if (rb.bodyId != 0xFFFFFFFFu) continue; // already has a body

        auto& tf = view.get<TransformComponent>(e);

        // Build the collision shape.
        JPH::ShapeRefC shape;
        switch (rb.shape) {
            case RigidBodyComponent::Shape::Sphere: {
                JPH::SphereShapeSettings s(rb.radius);
                auto r = s.Create();
                if (r.HasError()) continue;
                shape = r.Get();
            } break;
            case RigidBodyComponent::Shape::Capsule: {
                JPH::CapsuleShapeSettings s(rb.halfHeight, rb.radius);
                auto r = s.Create();
                if (r.HasError()) continue;
                shape = r.Get();
            } break;
            case RigidBodyComponent::Shape::Box:
            default: {
                glm::vec3 he = rb.autoShapeFromScale
                    ? glm::max(glm::abs(tf.scale) * 0.5f, glm::vec3(0.01f))
                    : rb.halfExtent;
                // Convex radius MUST be strictly less than the smallest half
                // extent or Jolt builds a degenerate box (the default 0.05 m
                // radius equals the thin floor's 0.05 m half-height — that
                // makes contact math divide by ~0, and since the vcpkg Jolt
                // build enables FP exceptions, that aborts the process). 0 is
                // a valid sharp-edged box.
                JPH::BoxShapeSettings s(toJ(he), 0.0f);
                auto r = s.Create();
                if (r.HasError()) continue;
                shape = r.Get();
            } break;
        }

        const bool isStatic = rb.motion == RigidBodyComponent::Motion::Static;
        JPH::EMotionType mt =
            rb.motion == RigidBodyComponent::Motion::Static    ? JPH::EMotionType::Static :
            rb.motion == RigidBodyComponent::Motion::Kinematic ? JPH::EMotionType::Kinematic :
                                                                 JPH::EMotionType::Dynamic;
        JPH::ObjectLayer layer = isStatic ? Layers::NON_MOVING : Layers::MOVING;

        JPH::BodyCreationSettings bcs(shape, toJ(tf.position),
                                      eulerDegToJQuat(tf.rotation), mt, layer);
        bcs.mFriction    = rb.friction;
        bcs.mRestitution = rb.restitution;
        if (mt == JPH::EMotionType::Dynamic) {
            bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            bcs.mMassPropertiesOverride.mMass = (rb.mass > 0.0f) ? rb.mass : 1.0f;
        }

        JPH::BodyID id = bi.CreateAndAddBody(
            bcs, isStatic ? JPH::EActivation::DontActivate : JPH::EActivation::Activate);
        if (id.IsInvalid()) continue;

        bi.SetUserData(id, static_cast<JPH::uint64>(entt::to_integral(e)));
        rb.bodyId = id.GetIndexAndSequenceNumber();
        m_impl->bodies.push_back(id);
        m_impl->lastPose[entt::to_integral(e)] = { tf.position, tf.rotation };
        createdAny = true;
    }

    // Recommended by Jolt after adding bodies (esp. the static floor added
    // before the first Update) so the broad phase quad-tree indexes them.
    if (createdAny) m_impl->system.OptimizeBroadPhase();
}

void PhysicsWorld::removeBody(entt::registry& reg, entt::entity e) {
    if (!m_impl) return;
    auto* rb = reg.try_get<RigidBodyComponent>(e);
    if (!rb || rb->bodyId == 0xFFFFFFFFu) return;
    JPH::BodyID id(rb->bodyId);
    JPH::BodyInterface& bi = m_impl->system.GetBodyInterface();
    if (!id.IsInvalid()) { bi.RemoveBody(id); bi.DestroyBody(id); }
    auto& v = m_impl->bodies;
    v.erase(std::remove(v.begin(), v.end(), id), v.end());
    rb->bodyId = 0xFFFFFFFFu;
}

void PhysicsWorld::step(entt::registry& reg, float dt) {
    if (!m_impl || m_paused) return;
    JPH::BodyInterface& bi = m_impl->system.GetBodyInterface();

    constexpr float kFixed   = 1.0f / 60.0f;
    constexpr int   kMaxSub  = 5;

    // Pre-step: detect transforms changed by the editor gizmo / UI / scripts
    // since physics last wrote them, and teleport the Jolt body to match.
    // Without this, the post-step write-back below would immediately discard
    // any manual move ("it snaps back to where it landed").
    {
        auto sv = reg.view<RigidBodyComponent, TransformComponent>();
        for (auto e : sv) {
            auto& rb = sv.get<RigidBodyComponent>(e);
            if (rb.bodyId == 0xFFFFFFFFu ||
                rb.motion == RigidBodyComponent::Motion::Kinematic) continue;
            auto& tf  = sv.get<TransformComponent>(e);
            auto  it  = m_impl->lastPose.find(entt::to_integral(e));
            if (it == m_impl->lastPose.end()) continue;
            if (!movedExternally(tf.position, tf.rotation, it->second)) continue;

            JPH::BodyID id(rb.bodyId);
            const bool isDyn = rb.motion == RigidBodyComponent::Motion::Dynamic;
            bi.SetPositionAndRotation(
                id, toJ(tf.position), eulerDegToJQuat(tf.rotation),
                isDyn ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
            if (isDyn) // drop stale momentum so it continues from the new pose
                bi.SetLinearAndAngularVelocity(id, JPH::Vec3::sZero(),
                                                   JPH::Vec3::sZero());
            it->second = { tf.position, tf.rotation };
        }
    }

    m_accumulator += dt * m_timeScale;
    int sub = 0;
    while (m_accumulator >= kFixed && sub < kMaxSub) {
        // Drive kinematic bodies toward their TransformComponent target.
        auto kv = reg.view<RigidBodyComponent, TransformComponent>();
        for (auto e : kv) {
            auto& rb = kv.get<RigidBodyComponent>(e);
            if (rb.motion != RigidBodyComponent::Motion::Kinematic ||
                rb.bodyId == 0xFFFFFFFFu) continue;
            auto& tf = kv.get<TransformComponent>(e);
            bi.MoveKinematic(JPH::BodyID(rb.bodyId), toJ(tf.position),
                             eulerDegToJQuat(tf.rotation), kFixed);
        }

        m_impl->system.Update(kFixed, 1, m_impl->temp.get(), m_impl->jobs.get());
        m_accumulator -= kFixed;
        ++sub;
    }
    if (sub == kMaxSub) m_accumulator = 0.0f; // avoid spiral of death

    // Write Dynamic poses back into the ECS (Kinematic is ECS-driven; Static
    // only moves when edited, handled above).
    auto wv = reg.view<RigidBodyComponent, TransformComponent>();
    for (auto e : wv) {
        auto& rb = wv.get<RigidBodyComponent>(e);
        if (rb.motion != RigidBodyComponent::Motion::Dynamic ||
            rb.bodyId == 0xFFFFFFFFu) continue;
        JPH::BodyID id(rb.bodyId);
        auto& tf = wv.get<TransformComponent>(e);
        glm::vec3 p = toG(bi.GetPosition(id));
        glm::vec3 r = jQuatToEulerDeg(bi.GetRotation(id));
        if (finite3(p)) tf.position = p;
        if (finite3(r)) tf.rotation = r;
        m_impl->lastPose[entt::to_integral(e)] = { tf.position, tf.rotation };
    }
}

std::optional<RaycastHit> PhysicsWorld::raycast(const glm::vec3& origin,
                                                const glm::vec3& dir,
                                                float maxDistance) const {
    if (!m_impl) return std::nullopt;
    glm::vec3 d = glm::length(dir) > 1e-6f ? glm::normalize(dir) : glm::vec3(0, 0, -1);

    JPH::RRayCast ray{ toJ(origin), toJ(d * maxDistance) };
    JPH::RayCastResult res;
    if (!m_impl->system.GetNarrowPhaseQuery().CastRay(ray, res))
        return std::nullopt;

    RaycastHit hit;
    hit.distance = res.mFraction * maxDistance;
    JPH::Vec3 p  = ray.GetPointOnRay(res.mFraction);
    hit.point    = toG(p);

    JPH::BodyLockRead lock(m_impl->system.GetBodyLockInterface(), res.mBodyID);
    if (lock.Succeeded()) {
        const JPH::Body& body = lock.GetBody();
        hit.normal = toG(body.GetWorldSpaceSurfaceNormal(res.mSubShapeID2, p));
        hit.entity = entt::entity{ static_cast<std::uint32_t>(body.GetUserData()) };
    }
    return hit;
}

void PhysicsWorld::setGravity(const glm::vec3& g) {
    if (m_impl) m_impl->system.SetGravity(toJ(g));
}

glm::vec3 PhysicsWorld::gravity() const {
    return m_impl ? toG(m_impl->system.GetGravity()) : glm::vec3(0.0f);
}

int PhysicsWorld::bodyCount() const {
    return m_impl ? static_cast<int>(m_impl->bodies.size()) : 0;
}
